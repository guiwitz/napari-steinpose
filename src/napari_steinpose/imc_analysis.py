from pathlib import Path
import warnings
import skimage.io
import skimage.segmentation
from skimage.measure import regionprops_table as sk_regionprops_table
from napari_skimage_regionprops._regionprops import regionprops_table
import pandas as pd
import numpy as np
import yaml
from readimc import MCDFile
from ._reader import read_mcd
from aicsimageio.writers import OmeTiffWriter

from steinbock.measurement.intensities import IntensityAggregation
from steinbock.measurement.neighbors import NeighborhoodType
from steinbock.measurement.intensities import try_measure_intensities_from_disk
from steinbock.measurement.regionprops import try_measure_regionprops_from_disk
from steinbock.measurement.neighbors import try_measure_neighbors_from_disk
from steinbock import io

_intensity_aggregations = {
    "sum": IntensityAggregation.SUM,
    "min": IntensityAggregation.MIN,
    "max": IntensityAggregation.MAX,
    "mean": IntensityAggregation.MEAN,
    "median": IntensityAggregation.MEDIAN,
    "std": IntensityAggregation.STD,
    "var": IntensityAggregation.VAR,
}

_neighborhood_types = {
    "centroids": NeighborhoodType.CENTROID_DISTANCE,
    "borders": NeighborhoodType.EUCLIDEAN_BORDER_DISTANCE,
    "expansion": NeighborhoodType.EUCLIDEAN_PIXEL_EXPANSION,
}


def run_cellpose(image_path, cellpose_model, output_path, acquisition=0,
                 diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 clear_border=True, channel_to_segment=None, channel_helper=None,
                 channel_measure=None, channel_measure_names=None,
                 options_file=None, proj_fun=np.max, label_expand=0):
    """Run cellpose on image.
    
    Parameters
    ----------
    image_path : str or Path
        path to image
    cellpose_model : cellpose model instance
    output_path : str or Path
        path to output folder
    acquisition : int
        acquisition number
    diameter : int
        diameter of cells to segment, only useful for native cellpose models
    flow_threshold : float
        cellpose setting: maximum allowed error of the flows for each mask
    cellprob_threshold : float
        cellpose setting: pixels greater than the cellprob_threshold are used to run dynamics and determine ROIs
    clear_border : bool
        remove cells touching border
    channel_to_segment : int, default 0
        index of channel to segment, if image is multi-channel
    channel_helper : int, default 0
        index of helper nucleus channel for models using both cell and nucleus channels
    channel_measure: int or list of int, default None
        index of channel(s) in which to measure intensity
    channel_measure_names: list of str, default None
        names of channel(s) in which to measure intensity
    options_file: str or Path, default None
        path to yaml options file for cellpose
    proj_fun: function
        function used to compute projection, default np.max
    label_expand: int
        number of pixels to expand labels by

    Returns
    -------
    cellpose_output : list of arrays
        list of segmented images
    """

    if not isinstance(image_path, list):
        image_path = [image_path]

    if channel_to_segment is None:
        raise ValueError("channel_to_segment must be specified")

    channels = [0, 0]
    image = []
    for p in image_path:
        
        cur_image = create_composite_proj(p, acquisition, True, channel_to_segment, proj_fun=proj_fun)

        if channel_helper is not None:
            image_nuclei = create_composite_proj(p, acquisition, True, channel_helper, proj_fun=proj_fun)

            cur_image = np.stack([cur_image, image_nuclei], axis=0)
            channels = [1, 2]
        image.append(cur_image)
    
    # handle yaml options file
    default_options = {'diameter': diameter, 'flow_threshold': flow_threshold, 'cellprob_threshold': cellprob_threshold}
    options_yml = {}
    if options_file is not None:
        with open(options_file) as file:
            options_yml = yaml.load(file, Loader=yaml.FullLoader)
        list_of_cellpose_options = cellpose_model.eval.__code__.co_varnames
        for k in options_yml.keys():
            if k not in list_of_cellpose_options:
                raise ValueError(f'options file contains key {k} which is not in cellpose model')
    merged_options = {**default_options, **options_yml}

        
    cellpose_output = cellpose_model.eval(
        image, channels=channels, channel_axis=0,
        **merged_options
    )
    cellpose_output = cellpose_output[0]

    if clear_border is True:

        cellpose_output = [skimage.segmentation.clear_border(im) for im in cellpose_output]
        cellpose_output = [skimage.segmentation.relabel_sequential(im)[0] for im in cellpose_output]
    
    if label_expand > 0:
        cellpose_output = [skimage.segmentation.expand_labels(im, label_expand) for im in cellpose_output]
    
    # save output
    for im_proj, im_mask, p in zip(image, cellpose_output, image_path):

        if output_path is not None:
            output_path = Path(output_path)
            output_path_mask = output_path.joinpath('masks')
            output_path_mask.mkdir(parents=True, exist_ok=True)
            save_path = output_path_mask.joinpath(f'{p.stem}_acq_{acquisition}.tiff')
            skimage.io.imsave(save_path, im_mask.astype(np.uint16), check_contrast=False)

            output_path_proj = output_path.joinpath('imgs_proj')
            output_path_proj.mkdir(parents=True, exist_ok=True)
            save_path = output_path_proj.joinpath(f'{p.stem}_acq_{acquisition}_proj.tiff')
            skimage.io.imsave(save_path, im_proj, check_contrast=False)

    return cellpose_output

def create_composite_proj(mcd_path, acquisition, rescale_percentile, planes_to_load, proj_fun=np.max):
    
    cur_image, _, _, _ = read_mcd(mcd_path, acquisition_id=acquisition, rescale_percentile=rescale_percentile, planes_to_load=planes_to_load)
    cur_image = proj_fun(cur_image, axis=0)
    cur_image = (255*(cur_image / cur_image.max())).astype(np.uint8)

    return cur_image

def export_for_steinbock(path, export_path):
    """Convert mcd files to tiff files for Steinbock and create panels.csv file
    
    Parameters
    ----------
    path : str or Path
        path to mcd file
    export_path : str or Path
        path to folder where to export
    
    """

    create_panel_file(path=path, export_path=export_path)

    data, labels, num_acquisitions, names = read_mcd(path, acquisition_id=0, rescale_percentile=False)
    
    p = Path(export_path).joinpath('img')
    p.mkdir(parents=True, exist_ok=True)

    for i in range(num_acquisitions):
        data, _, _, _ = read_mcd(path, acquisition_id=i, rescale_percentile=False)
        OmeTiffWriter.save(data, p.joinpath(f'{path.stem}_acq_{i}.tiff'), dim_order="CYX")

def create_panel_file(path, export_path):
    """Create panel file for Steinbock

    Parameters
    ----------
    path : str or Path
        path to mcd file
    export_path : str or Path
        path to folder where to export
    
    """

    data, labels, num_acquisitions, names = read_mcd(path, acquisition_id=0, rescale_percentile=False)
    #panel = pd.DataFrame({'channel': names, 'name': labels})
    panel = pd.DataFrame({'channel': names, 'name': names})
    panel.to_csv(export_path.joinpath('panel.csv'), index=False)


def measure_intensities_steinbock(output_folder, statistic='mean'):
    """Measure intensities using Steinbock

    Parameters
    ----------
    output_folder : str or Path
        path to output folder
    statistic : str, default 'mean'
        statistic to use for intensity measurement. Any of 'mean', 'median', 'min', 'max'
    """
    
    img_dir = output_folder.joinpath('img')
    mask_dir = output_folder.joinpath('masks')
    intensities_dir = output_folder.joinpath('intensities')

    panel = io.read_panel(output_folder.joinpath('panel.csv'))
    channel_names = panel["name"].tolist()
    img_files = io.list_image_files(img_dir)
    mask_files = io.list_mask_files(mask_dir, base_files=img_files)
    Path(intensities_dir).mkdir(exist_ok=True)

    for img_file, mask_file, intensities in try_measure_intensities_from_disk(
        img_files,
        mask_files,
        channel_names,
        _intensity_aggregations[statistic],
        mmap=False,
    ):
        intensities_file = io._as_path_with_suffix(
            Path(intensities_dir) / img_file.name, ".csv"
        )
        io.write_data(intensities, intensities_file)

def measure_region_props_steinbock(output_folder, skimage_regionprops=None):
    """Measure region properties using Steinbock

    Parameters
    ----------
    output_folder : str or Path
        path to output folder
    """
    
    img_dir = output_folder.joinpath('img')
    mask_dir = output_folder.joinpath('masks')
    regionprops_dir = output_folder.joinpath('regionprops')

    img_files = io.list_image_files(img_dir)
    mask_files = io.list_mask_files(mask_dir, base_files=img_files)
    Path(regionprops_dir).mkdir(exist_ok=True)

    if not skimage_regionprops:
        skimage_regionprops = [
            "area",
            "centroid",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
        ]
    for img_file, mask_file, regionprops in try_measure_regionprops_from_disk(
        img_files, mask_files, skimage_regionprops, mmap=False
    ):
        regionprops_file = io._as_path_with_suffix(
            Path(regionprops_dir) / img_file.name, ".csv"
        )
        io.write_data(regionprops, regionprops_file)


def measure_neighborhood_steinbock(
    output_folder, neighborhood_type_name='centroids', dmax=15, metric='euclidean'):
    """Measure neighborhood using Steinbock

    Parameters
    ----------
    output_folder : str or Path
        path to output folder
    neighborhood_type_name : str, default 'centroids'
        neighborhood type to use. Any of 'centroids'
    dmax : int, default 15
        maximum distance to consider
    metric : str, default 'euclidean'
        'border', 'expansion', or 'euclidean'
    """
    
    img_dir = output_folder.joinpath('img')
    mask_dir = output_folder.joinpath('masks')
    neighbors_dir = output_folder.joinpath('neighbors')

    neighborhood_type_name = 'centroids'
    dmax = 15
    metric = 'euclidean'

    img_files = io.list_image_files(img_dir)
    mask_files = io.list_mask_files(mask_dir, base_files=img_files)
    Path(neighbors_dir).mkdir(exist_ok=True)

    for mask_file, neighbors in try_measure_neighbors_from_disk(
        mask_files,
        _neighborhood_types[neighborhood_type_name],
        metric=metric,
        dmax=dmax,
        #kmax=kmax,
        mmap=False,
    ):
        neighbors_file = io._as_path_with_suffix(
            Path(neighbors_dir) / Path(mask_file).name, ".csv"
        )
        io.write_neighbors(neighbors, neighbors_file)

