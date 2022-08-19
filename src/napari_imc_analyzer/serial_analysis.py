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

def run_cellpose(image_path, cellpose_model, output_path, scaling_factor=1,
                 diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 clear_border=True, channel_to_segment=None, channel_helper=None,
                 channel_measure=None, channel_measure_names=None, properties=None,
                 options_file=None, proj_fun=np.max):
    """Run cellpose on image.
    
    Parameters
    ----------
    image_path : str or Path
        path to image
    cellpose_model : cellpose model instance
    output_path : str or Path
        path to output folder
    scaling_factor : int
        scaling factor for image (not implemented)
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
    properties = list of str, default None
        list of types of properties to compute. Any of 'intensity', 'perimeter', 'shape', 'position', 'moments'
    options_file: str or Path, default None
        path to yaml options file for cellpose
    proj_fun: function
        function used to compute projection, default np.max

    Returns
    -------
    cellpose_output : list of arrays
        list of segmented images
    props : pandas dataframe
        properties of segmented cells for the last analyzed image
    """

    if not isinstance(image_path, list):
        image_path = [image_path]

    if properties is None:
        properties = []

    if channel_to_segment is None:
        raise ValueError("channel_to_segment must be specified")
    else:
        channel_to_segment = np.array(channel_to_segment) - 1

    channels = [0, 0]
    image = []
    image_measure=None
    for p in image_path:
        with MCDFile(p) as f:
            acquisition = f.slides[0].acquisitions[0]  # first acquisition of first slide
            data = f.read_acquisition(acquisition)

        cur_image = proj_fun(data[channel_to_segment, :, :], axis=0)
        if channel_helper is not None:
            channel_helper = np.array(channel_helper) - 1
            image_nuclei = proj_fun(data[channel_helper, :, :], axis=0)
            cur_image = np.stack([cur_image, image_nuclei], axis=0)
            channels = [1, 2]
        image.append(cur_image)

        if channel_measure is not None:
            if image_measure is None:
                image_measure = []
            cur_image_measure = np.moveaxis(data[channel_measure, :, :], 0, -1)
            image_measure.append(cur_image_measure)

    if image_measure is None:
        image_measure = [None]*len(image)
    
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
    
    # save output
    for im, im_m, p in zip(cellpose_output, image_measure, image_path):
        
        props=None
        if len(properties) > 0:
            props = compute_props(
                    label_image=im,
                    intensity_image=im_m,
                    output_path=output_path,
                    image_name=p,
                    properties=properties,
                    channel_names=channel_measure_names
                    )

        if output_path is not None:
            output_path = Path(output_path)
            save_path = output_path.joinpath(p.stem+'_mask.tif')
            skimage.io.imsave(save_path, im, check_contrast=False)

    return cellpose_output, props


def compute_props(
    label_image, intensity_image, output_path=None,
    image_name=None, properties=None, channel_names=None):
    """Compute properties of segmented image.
    
    Parameters
    ----------
    label_image : array
        image with labeled cells
    intensity_image : array
        image with intensity values
    output_path : str or Path
        path to output folder
    image_name : str or Path
        either path to image or image name
    properties = list of str, default None
        list of types of properties to compute. Any of 'intensity', 'perimeter', 'shape', 'position', 'moments'
    channel_names: list of str, default None
        names of channel(s) in which to measure intensity
    """
    
    if (image_name is not None) and (output_path is not None):
        image_name = Path(image_name)
        output_path = Path(output_path).joinpath('tables')
        if not output_path.exists():
            output_path.mkdir(parents=True)
    
    if properties is None:
        properties = []
        
    if intensity_image is None:
        if "intensity" in properties:
            warnings.warn("Computing intensity features but no intensity image provided. Result will be zero.")
        intensity_image = np.zeros(label_image.shape)[:,:,np.newaxis]
        
    props = regionprops_table(
        image=intensity_image[:,:,-1], labels=label_image,
        size='size' in properties,
        perimeter='perimeter' in properties,
        shape='shape' in properties,
        position='position' in properties,
        moments='moments' in properties,
        intensity=False,
        )

    if 'intensity' in properties:
        intensity_measure = sk_regionprops_table(
            label_image=label_image, intensity_image=intensity_image,
            properties=['max_intensity', 'mean_intensity', 'min_intensity'])
        intensity_measure = pd.DataFrame(intensity_measure)
        if channel_names is not None:
            for ind, c in enumerate(channel_names):
                intensity_measure.rename(
                    columns={
                        f'mean_intensity-{ind}': f'mean_intensity-{c}',
                        f'min_intensity-{ind}': f'min_intensity-{c}',
                        f'max_intensity-{ind}': f'max_intensity-{c}'}, inplace=True)
        props = pd.concat([props, intensity_measure], axis=1)

    if output_path is not None:
        props.to_csv(output_path.joinpath(image_name.stem+'_props.csv'), index=False)

    return props


def load_props(output_path, image_name):
    """Load properties for an analyzed image.
    
    Parameters
    ----------
    output_path : str or Path
        path to output folder
    image_name : str or Path
        either path to image or image name
    
    Returns
    -------
    props : pandas dataframe
        properties of segmented cells
    """

    # get file name
    image_name = Path(image_name)
    output_path = Path(output_path).joinpath('tables')

    # load properties
    props_path = Path(output_path).joinpath(image_name.stem+'_props.csv')
    props=None
    if props_path.exists():
        props = pd.read_csv(props_path)

    return props

def load_allprops(output_path):
    """Load all properties files for a given folder.
    
    Parameters
    ----------
    output_path : str or Path
        path to output folder

    Returns
    -------
    all_props : pandas dataframe
        properties of segmented cells in all images
    
    """

    # get file name
    output_path = Path(output_path).joinpath('tables')
    if not output_path.exists():
        return None
    table_names = list(output_path.glob('*_props.csv'))
    
    all_props = []
    for p in table_names:
        props = pd.read_csv(p)
        props['name'] = p.stem
        all_props.append(props)
    all_props = pd.concat(all_props)

    all_props.to_csv(output_path.joinpath('summary.csv'), index=False)

    return all_props

