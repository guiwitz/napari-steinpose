import numpy as np
from readimc import MCDFile
import skimage
from aicsimageio import AICSImage
from pathlib import Path
import warnings

def napari_get_reader_mcd(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".mcd"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """

    '''# handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))'''

    data, labels, _, names = read_mcd(path)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {'channel_axis': 0, 'name': names}

    layer_type = "image"  # optional, default is "image"
    return (data, add_kwargs, layer_type)

def read_mcd(path, acquisition_id=0, rescale_percentile=True, planes_to_load=None):
    """Read an mcd file and return the data, labels and number of acquisitions.

    Parameters
    ----------
    path : str
        Path to file
    acquisition_id : int
        The acquisition id to read.
    rescale_percentile: bool
        rescale the intensity
    planes_to_load : int or list of int
        list of planes to load

    Returns
    -------
    data : numpy array
        The data from the mcd file.
    channels : list of str
        The channel labels from the mcd file.
    num_acquisitions : int
        The number of acquisitions in the mcd file.
    names : list of str
        The channel names from the mcd file.
    """

    if isinstance(planes_to_load, int):
        planes_to_load = [planes_to_load]

    path = Path(path)
    if path.suffix == ".mcd":
        with MCDFile(path) as f:
            num_acquisitions = get_actual_num_acquisition(f.slides[0].acquisitions)
            if acquisition_id > num_acquisitions:
                raise ValueError(f"acquisition_id {acquisition_id} is larger than the number of acquisitions {num_acquisitions}. Probably missing roi.")
            else:
                acquisition = f.slides[0].acquisitions[acquisition_id]  # first acquisition of first slide
                data = f.read_acquisition(acquisition)   
      
        names = acquisition.channel_labels
        channels = acquisition.channel_names
    
    elif path.suffix == ".tiff":
        im_aics = AICSImage(path)
        data = im_aics.get_image_data(dimension_order_out='CYX', T=acquisition_id)
        names_labels = im_aics.channel_names
        channels = [x.split('/')[0] for x in names_labels]
        names = [x.split('/')[1] for x in names_labels]
        num_acquisitions = im_aics.dims.T
    else:
        raise ValueError("File is not an mcd file nor a ome tiff file.")
    
    # fill empty target names with metal names. If metal name is empty, use unlabelled name
    unlabel_count = 1
    for i, n in enumerate(names):
        if n == '':
            if channels[i] == '':
                names[i] = 'unlabelled_' + str(unlabel_count)
                unlabel_count += 1
            else:
                names[i] = channels[i]

    if planes_to_load is not None:
        data = data[planes_to_load]
        channels = np.array(channels)[planes_to_load]
        names = np.array(names)[planes_to_load]

    if rescale_percentile is True:
        for i in range(len(data)):
            p2, p98 = np.percentile(data[i], (2, 98))
            data[i] = skimage.exposure.rescale_intensity(data[i], in_range=(p2, p98))

    return data, channels, num_acquisitions, names

def get_actual_num_acquisition(acquisitions=None, path=None):
    """
    Keep only acquisitions where number of channels is larger than zero. 
    This is to avoid acquisitions where the stage was moved but no image was taken.
    This assumes that missing acquisitions are at the end of the list
    
    Parameters
    ----------
    acquisitions : list of Acquisition
        The acquisitions to check.
    path : str
        Path to mcd file

    Returns
    -------
    num_acquisitions : int
        The number of acquisitions in the mcd file.
    
    """

    if acquisitions is None:
        if path is not None:
            path = Path(path)
        else:
            raise ValueError("Either acquisitions or path must be provided.")
        if path.suffix == ".mcd":
            with MCDFile(path) as f:
                acquisitions = f.slides[0].acquisitions
        else:
            raise ValueError("File is not an mcd file.")

    num_acquisitions_all = [x.num_channels for x in acquisitions]
    num_acquisitions = [x for x in num_acquisitions_all if x > 0]
    
    if len(num_acquisitions) < len(num_acquisitions_all):
        warnings.warn(f"Probably missing rois.")

    return len(num_acquisitions)