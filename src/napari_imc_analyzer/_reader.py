"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from readimc import MCDFile, TXTFile


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

    data, labels, _ = read_mcd(path, 0)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {'channel_axis': 0, 'name': labels}

    layer_type = "image"  # optional, default is "image"
    return (data, add_kwargs, layer_type)

def read_mcd(path, acquisition_id=0):
    """Read an mcd file and return the data, labels and number of acquisitions.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
    acquisition_id : int
        The acquisition id to read.

    Returns
    -------
    data : numpy array
        The data from the mcd file.
    labels : list of str
        The channel labels from the mcd file.
    num_acquisitions : int
        The number of acquisitions in the mcd file.
    """

    with MCDFile(path) as f:
        num_acquisitions = len(f.slides[0].acquisitions)
        acquisition = f.slides[0].acquisitions[acquisition_id]  # first acquisition of first slide
        data = f.read_acquisition(acquisition)
        labels = acquisition.channel_labels

        import skimage
        for i in range(len(data)):
            p2, p98 = np.percentile(data[i], (2, 98))
            data[i] = skimage.exposure.rescale_intensity(data[i], in_range=(p2, p98))

        return data, labels, num_acquisitions