import numpy as np

from napari_steinpose import napari_get_reader_mcd
from napari_steinpose._reader import read_mcd


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""
    import os
    print(os.getcwd())
    print(os.listdir())

    # write some fake data using your supported file format
    my_test_file = 'src/napari_steinpose/_tests/data/test_steinpose1.tiff'

    # try to read it back in
    reader = napari_get_reader_mcd(my_test_file)
    assert reader is None

def test_read_mcd():
    """Test mcd importer with fake tiff data"""

    my_test_file = 'src/napari_steinpose/_tests/data/test_steinpose1.tiff'
    data, channels, num_acquisitions, names = read_mcd(my_test_file)
    assert data.shape == (48, 100, 100)
    assert len(channels) == 48
    assert num_acquisitions == 2
    assert len(names) == 48

def test_read_mcd_channels():
    """Test mcd importer with fake tiff data for subset of channels"""
    
    my_test_file = 'src/napari_steinpose/_tests/data/test_steinpose1.tiff'
    data, channels, num_acquisitions, names = read_mcd(my_test_file, planes_to_load=[0, 11, 25])
    assert data.shape == (3, 100, 100)
    assert len(channels) == 3
    assert num_acquisitions == 2
    assert len(names) == 3