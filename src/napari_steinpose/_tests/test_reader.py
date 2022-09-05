import numpy as np

from napari_steinpose import napari_get_reader_mcd


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = str(tmp_path / "myfile.npy")
    original_data = np.random.rand(20, 20)
    np.save(my_test_file, original_data)

    # try to read it back in
    reader = napari_get_reader_mcd(my_test_file)
    assert reader is None