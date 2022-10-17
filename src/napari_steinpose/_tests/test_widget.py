import numpy as np
from pathlib import Path
from napari_steinpose import SteinposeWidget
import pytest

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_widget_loads(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    widget = SteinposeWidget(viewer)

    assert isinstance(widget, SteinposeWidget)

def test_select_folder(make_napari_viewer):
    
    viewer = make_napari_viewer()
    widget = SteinposeWidget(viewer)

    file_folder = Path('src/napari_steinpose/_tests/data/')
    widget.file_list.update_from_path(file_folder)
    assert widget.file_list.count() == 2, "Wrong number of files added"

def test_select_file(make_napari_viewer):

    viewer = make_napari_viewer()
    widget = SteinposeWidget(viewer)

    file_folder = Path('src/napari_steinpose/_tests/data/')
    widget.file_list.update_from_path(file_folder)
    
    assert len(viewer.layers) == 0, "There should be no layers yet"
    
    widget.file_list.setCurrentRow(0)

    assert len(viewer.layers) == 48, "Wrong number of channels"

    assert widget.qlist_merge_cell.count() == 48, "Wrong number of channels in cell merge list"
    assert widget.qlist_merge_nuclei.count() == 48, "Wrong number of channels in nuclei merge list"

    assert len(widget.qlist_merge_cell.selectedItems()) == 0, "No channels should be selected"
    assert len(widget.qlist_merge_nuclei.selectedItems()) == 0, "No channels should be selected"

def test_create_merged_images(make_napari_viewer):

    viewer = make_napari_viewer()
    widget = SteinposeWidget(viewer)

    file_folder = Path('src/napari_steinpose/_tests/data/')
    widget.file_list.update_from_path(file_folder)    
    widget.file_list.setCurrentRow(0)

    cell_ch = ['ICSK1', 'ICSK2', 'ICSK3']
    nucl_ch = ['DNA1','DNA2']

    channel_list = [widget.qlist_merge_cell.item(x).text() for x in range(widget.qlist_merge_cell.count())]

    [widget.qlist_merge_cell.item(channel_list.index(x)).setSelected(True) for x in cell_ch]
    [widget.qlist_merge_nuclei.item(channel_list.index(x)).setSelected(True) for x in nucl_ch]

    widget._on_change_merge_cell_selection()
    widget._on_change_merge_nuclei_selection()

    assert viewer.layers[-1].name == 'merged_nuclei'
    assert viewer.layers[-2].name == 'merged_cell'

@pytest.mark.order(1)
def test_segmentation(make_napari_viewer):

    viewer = make_napari_viewer()
    widget = SteinposeWidget(viewer)

    file_folder = Path('src/napari_steinpose/_tests/data/')
    widget.file_list.update_from_path(file_folder)
    widget.file_list.setCurrentRow(1)

    cell_ch = ['ICSK1', 'ICSK2', 'ICSK3']
    nucl_ch = ['DNA1','DNA2']

    channel_list = [widget.qlist_merge_cell.item(x).text() for x in range(widget.qlist_merge_cell.count())]

    [widget.qlist_merge_cell.item(channel_list.index(x)).setSelected(True) for x in cell_ch]
    [widget.qlist_merge_nuclei.item(channel_list.index(x)).setSelected(True) for x in nucl_ch]

    widget._on_change_merge_cell_selection()
    widget._on_change_merge_nuclei_selection()

    widget.qcbox_model_choice.setCurrentText('cyto2')
    widget.flow_threshold.setValue(1)
    widget.cellprob_threshold.setValue(-6)
    widget.spinbox_diameter.setValue(25)
    widget._on_click_run_on_current()

    assert viewer.layers[-1].name == 'mask'
    assert viewer.layers['mask'].data.max() == 17

    # run on folder
    output_path = file_folder.joinpath('output')
    output_path.mkdir(exist_ok=True)

    widget.output_folder = output_path

    widget._on_click_run_on_folder()

    assert output_path.joinpath('imgs_proj').exists(), "Missing projection images"
    assert output_path.joinpath('masks').exists(), "Missing segmentation images"

    assert len(list(output_path.joinpath('imgs_proj').glob('*.tiff'))) == 4, "Wrong number of projection images"
    assert len(list(output_path.joinpath('masks').glob('*.tiff'))) == 4, "Wrong number of segmentation images"

@pytest.mark.order(2)
def test_steinbock(make_napari_viewer):

    viewer = make_napari_viewer()
    widget = SteinposeWidget(viewer)

    file_folder = Path('src/napari_steinpose/_tests/data/')
    widget.file_list.update_from_path(file_folder)

    output_path = file_folder.joinpath('output')
    widget.output_folder = output_path

    widget.run_steinbock_postproc()

    assert output_path.joinpath('intensities').exists(), "Missing intensity folder"
    assert output_path.joinpath('regionprops').exists(), "Missing regionprops folder"
    assert output_path.joinpath('neighbors').exists(), "Missing neihgbors folder"

    assert len(list(output_path.joinpath('intensities').glob('*.csv'))) == 4, "Wrong number of intensity files"
    assert len(list(output_path.joinpath('regionprops').glob('*.csv'))) == 4, "Wrong number of intensity files"
    assert len(list(output_path.joinpath('neighbors').glob('*.csv'))) == 4, "Wrong number of intensity files"

    assert output_path.joinpath('panel.csv').exists(), "Missing panel.csv file"
    assert output_path.joinpath('images.csv').exists(), "Missing images.csv file"