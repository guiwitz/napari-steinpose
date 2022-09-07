from qtpy.QtWidgets import (QWidget, QVBoxLayout,QFileDialog, QPushButton,
QSpinBox, QDoubleSpinBox, QLabel, QGridLayout, QHBoxLayout, QGroupBox, QComboBox, QTabWidget,
QCheckBox, QListWidget, QAbstractItemView)
from qtpy.QtCore import Qt
from napari.layers import Image

from .folder_list_widget import FolderList
from .imc_analysis import (run_cellpose, export_for_steinbock, create_panel_file,
create_images_file)
from ._reader import read_mcd

from pathlib import Path
import skimage.io
import numpy as np
from cellpose import models
import yaml
from .imc_analysis import measure_intensities_steinbock, measure_neighborhood_steinbock, measure_region_props_steinbock


class SteinposeWidget(QWidget):
    """
    Implementation of a napari widget allowing to select a folder filled with images and 
    segment them using cellpose.
    """
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.cellpose_model_path = None
        self.cellpose_model = None
        self.output_folder = None
        self.options_file_path = None
        self.num_acquisitions = 0
        self.proj = {'mean': np.mean, 'max': np.max, 'min': np.min, 'median': np.median}

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # segmentation tab
        self.segmentation = QWidget()
        self._segmentation_layout = QVBoxLayout()
        self.segmentation.setLayout(self._segmentation_layout)
        self.tabs.addTab(self.segmentation, 'Segmentation')

        # channels tab
        self.channels_tab = QWidget()
        self._channels_tab_layout = QVBoxLayout()
        self.channels_tab.setLayout(self._channels_tab_layout)
        self.tabs.addTab(self.channels_tab, 'Channels')

        # options tab
        self.options_tab = QWidget()
        self._options_tab_layout = QVBoxLayout()
        self.options_tab.setLayout(self._options_tab_layout)
        self.tabs.addTab(self.options_tab, 'Options')

        # export tab
        self.export_tab = QWidget()
        self._export_tab_layout = QGridLayout()
        self.export_tab.setLayout(self._export_tab_layout)
        self.tabs.addTab(self.export_tab, 'Export')
        self._export_tab_layout.setAlignment(Qt.AlignTop)

        #/////// Segmentation tab /////////
        self.files_group = VHGroup('File selection', orientation='G')
        self._segmentation_layout.addWidget(self.files_group.gbox)

        self.files_group.glayout.addWidget(QLabel("List of images"), 0,0, 1, 2)

        self.file_list = FolderList(napari_viewer)
        self.files_group.glayout.addWidget(self.file_list, 1, 0, 1, 2)

        self.slider_acquisition = QSpinBox()
        self.slider_acquisition.setMinimum(0)
        self.slider_acquisition.setMaximum(0)
        self.files_group.glayout.addWidget(QLabel('ROI'), 2, 0, 1, 1)
        self.files_group.glayout.addWidget(self.slider_acquisition, 2, 1, 1, 1)

        self.folder_group = VHGroup('Folder selection')
        self._segmentation_layout.addWidget(self.folder_group.gbox)

        self.btn_select_file_folder = QPushButton("Select data folder")
        self.folder_group.glayout.addWidget(self.btn_select_file_folder)

        self.btn_select_output_folder = QPushButton("Select output folder")
        self.folder_group.glayout.addWidget(self.btn_select_output_folder)

        self.qcbox_model_choice = QComboBox(visible=True)
        self.qcbox_model_choice.addItems([
            'custom', 'cyto', 'cyto2', 'nuclei', 'tissuenet', 'CP'])
        self.folder_group.glayout.addWidget(self.qcbox_model_choice)

        self.btn_select_cellpose_model = QPushButton("Select custom cellpose model file")
        self.folder_group.glayout.addWidget(self.btn_select_cellpose_model)

        self.run_group = VHGroup('Run analysis', orientation='G')
        self._segmentation_layout.addWidget(self.run_group.gbox)

        self.btn_run_on_current = QPushButton("Run on current image")
        self.run_group.glayout.addWidget(self.btn_run_on_current, 0, 0, 1, 2)

        self.btn_run_on_folder = QPushButton("Run on folder")
        self.run_group.glayout.addWidget(self.btn_run_on_folder, 1, 0, 1, 2)

        self.check_usegpu = QCheckBox('Use GPU')
        self.run_group.glayout.addWidget(self.check_usegpu, 2, 0, 1, 1)

        self.check_run_steinbock = QCheckBox('Run Steinbock post-processing')
        self.run_group.glayout.addWidget(self.check_run_steinbock, 2, 1, 1, 1)

        self.mainoptions_group = VHGroup('Main options', orientation='G')
        self._segmentation_layout.addWidget(self.mainoptions_group.gbox)

        self.diameter_label = QLabel("Diameter", visible=False)
        self.mainoptions_group.glayout.addWidget(self.diameter_label, 0, 0, 1, 1)
        self.spinbox_diameter = QSpinBox(visible=False)
        self.spinbox_diameter.setValue(30)
        self.spinbox_diameter.setMaximum(1000)
        self.mainoptions_group.glayout.addWidget(self.spinbox_diameter, 0, 1, 1, 1)

        #/////// Channels tab /////////
        self.channel_merge_group = VHGroup('Channels to merge', orientation='G')
        self._channels_tab_layout.addWidget(self.channel_merge_group.gbox)
        
        self.channel_merge_group.glayout.addWidget(QLabel('Channels for cell'), 0, 0, 1, 1)
        self.qlist_merge_cell = QListWidget()
        self.qlist_merge_cell.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_merge_group.glayout.addWidget(self.qlist_merge_cell, 0,1,1,1)

        self.channel_merge_group.glayout.addWidget(QLabel('Channels for nuclei'), 1, 0, 1, 1)
        self.qlist_merge_nuclei = QListWidget()
        self.qlist_merge_nuclei.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_merge_group.glayout.addWidget(self.qlist_merge_nuclei, 1,1,1,1)

        self.qcbox_projection_method = QComboBox()
        self.qcbox_projection_method.addItems(['max', 'median', 'mean', 'min'])
        self.qcbox_projection_method.setCurrentIndex(0)
        self.channel_merge_group.glayout.addWidget(QLabel('Projection type'), 2, 0, 1, 1)
        self.channel_merge_group.glayout.addWidget(self.qcbox_projection_method, 2,1,1,1)

        #/////// Options tab /////////
        self._options_tab_layout.setAlignment(Qt.AlignTop)
        self.options_group = VHGroup('Segmentation Options', orientation='G')
        self._options_tab_layout.addWidget(self.options_group.gbox)

        self.flow_threshold_label = QLabel("Flow threshold")
        self.options_group.glayout.addWidget(self.flow_threshold_label, 3, 0, 1, 1)
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setMaximum(10)
        self.flow_threshold.setMinimum(-10)
        self.flow_threshold.setValue(0.4)
        self.options_group.glayout.addWidget(self.flow_threshold, 3, 1, 1, 1)

        self.cellprob_threshold_label = QLabel("Cell probability threshold")
        self.options_group.glayout.addWidget(self.cellprob_threshold_label, 4, 0, 1, 1)
        self.cellprob_threshold = QDoubleSpinBox()
        self.cellprob_threshold.setSingleStep(0.1)
        self.cellprob_threshold.setMaximum(10)
        self.cellprob_threshold.setMinimum(-10)
        self.cellprob_threshold.setValue(0.0)
        self.options_group.glayout.addWidget(self.cellprob_threshold, 4, 1, 1, 1)

        self.check_clear_border = QCheckBox('Clear labels on border')
        self.check_clear_border.setChecked(True)
        self.options_group.glayout.addWidget(self.check_clear_border, 5, 0, 1, 1)

        self.expand_label = QLabel("Expansion (px)")
        self.options_group.glayout.addWidget(self.expand_label, 6, 0, 1, 1)
        self.spinbox_expand = QSpinBox()
        self.spinbox_expand.setValue(0)
        self.spinbox_expand.setMaximum(20)
        self.options_group.glayout.addWidget(self.spinbox_expand, 6, 1, 1, 1)

        self.check_hpf = QCheckBox('Hot pixel filter')
        self.options_group.glayout.addWidget(self.check_hpf, 7, 0, 1, 1)
        self.spinbox_hpf = QSpinBox()
        self.spinbox_hpf.setValue(0)
        self.spinbox_hpf.setMaximum(1000)
        self.options_group.glayout.addWidget(self.spinbox_hpf, 7, 1, 1, 1)

        self.btn_select_options_file = QPushButton("Select options yaml file")
        self.btn_select_options_file.setToolTip(("Select a yaml file containing special options for "
            "the cellpose model eval segmentation function"))
        self.options_group.glayout.addWidget(self.btn_select_options_file, 8, 0, 1, 1)

        self.config_group = VHGroup('Configuration', orientation='G')
        self._options_tab_layout.addWidget(self.config_group.gbox)

        self.btn_run_export_config = QPushButton("Export config")
        self.config_group.glayout.addWidget(self.btn_run_export_config, 0, 0, 1, 2)

        self.btn_run_import_config = QPushButton("Import config")
        self.config_group.glayout.addWidget(self.btn_run_import_config, 1, 0, 1, 2)

        #!!!!!!!!!!!! Steinbock !!!!!!!!!!!!!!!
        self.btn_export_tiffs = QPushButton("Export tiffs")
        self._export_tab_layout.addWidget(self.btn_export_tiffs, 0, 0, 1, 2)

        self.btn_run_steinbock_postproc = QPushButton("Run Steinbock postproc")
        self._export_tab_layout.addWidget(self.btn_run_steinbock_postproc, 1, 0, 1, 2)

        self.intensity_group = VHGroup('Intensity', orientation='G')
        self._export_tab_layout.addWidget(self.intensity_group.gbox, 2, 0, 1, 2)
        self.check_intensities = QCheckBox("Measure intensities")
        self.check_intensities.setChecked(True)
        self.intensity_group.glayout.addWidget(self.check_intensities, 0, 0, 1, 1)
        self.qcbox_intensity_stat = QComboBox()
        self.qcbox_intensity_stat.addItems([
            'mean', 'median', 'min', 'max'])
        self.qcbox_intensity_stat.setCurrentText('mean')
        self.intensity_group.glayout.addWidget(QLabel('Intensity stat'), 1, 0, 1, 1)
        self.intensity_group.glayout.addWidget(self.qcbox_intensity_stat, 1, 1, 1, 1)

        self.regions_group = VHGroup('Regions', orientation='G')
        self._export_tab_layout.addWidget(self.regions_group.gbox, 3, 0, 1, 2)
        self.check_regionprops = QCheckBox("Measure regions")
        self.check_regionprops.setChecked(True)
        self.regions_group.glayout.addWidget(self.check_regionprops, 0, 0, 1, 1)

        self.neighbours_group = VHGroup('Neighbourhood', orientation='G')
        self._export_tab_layout.addWidget(self.neighbours_group.gbox, 4, 0, 1, 2)
        self.check_neighbours = QCheckBox("Measure neighbours")
        self.check_neighbours.setChecked(True)
        self.neighbours_group.glayout.addWidget(self.check_neighbours, 0, 0, 1, 1)
        self.spin_max_dist_neighbours = QSpinBox()
        self.spin_max_dist_neighbours.setValue(15)
        self.spin_max_dist_neighbours.setMaximum(100)
        self.neighbours_group.glayout.addWidget(QLabel('Max distance'), 1, 0, 1, 1)
        self.neighbours_group.glayout.addWidget(self.spin_max_dist_neighbours, 1, 1, 1, 1)
        self.qcbox_neighbour_type = QComboBox()
        self.qcbox_neighbour_type.addItems([
            'centroids', 'borders', 'expand'])
        self.qcbox_neighbour_type.setCurrentText('centroids')
        self.neighbours_group.glayout.addWidget(QLabel('Neighbour type'), 2, 0, 1, 1)
        self.neighbours_group.glayout.addWidget(self.qcbox_neighbour_type, 2, 1, 1, 1)


        self.add_connections()

    def add_connections(self):
        """Add callbacks"""

        self.btn_select_file_folder.clicked.connect(self._on_click_select_file_folder)
        self.btn_select_cellpose_model.clicked.connect(self._on_click_select_cellpose_model)
        self.btn_select_options_file.clicked.connect(self._on_click_select_options_file)
        self.btn_select_output_folder.clicked.connect(self._on_click_select_output_folder)
        self.file_list.currentItemChanged.connect(self._on_select_file)
        self.slider_acquisition.valueChanged.connect(self._on_slider_acquisition_change)
        self.btn_run_on_current.clicked.connect(self._on_click_run_on_current)
        self.btn_run_on_folder.clicked.connect(self._on_click_run_on_folder)
        self.qcbox_model_choice.currentTextChanged.connect(self._on_change_modeltype)
        self.viewer.layers.events.inserted.connect(self._on_change_layers)

        self.qlist_merge_cell.itemClicked.connect(self._on_change_merge_cell_selection)
        self.qlist_merge_nuclei.itemClicked.connect(self._on_change_merge_nuclei_selection)
        self.qcbox_projection_method.currentTextChanged.connect(self._on_change_merge_cell_selection)
        self.qcbox_projection_method.currentTextChanged.connect(self._on_change_merge_nuclei_selection)

        self.btn_export_tiffs.clicked.connect(self._on_export_tiffs)

        self.spinbox_expand.valueChanged.connect(self._on_change_expand)
        self.btn_run_steinbock_postproc.clicked.connect(self.run_steinbock_postproc)
        self.btn_run_export_config.clicked.connect(self._on_click_export_configuration)
        self.btn_run_import_config.clicked.connect(self._on_click_import_configuration)

    def _on_change_merge_cell_selection(self):
        """Create projection of multiple channels for cells using selected channels"""

        curr_proj = self.proj[self.qcbox_projection_method.currentText()]

        channel_to_merge_cell = []
        merged_cell_array = np.zeros((20,20), dtype=np.uint8)
        if len(self.qlist_merge_cell.selectedItems()) > 0:
            channel_to_merge_cell = [x.row() for x in self.qlist_merge_cell.selectedIndexes()]
            merged_cell_array = curr_proj(np.stack([self.viewer.layers[x].data for x in channel_to_merge_cell], axis=0), axis=0)
       
        self.viewer.layers.events.inserted.disconnect(self._on_change_layers)
        if 'merged_cell' in [x.name for x in self.viewer.layers]:
            self.viewer.layers['merged_cell'].data = merged_cell_array
        else:
            self.viewer.add_image(
                merged_cell_array,
                name='merged_cell',
                colormap='magenta',
                blending='additive')
        
        self.viewer.layers.events.inserted.connect(self._on_change_layers)

    def _on_change_merge_nuclei_selection(self):
        """Create projection of multiple channels for nuclei using selected channels"""

        curr_proj = self.proj[self.qcbox_projection_method.currentText()]

        channel_to_merge_nuclei = []
        merged_nuclei_array = np.zeros((20,20), dtype=np.uint8)
        if len(self.qlist_merge_nuclei.selectedItems()) > 0:
            channel_to_merge_nuclei = [x.row() for x in self.qlist_merge_nuclei.selectedIndexes()]
            merged_nuclei_array = curr_proj(np.stack([self.viewer.layers[x].data for x in channel_to_merge_nuclei], axis=0), axis=0)

        self.viewer.layers.events.inserted.disconnect(self._on_change_layers)
        
        if 'merged_nuclei' in [x.name for x in self.viewer.layers]:
            self.viewer.layers['merged_nuclei'].data = merged_nuclei_array
        else:
            self.viewer.add_image(
                merged_nuclei_array,
                name='merged_nuclei',
                colormap='cyan',
                blending='additive',
                )
        self.viewer.layers.events.inserted.connect(self._on_change_layers)

    def open_file(self):
        """Open file selected in list. Returns True if file was opened."""

        # clear existing layers.
        self.viewer.layers.clear()
        
        # if file list is empty stop here
        if self.file_list.currentItem() is None:
            return False
        
        # open image
        image_name = self.file_list.currentItem().text()
        image_path = self.file_list.folder_path.joinpath(image_name)

        data, _, self.num_acquisitions, names = read_mcd(image_path, self.slider_acquisition.value())
        self.viewer.add_image(data, channel_axis=0, name=names)
        self.slider_acquisition.setMaximum(self.num_acquisitions-1)

        if self.output_folder is not None:
            mask_path = Path(self.output_folder).joinpath('masks').joinpath(f'{image_path.stem}_acq_{self.slider_acquisition.value()}.tiff')
            if mask_path.exists():
                self.mask = skimage.io.imread(mask_path)
                self.viewer.add_labels(self.mask, name='mask')
            
            proj_path = Path(self.output_folder).joinpath('imgs_proj').joinpath(f'{image_path.stem}_acq_{self.slider_acquisition.value()}_proj.tiff')
            if proj_path.exists():
                proj = skimage.io.imread(proj_path)
                if proj.ndim == 2:
                    self.viewer.add_image(
                        proj,
                        name='merged_cell',
                        colormap='magenta',
                        blending='additive')
                elif proj.ndim == 3:
                    self.viewer.add_image(
                        proj[0,:,:],
                        name='merged_cell',
                        colormap='magenta',
                        blending='additive')
                    self.viewer.add_image(
                        proj[1,:,:],
                        name='merged_nuclei',
                        colormap='cyan',
                        blending='additive')

        return True

    def _on_change_expand(self):
        """Expand or shrink mask"""

        if 'mask' in [x.name for x in self.viewer.layers]:
            mask = skimage.segmentation.expand_labels(self.mask, self.spinbox_expand.value())
            self.viewer.layers['mask'].data = mask

    def _on_export_tiffs(self):
        """Export tiffs of the current acquisition as well as a panel.csv file."""
        
        if self.output_folder is None:
            self._on_click_select_output_folder()

        file_list = self.get_file_list()
        create_panel_file(mcd_path=file_list[0], export_path=self.output_folder)
        create_images_file(file_list_mcd=file_list, export_path=self.output_folder)
        for f in file_list:
            export_for_steinbock(f, self.output_folder, hpf=(self.spinbox_hpf.value() if self.check_hpf.isChecked() else None))

    def _on_slider_acquisition_change(self):

        self.open_file()

    def _on_click_select_file_folder(self):
        """Interactively select folder to analyze"""

        file_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.file_list.update_from_path(file_folder)

    def _on_click_select_output_folder (self):
        """Interactively select folder where to save results"""

        self.output_folder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))

    def _on_click_select_cellpose_model(self):
        """Interactively select cellpose model"""

        self.cellpose_model_path = QFileDialog.getOpenFileName(self, "Select model file")[0]
    
    def _on_select_file(self, current_item, previous_item):
        
        success = self.open_file()
        if not success:
            return False

    def _on_click_select_options_file(self):
        """Interactively select cellpose model"""

        self.options_file_path = QFileDialog.getOpenFileName(self, "Select options file")[0]

    def _on_click_export_configuration(self):

        channels_cell = [x.text() for x in self.qlist_merge_cell.selectedItems()]
        channels_nuclei = [x.text() for x in self.qlist_merge_nuclei.selectedItems()]

        dict_file = {
            'channels_cell' : channels_cell,
            'channels_nuclei' : channels_nuclei,
            'cellpose_model': self.qcbox_model_choice.currentText(),
            'diameter': self.spinbox_diameter.value(),
            'expand': self.spinbox_expand.value(),
            'flow_threshold': self.flow_threshold.value(),
            'cellprob_threshold': self.cellprob_threshold.value()}

        if self.output_folder is None:
            self._on_click_select_output_folder()
        with open(self.output_folder.joinpath('config.yml'), 'w') as file:
            documents = yaml.dump(dict_file, file)

    def _on_click_import_configuration(self):
        
        channel_list = [self.qlist_merge_cell.item(x).text() for x in range(self.qlist_merge_cell.count())]
        
        with open(self.output_folder.joinpath('config.yml')) as file:
            options_yml = yaml.load(file, Loader=yaml.FullLoader)
            for k, val in options_yml.items():
                if k == 'cellpose_model':
                    #model_index = [self.qcbox_model_choice.itemText(x) for x in range(self.qcbox_model_choice.count())].index(val)
                    self.qcbox_model_choice.setCurrentText(val)
                elif k == 'channels_cell':
                    [self.qlist_merge_cell.item(channel_list.index(x)).setSelected(True) for x in val]
                    self._on_change_merge_cell_selection()
                elif k == 'channels_nuclei':
                    [self.qlist_merge_nuclei.item(channel_list.index(x)).setSelected(True) for x in val]
                    self._on_change_merge_nuclei_selection()
                elif k == 'diameter':
                    self.spinbox_diameter.setValue(val)
                elif k == 'expand':
                    self.spinbox_expand.setValue(val)
                elif k == 'flow_threshold':
                    self.flow_threshold.setValue(val)
                elif k == 'cellprob_threshold':
                    self.cellprob_threshold.setValue(val)
                

    def _on_click_run_on_current(self):
        """Run cellpose on current image"""

        model_type = self.qcbox_model_choice.currentText()
        self.output_and_model_check(choose_output=False)

        image_path = self.file_list.folder_path.joinpath(self.file_list.currentItem().text())
        
        self.cellpose_model, diameter = self.get_cellpose_model(model_type=model_type)
        
        channel_to_segment, channel_helper = self.get_channels_to_use()
        curr_proj = self.proj[self.qcbox_projection_method.currentText()]

        # run cellpose
        mask = run_cellpose(
            image_path=image_path,
            cellpose_model=self.cellpose_model,
            output_path=self.output_folder,
            acquisition=self.slider_acquisition.value(),
            diameter=diameter,
            flow_threshold=self.flow_threshold.value(),
            cellprob_threshold=self.cellprob_threshold.value(),
            clear_border=self.check_clear_border.isChecked(),
            channel_to_segment=channel_to_segment,
            channel_helper=channel_helper,
            options_file=self.options_file_path,
            proj_fun=curr_proj,
            label_expand=self.spinbox_expand.value()
        )
        self.mask = mask[0]

        self.viewer.layers.events.inserted.disconnect(self._on_change_layers)

        self.viewer.add_labels(self.mask, name='mask')
        
        self.viewer.layers.events.inserted.connect(self._on_change_layers)

    def _on_click_run_on_folder(self):
        """Run cellpose on all images in folder"""

        model_type = self.qcbox_model_choice.currentText()
        self.output_and_model_check()
        
        file_list = self.get_file_list()

        self.cellpose_model, diameter = self.get_cellpose_model(model_type=model_type)

        channel_to_segment, channel_helper = self.get_channels_to_use()
        curr_proj = self.proj[self.qcbox_projection_method.currentText()]

        for f in file_list:
            for acq in range(self.num_acquisitions):
                print(f'Running cellpose on {f.name} acquisition {acq}')
                _ = run_cellpose(
                    image_path=f,
                    cellpose_model=self.cellpose_model,
                    output_path=self.output_folder,
                    acquisition=acq,
                    diameter=diameter,
                    flow_threshold=self.flow_threshold.value(),
                    cellprob_threshold=self.cellprob_threshold.value(),
                    clear_border=self.check_clear_border.isChecked(),
                    channel_to_segment=channel_to_segment,
                    channel_helper=channel_helper,
                    options_file=self.options_file_path,
                    proj_fun=curr_proj,
                    label_expand=self.spinbox_expand.value()
            )
        
        if self.check_run_steinbock.isChecked():
            self.run_steinbock_postproc()
        
    def get_file_list(self):
        
        file_list = [self.file_list.item(x).text() for x in range(self.file_list.count())]
        file_list = [f for f in file_list if f[0] != '.']
        file_list = [self.file_list.folder_path.joinpath(x) for x in file_list]
        file_list = [f for f in file_list if f.is_file()]
        
        return file_list

    def run_steinbock_postproc(self):
        """Run steinbock postprocessing on current folder"""

        file_list = self.get_file_list()
        create_panel_file(mcd_path=file_list[0], export_path=self.output_folder)
        create_images_file(file_list_mcd=file_list, export_path=self.output_folder)
        
        for f in file_list:
            export_for_steinbock(f, self.output_folder, hpf=(self.spinbox_hpf.value() if self.check_hpf.isChecked() else None))

        if self.check_intensities.isChecked():
            measure_intensities_steinbock(self.output_folder, statistic=self.qcbox_intensity_stat.currentText())
        if self.check_regionprops.isChecked():
            measure_region_props_steinbock(self.output_folder)
        if self.check_neighbours.isChecked():
            measure_neighborhood_steinbock(
                self.output_folder, dmax=self.spin_max_dist_neighbours.value(),
                neighborhood_type_name=self.qcbox_neighbour_type.currentText())


    def get_channels_to_use(self):
        """Translate selected channels in QCombox into indices.
        As the first choice is None, channels are already incremented by one
        as expected by cellpose"""
        
        channel_to_segment = None
        channel_helper = None
        if len(self.qlist_merge_cell.selectedItems()) > 0:
            channel_to_segment = [x.row() for x in self.qlist_merge_cell.selectedIndexes()]
        if len(self.qlist_merge_nuclei.selectedItems()) > 0:
            channel_helper = [x.row() for x in self.qlist_merge_nuclei.selectedIndexes()]
        
        return channel_to_segment, channel_helper

    def output_and_model_check(self, choose_output=True):
        """Check if output folder and model are set"""

        model_type = self.qcbox_model_choice.currentText()
        if (self.output_folder is None) and (choose_output):
            self._on_click_select_output_folder()
        if (self.cellpose_model_path is None) and (model_type == 'custom'):
            self._on_click_select_cellpose_model()

    def get_cellpose_model(self, model_type):
        """Get cellpose model. For non-custom model provide a model name
        in model_type. For custom models the returned diameter is None, otherwise
        it is the GUI value."""

        diameter = None
        if self.qcbox_model_choice.currentText() == 'custom':
            cellpose_model = models.CellposeModel(
                gpu=self.check_usegpu.isChecked(),
                pretrained_model=self.cellpose_model_path)
        else:
            cellpose_model = models.CellposeModel(
                gpu=self.check_usegpu.isChecked(),
                model_type=model_type)
            diameter = self.spinbox_diameter.value()
        
        return cellpose_model, diameter

    def _on_change_layers(self):
        """Update layers lists in comboxes when layer is added or removed"""

        self.qlist_merge_cell.clear()
        self.qlist_merge_cell.addItems([x.name for x in self.viewer.layers if isinstance(x, Image)])

        self.qlist_merge_nuclei.clear()
        self.qlist_merge_nuclei.addItems([x.name for x in self.viewer.layers if isinstance(x, Image)])

    def _on_change_modeltype(self):
        "if selecting non-custom model, show diameter box"

        if self.qcbox_model_choice.currentText() != 'custom':
            self.diameter_label.setVisible(True)
            self.spinbox_diameter.setVisible(True)
        else:
            self.diameter_label.setVisible(False)
            self.spinbox_diameter.setVisible(False)

class VHGroup():
    """Group box with specific layout.

    Parameters
    ----------
    name: str
        Name of the group box
    orientation: str
        'V' for vertical, 'H' for horizontal, 'G' for grid
    """

    def __init__(self, name, orientation='V'):
        self.gbox = QGroupBox(name)
        if orientation=='V':
            self.glayout = QVBoxLayout()
        elif orientation=='H':
            self.glayout = QHBoxLayout()
        elif orientation=='G':
            self.glayout = QGridLayout()
        else:
            raise Exception(f"Unknown orientation {orientation}") 

        self.gbox.setLayout(self.glayout)