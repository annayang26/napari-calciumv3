import importlib.resources
import pathlib
from napari_plugin_engine import napari_hook_implementation
import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QMainWindow

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, segmentation, morphology, feature
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import json
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
import csv
import pandas as pd


class Calcium(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        self.file_path = ""

        # batch_btn = QPushButton("Batch Process")
        # batch_btn.clicked.connect(self._batch_process)
        # self.layout().addWidget(batch_btn)

        btn = QPushButton("Analyze")
        btn.clicked.connect(self._on_click)
        self.layout().addWidget(btn)

        self.canvas_traces = FigureCanvas(Figure(constrained_layout=False))
        self.axes = self.canvas_traces.figure.subplots()
        self.layout().addWidget(self.canvas_traces)

        self.canvas_just_traces = FigureCanvas(Figure(constrained_layout=False))
        self.axes_just_traces = self.canvas_just_traces.figure.subplots()

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_files)
        self.layout().addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        self.layout().addWidget(self.clear_btn)

        self.img_stack = None
        self.img_name = None
        self.labels = None
        self.label_layer = None
        self.model_unet = None
        self.prediction_layer = None
        self.roi_dict = None
        self.roi_signal = None
        self.roi_dff = None
        self.median = None
        self.bg = None
        self.spike_times = None
        self.max_correlations = None
        self.max_cor_templates = None
        self.roi_analysis = None
        self.framerate = None
        self.mean_connect = None
        self.img_path = None
        self.colors = []

    def _on_click(self):
        '''
        once hit the "Analyze" botton, the program will load the trained 
        neural network model based on the image size and call the functions
        -- segment, ROI intensity, DFF, find peaks, analyze ROI, connectivity-- 
        to analyze the image stacks and plot the data

        Parameter:
        -------------
        None

        Returns:
        -------------
        None

        '''
        self.img_stack = self.viewer.layers[0].data
        self.img_name = self.viewer.layers[0].name
        self.img_path = self.viewer.layers[0].source.path
        print("working")
        # img_size = self.img_stack.shape[1]

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')

        # self.model_unet = load_model(path, custom_objects={"K": K})
        # background_layer = 0
        # minsize = 100
        # self.labels, self.label_layer, self.roi_dict = self.segment(self.img_stack, minsize, background_layer)

        # if self.label_layer:
        #     self.roi_signal = self.calculate_ROI_intensity(self.roi_dict, self.img_stack)
        #     self.roi_dff = self.calculateDFF(self.roi_signal)

        #     spike_templates_file = 'spikes.json'
        #     self.spike_times = self.find_peaks(self.roi_dff, spike_templates_file, 0.85, 0.80)
        #     self.roi_analysis, self.framerate = self.analyze_ROI(self.roi_dff, self.spike_times)
        #     self.mean_connect = self.get_mean_connect(self.roi_dff, self.spike_times)

        #     self.plot_values(self.roi_dff, self.labels, self.label_layer, self.spike_times)
            # print('ROI average prediction:', self.get_ROI_prediction(self.roi_dict, self.prediction_layer.data))


def calcium_analyze():
    return Calcium()

