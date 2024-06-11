import csv
import json
import os
import pickle
import random
from datetime import date
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tifffile as tff
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from scipy import signal, stats
from skimage import feature, filters, morphology, segmentation

from _tensorstore_zarr_reader import TensorstoreZarrReader

if TYPE_CHECKING:
    import napari

class Calcium(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        self.bp_btn = QPushButton("Batch Process (spontaneous)")
        self.bp_btn.clicked.connect(self._select_folder)
        self.layout().addWidget(self.bp_btn)

        self.canvas_traces = FigureCanvas(Figure(constrained_layout=False))
        self.axes = self.canvas_traces.figure.subplots()

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
        self.roi_analysis = None
        self.framerate = None
        self.binning = None
        self.objective = None
        self.pixel_size = None
        self.mean_connect = None
        self.img_path = None
        self.colors = []
        self.binning = None
        self.objective = None
        self.pixel_size = None
        self.magnification = None
        self.folder_list: list = []

        # batch process
        self.batch_process = False
        self.model_size = 0

    def _select_folder(self) -> None:
        '''
        allow user to select a folder to analyze all the tif file in the folder

        parameters:
        ------------------
        None
        '''
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)

        if dlg.exec_():
            folder_name = dlg.selectedFiles()[0] # list of the path to the folder selected

        self.batch_process = True

        if not folder_name.endswith("tensorstore.zarr"):
            for folder_path, _, _ in os.walk(folder_name):
                if folder_path.endswith("tensorstore.zarr"):
                    folder_name = folder_path

        r = TensorstoreZarrReader(folder_name)
        folder_path = r.path
        r_shape = r.store.shape
        total_pos = r_shape[0]

        all_pos = r.sequence.stage_positions
        self.framerate, self.binning, self.pixel_size, self.objective, self.magnification = self.extract_metadata(r)

        for pos in range(total_pos):
            rec = r.isel({'p': pos}) # shape(n frames, x, y)

            self.img_stack = rec
            self.img_path = folder_path
            self.img_name = all_pos[pos].name

            img_size = rec.shape[-1]
            self.img_size = img_size

            print(f'           Analyzing {self.img_name} at pos {pos}. shape: {rec.shape}')

            # only initiate the trained model once
            if self.model_size != img_size:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')
                self.model_unet = tf.keras.models.load_model(path, custom_objects={"K": K})
                self.model_size = img_size

        self._on_click()
        self.save_files()
        self.clear()

        # if len(self.folder_list) > 0:
        #     self.compile_data(self.folder_list[-1], "summary.txt", None, "_compiled.csv")
        #     del self.folder_list[-1]
        # reset the model
        self.model_unet = None
        self.model_size = 0
        self.folder_list = []

        self.batch_process = False
        print('Batch Processing (spontaneous activity) Done')

    def _record_folders(self, folder: str):
        """Record folder location for compilation."""
        if folder not in self.folder_list:
            self.folder_list.append(folder)

    def compile_data(self, base_folder: str, file_name: str, variable: list,
                      output_name: str) -> None:
        '''
        to compile all the data from different folders into one csv file
        options to include the line name and the variable name(s) to look for;
        otherwise the programs finds the average amplitude in all the summary.txt

        parameters:
        ------------
        base_folder: str. the name of the base folder
        file_name: str. optional
            the name of the file to pull data from
        variable: list of str. optional. Be specific!
            a list of str that the user wants from each data file
            default to average amplitude
        output_name: str
            name of the output file

        returns:
        ------------
        None
        '''
        if variable is None:
            variable = ["Total ROI", "Percent Active ROI", "Average Cell Size", "Cell Size Standard Deviation",
                        "Average Amplitude", "Amplitude Standard Deviation",
                        "Average Max Slope", "Max Slope Standard Deviation", "Average Time to Rise",
                        "Time to Rise Standard Deviation", "Average Interevent Interval (IEI)",
                        "IEI Standard Deviation", "Average Number of events", "Number of events Standard Deviation",
                        "Frequency", "Global Connectivity"]

        dir_list = []

        for (dir_path, _dir_names, file_names) in os.walk(base_folder):
            if file_name in file_names:
                dir_list.append(dir_path)

        files = []

        # traverse through all the matching files
        for dir_name in dir_list:
            with open (os.path.join(dir_name, file_name)) as file:
            # with open(dir_name + "/" + file_name) as file:
                data = {}
                data['name'] = dir_name.split(os.path.sep)[-1]
                if data['name'] == 'stimulated':
                    data['name'] = dir_name.split(os.path.sep)[-2] + '_ST'
                elif data['name'] == 'non_stimulated':
                    data['name'] = dir_name.split(os.path.sep)[-2] + '_NST'
                lines = file.readlines()
                # find the variable in the file
                for line in lines:
                    for old_var in variable:
                        if old_var.lower().strip() in line.lower():
                            items = line.split(":")
                            var = items[0].strip()

                            if var not in data:
                                data[var] = []

                            values = items[1].strip().split(" ")
                            num = values[0].strip("%")

                            if values[0] == "N/A" or values[0] == "No":
                                num = 0
                            data[var] = float(num)

                if len(data) > 1:
                    files.append(data)
                else:
                    print(f'There is no {var} mentioned in the {dir_name}. Please check again.')

        if len(files) > 0:
            # write into a new csv file
            field_names = list(data.keys())
            compile_name = base_folder + output_name

            with open(os.path.join(base_folder, compile_name), 'w', newline='') as c_file:
                writer = csv.DictWriter(c_file, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(files)
        else:
            print('no data was found. please check the folder to see if there is any matching file')  # noqa: E501

    def _on_click(self) -> None:
        '''
        once hit the "Analyze" botton, the program will load the trained
        neural network model based on the image size and call the functions
        -- segment, ROI intensity, DFF, find peaks, analyze ROI, connectivity--
        to analyze the image stacks and plot the data

        Parameter:
        -------------
        None
        '''
        background_layer = 0
        minsize = 100
        self.labels, self.label_layer, self.roi_dict = self.segment(self.img_stack, minsize, background_layer)

        if self.label_layer:
            self.roi_signal = self.calculate_ROI_intensity(self.roi_dict, self.img_stack)
            self.roi_dff, self.median, self.bg = self.calculateDFF(self.roi_signal)
            self.spike_times = self.scipy_find_peaks(self.roi_dff)
            self.roi_analysis = self.analyze_ROI(self.roi_dff, self.spike_times, self.framerate)
            self.mean_connect = self.get_mean_connect(self.roi_dff, self.spike_times)

            self.plot_values(self.roi_dff, self.labels, self.label_layer, self.spike_times)

    def segment(self, img_stack, minsize, background_label):
        '''
        Predict the cell bodies using the trained NN model
        and further segment after removing small holes and objects

        Parameters:
        -------------
        img_stack: ndarray. shape=(# of frames, # of rows, # of colomns) array of the images
        minsize: float or int. the threshold to determine if the predicted cell body is too small
        background_label: int. the value set to be the background

        Return:
        -------------
        labels: ndarray. shape= shape of img_stack. a labeled matrix of segmentations of the same type as markers
        label_layer: ndarray. the label/segmentation layer image
        roi_dict: dict. a dictionary of label-position pairs
        '''
        img_norm = np.max(img_stack, axis=0) / np.max(img_stack)
        img_predict = self.model_unet.predict(img_norm[np.newaxis, :, :])[0, :, :]

        if np.max(img_predict) > 0.3:
            # the prediction layer shows the prediction of the NN
            self.prediction_layer = self.viewer.add_image(img_predict, name='Prediction')

            # use Otsu's method to find the cooridnates of the cell bodies
            th = filters.threshold_otsu(img_predict)
            img_predict_th = img_predict > th
            img_predict_remove_holes_th = morphology.remove_small_holes(img_predict_th, area_threshold=minsize * 0.3)
            img_predict_filtered_th = morphology.remove_small_objects(img_predict_remove_holes_th, min_size=minsize)
            distance = ndi.distance_transform_edt(img_predict_filtered_th)
            local_max = feature.peak_local_max(distance,
                                               min_distance=10,
                                               footprint=np.ones((15, 15)),
                                               labels=img_predict_filtered_th)

            # create masks over the predicted cell bodies and add a segmentation layer
            local_max_mask = np.zeros_like(img_predict_filtered_th, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            markers = morphology.label(local_max_mask)
            labels = segmentation.watershed(-distance, markers, mask=img_predict_filtered_th)
            roi_dict, labels = self.getROIpos(labels, background_label)
            label_layer = self.viewer.add_labels(labels, name='Segmentation', opacity=1)
        else:
            if self.batch_process:
                print(f'There were no cells detected in <{self.img_name}>')
            else:
                self.general_msg('No ROI', 'There were no cells detected')
            labels, label_layer, roi_dict = None, None, None

        return labels, label_layer, roi_dict

    def getROIpos(self, labels, background_label):
        '''
        to get the positions of the labels without the background labels

        Parameters:
        -------------
        labels: ndarray. shape=(# of rows in image, # of col in image). an array of the labels created after prediciton and segmentation
        background_label: int. the value set to be background

        Returns:
        -------------
        roi_dict: dict. a dictionary of label-position (a list of lists) pairs
        labels: ndarray. shape=(img_size, img_size) updated labels without the small ROIs
        '''
        # sort the labels and filter the unique ones
        u_labels = np.unique(labels)

        # create a dict for the labels
        roi_dict = {}
        for u in u_labels:
            roi_dict[u.item()] = []

        # record the coordinates for each label
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                roi_dict[labels[x, y]].append([x, y])

        # delete any background labels
        del roi_dict[background_label]

        area_dict, roi_to_delete = self.get_ROI_area(roi_dict, 100)

        # delete roi in label layer and dict
        for r in roi_to_delete:
            coords_to_delete = np.array(roi_dict[r]).T.tolist()
            labels[tuple(coords_to_delete)] = 0
            roi_dict[r] = []

        # move roi in roi_dict after removing some labels
        for r in range(1, (len(roi_dict) - len(roi_to_delete) + 1)):
            i = 1
            while not roi_dict[r]:
                roi_dict[r] = roi_dict[r + i]
                roi_dict[r + i] = []
                i += 1

        # delete extra roi keys
        for r in range((len(roi_dict) - len(roi_to_delete) + 1), (len(roi_dict) + 1)):
            del roi_dict[r]

        # update label layer with new roi
        for r in roi_dict:
            roi_coords = np.array(roi_dict[r]).T.tolist()
            labels[tuple(roi_coords)] = r
        return roi_dict, labels

    def get_ROI_area(self, roi_dict: dict, threshold: float):
        '''
        to get the areas of each ROI in the ROI_dict

        Parameters:
        ------------
        roi_dict: dict. the dictionary of the segmented ROIs
        threshold: float or int. The value below which the segmentation would be considered small

        Returns:
        -----------
        area: dict. a dictionary of the length of the coordinates (?) in the roi_dict
        small_roi: list. a list of small rois if their coordinates are smaller than the set threshold
        '''
        area = {}
        small_roi = []
        for r in roi_dict:
            area[r] = len(roi_dict[r])
            if area[r] < threshold:
                small_roi.append(r)
        return area, small_roi

    def calculate_ROI_intensity(self, roi_dict: dict, img_stack) -> dict:
        '''
        calculate the average intensity of each roi

        parameters:
        --------------
        roi_dict: dict. a dictionary of label (int)-coordinates (list of (x,y) coords) pairs
        img_stack: ndarray. shape=(# frames, # rows, # cols). an ndarray of the image

        returns:
        ---------------
        f: dict. the label (int)-intensity (a list of averaged intensity across
            all the pixels with the same label at each frame) pair segmemted
        '''
        f = {}
        for r in roi_dict:
            f[r] = np.zeros(img_stack.shape[0])
            roi_coords = np.array(roi_dict[r]).T.tolist()
            for z in range(img_stack.shape[0]):
                img_frame = img_stack[z, :, :]
                f[r][z] = np.mean(img_frame[tuple(roi_coords)])
        return f

    def calculateDFF(self, roi_signal: dict) -> dict:
        '''
        to calculate the change in intensity compared to the background fluorescence signal

        parameters:
        --------------
        roi_signal: dict. the label (int)-intensity (a list of averaged intensity \
            across all the pixels with the same label at each frame) pair segmemted

        returns:
        --------------
        dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        median: dict. a dictionary of label(int)-median signal pair
        bg: dict. a dictionary of label(int)- bg pixel value pair

        '''
        dff = {}
        median = {}
        bg = {}
        for n in roi_signal:
            background, median[n] = self.calculate_background(roi_signal[n], 200)
            bg[n] = background.tolist()
            dff[n] = (roi_signal[n] - background) / background
            dff[n] = dff[n] - np.min(dff[n])
        return dff, median, bg

    def calculate_background(self, f, window):
        '''
        calculate the background fluorescence intensity based on the average of a specific number of
            windows at the beginning

        parameters:
        -------------
        f: array. list of averaged intensity at each frame for pixels with the same label
        window: int. the size of the window to average the pixel intensity from.

        returns:
        -------------
        background: array. the pixels in the specified window that is below mean intensity
        median: the median of the pixels in the specified window
        '''

        background = np.zeros_like(f)
        background[0] = f[0]
        median = [background[0]]
        for y in range(1, len(f)):
            x = y - window
            if x < 0:
                x = 0
            lower_quantile = f[x:y] <= np.median(f[x:y])
            background[y] = np.mean(f[x:y][lower_quantile])
            median.append(np.median(f[x:y]))
        return background, median

    def plot_values(self, dff, labels, layer, spike_times) -> None:
        '''
        generate plots for dff, labeled_ROI, layers, and spike times

        parameters:
        --------------
        dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        labels: ndarray. a labeled matrix of segmentations of the same type as markers
        layer: ndarray. the label/segmentation layer image
        spike_times: dict. a dictionary of label-position pairs

        returns:
        --------------
        None

        '''
        for i in range(1, np.max(labels) + 1):
            color = layer.get_color(i)
            color = (color[0], color[1], color[2], color[3])
            self.colors.append(color)

        roi_to_plot = []
        colors_to_plot = []
        for i, r in enumerate(spike_times):
            if len(spike_times[r]) > 0:
                roi_to_plot.append(r)
                colors_to_plot.append(self.colors[i])

        if len(roi_to_plot) > 0:
            num_roi_to_plot, new_colors = self._random_pick(roi_to_plot, colors_to_plot, 10)

            self.axes.set_prop_cycle(color=new_colors)
            self.axes_just_traces.set_prop_cycle(color=new_colors)

            dff_max = np.zeros(len(num_roi_to_plot))
            for dff_index, dff_key in enumerate(num_roi_to_plot):
                dff_max[dff_index] = np.max(dff[dff_key])
            height_increment = max(dff_max)

            y_pos = []
            for height_index, d in enumerate(num_roi_to_plot):
                y_pos.append(height_index * (1.2 * height_increment))
                self.axes_just_traces.plot(dff[d] + height_index * (1.2 * height_increment))
                self.axes.plot(dff[d] + height_index * (1.2 * height_increment))
                if len(spike_times[d]) > 0:
                    self.axes.plot(spike_times[d], dff[d][spike_times[d]] + height_index * (1.2 * height_increment),
                                   ms=2, color='k', marker='o', ls='')
                self.canvas_traces.draw_idle()
                self.canvas_just_traces.draw_idle()
            self.axes.set_yticks(y_pos, labels=num_roi_to_plot)
        else:
            if self.batch_process:
                print(f'No calcium events were detected for any ROIs in <{self.img_name}>')
            else:
                self.general_msg('No activity', 'No calcium events were detected for any ROI')

    def _random_pick(self, og_list, color_list, num):
        '''
        to randomly pick num of roi to plot the calcium traces
        '''
        num_f = np.min([num, len(og_list)])
        final_list = random.sample(og_list, num_f)
        final_list.sort()
        new_color = []
        for i, index in enumerate(og_list):
            if index in final_list:
                new_color.append(color_list[i])

        return final_list, new_color

    # scipy find peaks (the new method)
    def scipy_find_peaks(self, roi_dff: dict, prom_pctg=0.35):
        '''
        using scipy.find_peaks for the peak detection instead of the old template-matching methods

        parameter:
        --------------
        roi_diff: dict. dictionary of label (int) - dff (dff at each frame) pair
        prom_pct: float. prominence to use in the find_peaks method

        returns:
        -------------
        spike_times: dict. label(int) - spike frame (list) pair
        '''
        spike_times = {}
        for roi in roi_dff:
            prominence = np.mean(roi_dff[roi]) * prom_pctg
            peaks, _ = signal.find_peaks(roi_dff[roi], prominence=prominence)
            spike_times[roi] = list(peaks)

        return spike_times

    def extract_metadata(self, r: TensorstoreZarrReader)->float:
        '''
        extract information from the metadata
        '''
        exposure = r.sequence.channels[0].exposure
        framerate = 1 / exposure

        binning = int(2048 / self.img_size)

        pixel_size = 0.325

        objective = 20

        magnification = 1

        return framerate, binning, pixel_size, objective, magnification

    def analyze_ROI(self, roi_dff: dict, spk_times: dict, framerate):
        '''
        to analyze the labeled ROI

        parameters:
        --------------
        roi_dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        spk_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)

        returns:
        --------------
        roi_analysis: dict. label - dict pair (amplitude, peak_indices,
            base_indices, spike_times, time_to_rise, max_slope)
        framerate: float. 1/ exposure
        '''

        amplitude_info = self.get_amplitude(roi_dff, spk_times)
        time_to_rise = self.get_time_to_rise(amplitude_info, framerate)
        max_slope = self.get_max_slope(roi_dff, amplitude_info)
        iei = self.analyze_IEI(spk_times, framerate)
        roi_analysis = amplitude_info

        for r in roi_analysis:
            # roi_analysis[r]['spike_times'] = spk_times[r]
            roi_analysis[r]['time_to_rise'] = time_to_rise[r]
            roi_analysis[r]['max_slope'] = max_slope[r]
            roi_analysis[r]['IEI'] = iei[r]

        return roi_analysis

    def get_amplitude(self, roi_dff: dict, spk_times: dict, deriv_threhold=0.01, reset_num=17, neg_reset_num=2, total_dist=40):
        '''
        find the locations (frames) of the peaks, with the peak indices and base indices

        parameters:
        --------------
        roi_dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        spk_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)
        deriv_threshold: float. optional.
        reset_num: int. optional.
            the threshold to stop the searching for the index
        neg_rest_num: int. optional.
            the threshold to stop the subsearching to address the collision with other spikes
        total_dist: int. optional.
            the maximum search that the program will attempt to search for the index

        returns:
        --------------
        amplitude_info: dict. label (int) - dict[amplitude, peak_indices, base_indices]

        '''
        amplitude_info = {}

        # for each ROI
        for r in spk_times:
            amplitude_info[r] = {}
            amplitude_info[r]['amplitudes'] = []
            amplitude_info[r]['peak_indices'] = []
            amplitude_info[r]['base_indices'] = []

            if len(spk_times[r]) > 0:
                dff_deriv = np.diff(roi_dff[r]) # the difference between each spike

                # for each spike in the ROI
                for i in range(len(spk_times[r])):
                    # Search for starting index for current spike
                    searching = True
                    under_thresh_count = 0
                    total_count = 0
                    start_index = spk_times[r][i] # the frame for the first spike

                    if start_index > 0:
                        while searching:
                            start_index -= 1
                            total_count += 1

                            # If collide with a new spike
                            if start_index in spk_times[r]:
                                subsearching = True
                                negative_count = 0

                                while subsearching:
                                    start_index += 1
                                    if start_index < len(dff_deriv):
                                        if dff_deriv[start_index] < 0:
                                            negative_count += 1

                                        else:
                                            negative_count = 0

                                        if negative_count == neg_reset_num:
                                            subsearching = False
                                    else:
                                        subsearching = False

                                break

                            # if the difference is below threshold
                            if dff_deriv[start_index] < deriv_threhold:
                                under_thresh_count += 1
                            else:
                                under_thresh_count = 0

                            # stop searching for starting index
                            if under_thresh_count >= reset_num or start_index == 0 or total_count == total_dist:
                                searching = False

                    # Search for ending index for current spike
                    searching = True
                    under_thresh_count = 0
                    total_count = 0
                    end_index = spk_times[r][i]

                    if end_index < (len(dff_deriv) - 1):
                        while searching:
                            end_index += 1
                            total_count += 1

                            # If collide with a new spike
                            if end_index in spk_times[r]:
                                subsearching = True
                                negative_count = 0
                                while subsearching:
                                    end_index -= 1
                                    if dff_deriv[end_index] < 0:
                                        negative_count += 1
                                    else:
                                        negative_count = 0
                                    if negative_count == neg_reset_num:
                                        subsearching = False
                                break
                            if dff_deriv[end_index] < deriv_threhold:
                                under_thresh_count += 1
                            else:
                                under_thresh_count = 0

                            # NOTE: changed the operator from == to >=
                            if under_thresh_count >= reset_num or end_index >= (len(dff_deriv) - 1) or \
                                    total_count == total_dist:
                                searching = False

                    # Save data
                    spk_to_end = roi_dff[r][spk_times[r][i]:(end_index + 1)]
                    start_to_spk = roi_dff[r][start_index:(spk_times[r][i] + 1)]
                    try:
                        amplitude_info[r]['amplitudes'].append(np.max(spk_to_end) - np.min(start_to_spk))
                        amplitude_info[r]['peak_indices'].append(int(spk_times[r][i] + np.argmax(spk_to_end)))
                        amplitude_info[r]['base_indices'].append(int(spk_times[r][i] -
                                                                    (len(start_to_spk) - (np.argmin(start_to_spk) + 1))))
                    except ValueError:
                        pass

        return amplitude_info

    def get_time_to_rise(self, amplitude_info: dict, framerate: float):
        '''
        get a list of time to rise for each peak throughout all the frames

        parameters:
        ---------------
        amplitude_info: dict. label (int) - dict[amplitude, peak_indices, base_indices]
        framerate: float. 1 / exposure

        returns:
        ---------------
        time_to_rise: dict. label (int) - list of time (frames/framerate) or frames (peak_index-base_index + 1) pair

        '''
        time_to_rise = {}
        for r in amplitude_info:
            time_to_rise[r] = []
            if len(amplitude_info[r]['peak_indices']) > 0:
                for i in range(len(amplitude_info[r]['peak_indices'])):
                    peak_index = amplitude_info[r]['peak_indices'][i]
                    base_index = amplitude_info[r]['base_indices'][i]
                    frames = peak_index - base_index + 1
                    if framerate:
                        time = frames / framerate  # frames * (seconds/frames) = seconds
                        time_to_rise[r].append(time)
                    else:
                        time_to_rise[r].append(frames)

        return time_to_rise

    def get_max_slope(self, roi_dff: dict, amplitude_info: dict):
        '''
        calculate the maximum slope of each peak of each label

        parameters:
        --------------
        roi_diff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        amplitude_info: dict. label (int) - dict[amplitude, peak_indices, base_indices]

        returns:
        --------------
        max_slope: dict. label (int) - list of maximum slope for each peak

        '''
        max_slope = {}
        for r in amplitude_info:
            max_slope[r] = []
            dff_deriv = np.diff(roi_dff[r])
            if len(amplitude_info[r]['peak_indices']) > 0:
                for i in range(len(amplitude_info[r]['peak_indices'])):
                    peak_index = amplitude_info[r]['peak_indices'][i]
                    base_index = amplitude_info[r]['base_indices'][i]
                    slope_window = dff_deriv[base_index:(peak_index + 1)]
                    max_slope[r].append(np.max(slope_window))

        return max_slope

    def analyze_IEI(self, spk_times: dict, framerate: float):
        '''
        calculate the inter-event interval (IEI)

        parameters:
        --------------
        spk_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)
        framerate: float. 1 / exposure

        returns:
        --------------
        iei: dict. label (int)- IEI_time or IEI_frame (float)
        '''
        iei = {}
        for r in spk_times:
            iei[r] = []

            if len(spk_times[r]) > 1:
                iei_frames = np.diff(np.array(spk_times[r]))
                if framerate:
                    iei[r] = iei_frames / framerate # in seconds
                else:
                    iei[r] = iei_frames
        return iei

    def analyze_active(self, spk_times: dict):
        '''
        calculate the percentage of active cell bodies

        parameters:
        -------------
        spk_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)

        returns:
        -------------
        active: float. percentage of active cells (with mspikes detected)

        '''
        active = 0
        for r in spk_times:
            if len(spk_times[r]) > 0:
                active += 1
        active = (active / len(spk_times)) * 100
        return active

    def get_mean_connect(self, roi_dff: dict, spk_times: dict):
        '''
        calculate functional connectivity described in Afshar Saber et al. 2018.

        parameters:
        -------------
        roi_dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        spk_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)

        returns:
        -------------
        mean_connect: float. the mean connectivity among all ROIs
        '''
        A = self.get_connect_matrix(roi_dff, spk_times)

        if A is not None:
            if len(A) > 1:
                mean_connect = np.median(np.sum(A, axis=0) - 1) / (len(A) - 1)
            else:
                mean_connect = 'N/A - Only one active ROI'
        else:
            mean_connect = 'No calcium events detected'

        return mean_connect

    def get_connect_matrix(self, roi_dff: dict, spk_times: dict):
        '''
        to calculate a matrix of synchronizatiion index described in Patel et al. 2015

        parameters:
        --------------
        roi_dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        spk_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)

        returns:
        --------------
        connect_matrix: NDarray. shape =(N, N), where N= total number of active ROIs
            a matrix of synchronization index between each peak of the ROIs
        '''
        # a list of ROI labels when it has spikes through out the frames
        active_roi = [r for r in spk_times if len(spk_times[r]) > 0]

        if len(active_roi) > 0:
            phases = {}
            for r in active_roi:
                phases[r] = self.get_phase(len(roi_dff[r]), spk_times[r])

            connect_matrix = np.zeros((len(active_roi), len(active_roi)))
            for i, r1 in enumerate(active_roi):
                for j, r2 in enumerate(active_roi):
                    connect_matrix[i, j] = self.get_sync_index(phases[r1], phases[r2])
        else:
            connect_matrix = None

        return connect_matrix

    def get_sync_index(self, x_phase: list, y_phase: list):
        '''
        to calculate the pair-wise synchronization index of the two ROIs

        parameters:
        --------------
        x_phase: list. list of instantaneous phase (float) over time for ROI x
        y_phase: list. list of instantaneous phase (float) over time for ROI y

        returns:
        --------------
        sync_index: float. the pair-wise synchronization index of the two ROIs

        '''
        phase_diff = self.get_phase_diff(x_phase, y_phase)
        sync_index = np.sqrt((np.mean(np.cos(phase_diff)) ** 2) + (np.mean(np.sin(phase_diff)) ** 2))

        return sync_index

    def get_phase_diff(self, x_phase: list, y_phase: list):
        '''
        to calculate the absolute phase difference between two calcium
            traces x and y phases from two different ROIs

        parameters:
        ---------------
        x_phase: list. list of instantaneous phase (float) over time for ROI x
        y_phase: list. list of instantaneous phase (float) over time for ROI y

        returns:
        ---------------
        phase_diff: numpy array. the absolute phase difference between the two given phases
        '''
        x_phase = np.array(x_phase)
        y_phase = np.array(y_phase)
        phase_diff = np.mod(np.abs(x_phase - y_phase), (2 * np.pi))

        return phase_diff # Numpy array

    def get_phase(self, total_frames: int, spks: list):
        '''
        calculate the instantaneous phase between each frame that contains the peak

        parameters:
        --------------
        total_frames: int. total frames of the image
        spks: list of int. frames at which the peaks occur

        returns:
        --------------
        phase: list. a list of instantaneous phase (float) over time for each ROI
        '''
        spikes = spks.copy()
        if len(spikes) == 0 or spikes[0] != 0:
            spikes.insert(0, 0)
        if spikes[-1] != (total_frames - 1):
            spikes.append(total_frames - 1)

        phase = []
        for k in range(len(spikes) - 1):
            t = spikes[k]

            while t < spikes[k + 1]:
                instant_phase = (2 * np.pi) * ((t - spikes[k]) / \
                                               (spikes[k+1] - spikes[k])) + (2 * np.pi * k)
                phase.append(instant_phase)
                t += 1
        phase.append(2 * np.pi * (len(spikes) - 1))

        return phase # Python list

    def add_num_to_img(self, img, roi_dict):
        # the centers of each ROI
        roi_centers = {}

        for roi_number, roi_coords in roi_dict.items():
            center = np.mean(roi_coords, axis=0)
            roi_centers[roi_number] = (int(center[1]), int(center[0]))

        img_w_num = img.copy()
        for r in roi_dict:
            draw = ImageDraw.Draw(img_w_num)
            font = ImageFont.truetype('segoeui.ttf', 12)
            pos = roi_centers[r]
            bbox = draw.textbbox(pos, str(r), font=font)
            draw.rectangle(bbox, fill="grey")
            draw.text(pos, str(r), font=font, fill="white")

        return img_w_num

    def save_files(self) -> None:
        '''
        to generate files for the analysis on the image

        parameters:
        ---------------
        None.

        returns:
        ---------------
        None.
        '''
        if self.roi_dict:
            save_path = self.img_path[0:-8]
            today = date.today().strftime("%y%m%d")

            # add date to the folder generated
            save_path = save_path + "_" + today
            # create the folder
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            print(f'files saved in {save_path}')

            # Raw signal
            # columns = number of segmented ROIs
            # rows (shape = num of frames)= averaged raw signals of all
            #   pixels with the same label at each frame
            raw_signal = np.zeros([len(self.roi_signal[list(self.roi_signal.keys())[0]]), len(self.roi_signal)])
            for i, r in enumerate(self.roi_signal):
                raw_signal[:, i] = self.roi_signal[r]

            with open(save_path + '/raw_signal.csv', 'w', newline='') as signal_file:
                writer = csv.writer(signal_file)
                writer.writerow(self.roi_signal.keys())
                for i in range(raw_signal.shape[0]):
                    writer.writerow(raw_signal[i, :])

            # signal corrected based on the background signal
            # columns = number of segmented ROIs
            # rows (shape= num of frames) = corrected signals based on
            #   the background signal at each frame for each ROI
            dff_signal = np.zeros([len(self.roi_dff[list(self.roi_dff.keys())[0]]), len(self.roi_dff)])
            for i, r in enumerate(self.roi_dff):
                dff_signal[:, i] = self.roi_dff[r]

            with open(save_path + '/dff.csv', 'w', newline='') as dff_file:
                writer = csv.writer(dff_file)
                writer.writerow(self.roi_dff.keys())
                for i in range(dff_signal.shape[0]):
                    writer.writerow(dff_signal[i, :])

            # the median background fluorescence
            with open(save_path + '/medians.json', 'w') as median_file:
                json.dump(self.median, median_file, indent="")

            # the background fluorescence signal
            with open(save_path + '/background.json', 'w') as bg_file:
                json.dump(self.bg, bg_file, indent="")

            # the label-frame of peaks pairs
            with open(save_path + '/spike_times.pkl', 'wb') as spike_file:
                pickle.dump(self.spike_times, spike_file)

            self.canvas_traces.print_png(save_path + '/traces.png')
            self.canvas_just_traces.print_png(save_path + '/traces_no_detections.png')

            label_array = np.stack((self.label_layer.data,) * 4, axis=-1).astype(float)
            for i in range(1, np.max(self.labels) + 1):
                i_coords = np.asarray(label_array == [i, i, i, i]).nonzero()
                label_array[(i_coords[0], i_coords[1])] = self.colors[i - 1]

            im = Image.fromarray((label_array*255).astype(np.uint8))
            bk_im = Image.new(im.mode, im.size, "black")
            bk_im.paste(im, im.split()[-1])
            bk_im_num = self.add_num_to_img(bk_im, self.roi_dict)
            bk_im_num.save(save_path + '/ROIs.png')

            # the centers of each ROI
            roi_centers = {}
            for roi_number, roi_coords in self.roi_dict.items():
                center = np.mean(roi_coords, axis=0)
                roi_centers[roi_number] = (int(center[0]), int(center[1]))

            with open(save_path + '/roi_centers.pkl', 'wb') as roi_file:
                pickle.dump(roi_centers, roi_file)

            cs_dict, _ = self.cell_size(self.roi_dict, self.binning, self.pixel_size, self.objective, self.magnification)
            roi_data = self.all_roi_data(self.roi_analysis, cs_dict, self.spike_times, self.framerate, self.img_stack.shape[0])
            with open(save_path + '/roi_data.csv', 'w', newline='') as roi_data_file:
                writer = csv.writer(roi_data_file, dialect='excel')
                fields = ['ROI', 'cell_size (um)', '# of events', 'frequency (num of events/s)',
                        'average amplitude', 'amplitude SEM', 'average time to rise', 'time to rise SEM',
                        'average max slope', 'max slope SEM',  'InterEvent Interval', 'IEI SEM']
                writer.writerow(fields)
                writer.writerows(roi_data)

            # prediction layer
            self.prediction_layer.save(save_path + '/prediction.tif')

            self.generate_summary(save_path, self.roi_analysis, self.spike_times, '/summary.txt', self.roi_dict, False)

        else:
            if self.batch_process:
                print(f'Cannot save data for <{self.img_name}>')
            else:
                self.general_msg('No ROI', 'Cannot save data')

    def generate_summary(self, save_path:str, roi_analysis: dict,
                         spike_times: dict, file_name: str, roi_dict: dict,
                         evk_group: bool) -> None:
        '''
        to generate a summary of the graphs that include the average and std of
        amplitudes, time_to_rise, IEI, and mean connectivity

        parameters:
        ------------
        save_path: str.  the prefix of the tif file name
        '''
        _, cs_arr = self.cell_size(roi_dict, self.binning, self.pixel_size, self.objective, self.magnification)
        avg_cs = float(np.mean(cs_arr, axis=0)[1])
        std_cs = float(np.std(cs_arr, axis=0)[1])

        total_amplitude = []
        total_time_to_rise = []
        total_max_slope = []
        total_IEI = []
        total_num_events = []

        for r in roi_analysis:
            if len(roi_analysis[r]['amplitudes']) > 0:
                total_amplitude.extend(roi_analysis[r]['amplitudes'])
                total_time_to_rise.extend(roi_analysis[r]['time_to_rise'])
                total_max_slope.extend(roi_analysis[r]['max_slope'])
                if len(spike_times[r]) > 0:
                    total_num_events.append(len(spike_times[r]))
            if len(roi_analysis[r]['IEI']) > 0:
                total_IEI.extend(roi_analysis[r]['IEI'])

        if any(spike_times.values()):
            avg_amplitude = np.mean(np.array(total_amplitude))
            std_amplitude = np.std(np.array(total_amplitude))
            avg_max_slope = np.mean(np.array(total_max_slope))
            std_max_slope = np.std(np.array(total_max_slope))
            # get the average num of events
            avg_num_events = np.mean(np.array(total_num_events))
            std_num_events = np.std(np.array(total_num_events))
            avg_time_to_rise = np.mean(np.array(total_time_to_rise))
            # avg_time_to_rise = f'{avg_time_to_rise} {units}'
            std_time_to_rise = np.std(np.array(total_time_to_rise))
            if len(total_IEI) > 0:
                avg_IEI = np.mean(np.array(total_IEI))
                # avg_IEI = f'{avg_IEI} {units}'
                std_IEI = np.std(np.array(total_IEI))
            else:
                avg_IEI = 'N/A - Only one event per ROI'
        else:
            avg_amplitude = 'No calcium events detected'
            avg_max_slope = 'No calcium events detected'
            avg_time_to_rise = 'No calcium events detected'
            avg_IEI = 'No calcium events detected'
            avg_num_events = 'No calcium events detected'
        percent_active = self.analyze_active(spike_times)

        with open(save_path + file_name, 'w') as sum_file:
            units = "seconds" if self.framerate else "frames"
            sum_file.write(f'File: {self.img_path}\n')
            if self.framerate:
                sum_file.write(f'Framerate: {self.framerate} fps\n')
            else:
                sum_file.write('No framerate detected\n')
            sum_file.write(f'Total ROI: {len(roi_dict)}\n')
            sum_file.write(f'Percent Active ROI (%): {percent_active}\n')

            # NOTE: include cell size in the summary text file
            sum_file.write(f'Average Cell Size(um): {avg_cs}\n')
            sum_file.write(f'\tCell Size Standard Deviation: {std_cs}\n')

            sum_file.write(f'Average Amplitude: {avg_amplitude}\n')

            if len(total_amplitude) > 0:
                sum_file.write(f'\tAmplitude Standard Deviation: {std_amplitude}\n')
            sum_file.write(f'Average Max Slope: {avg_max_slope}\n')
            if len(total_max_slope) > 0:
                sum_file.write(f'\tMax Slope Standard Deviation: {std_max_slope}\n')
            sum_file.write(f'Average Time to Rise ({units}): {avg_time_to_rise}\n')
            if len(total_time_to_rise) > 0:
                sum_file.write(f'\tTime to Rise Standard Deviation: {std_time_to_rise}\n')
            sum_file.write(f'Average Interevent Interval (IEI) ({units}): {avg_IEI}\n')
            if len(total_IEI) > 0:
                sum_file.write(f'\tIEI Standard Deviation: {std_IEI}\n')
            sum_file.write(f'Average Number of events: {avg_num_events}\n')
            if len(total_num_events) > 0:
                sum_file.write(f'\tNumber of events Standard Deviation: {std_num_events}\n')
                if self.framerate:
                    frequency = avg_num_events/(self.img_stack.shape[0]/self.framerate)
                    sum_file.write(f'Average Frequency (num_events/s): {frequency}\n')
                else:
                    if len(self.img_stack) > 3:
                        frame = self.img_stack.shape[1]
                    else:
                        frame = self.img_stack.shape[0]
                    sum_file.write(f'Frequency (num_events/frame): {avg_num_events/frame}\n')

            if not evk_group:
                sum_file.write(f'Global Connectivity: {self.mean_connect}')

    def cell_size(self, roi_dict: dict, binning: int, pixel_size: float, objective:int, magnification: float) -> dict:
        '''
        calculate the cell size of each labeled cell

        parameters:
        -------------
        roi_dict: dict. label-pixel(pos) pair

        return:
        -------------
        cs_dict: dict. label-cell_size pair
        '''
        cs_dict = {}

        for r in roi_dict:
            cs_dict[r] = len(roi_dict[r]) * binning * pixel_size / (objective*magnification) # pixel to um

        cs_arr = np.array(list(cs_dict.items()))

        return cs_dict, cs_arr

    def all_roi_data(self, roi_analysis: dict, cell_size: dict, spk_times: dict, framerate: float, total_frames: int) -> list:
        '''
        to compile data for all the ROIs (including non-active ones) in one file

        parameters:
        ------------
        roi_analysis: dict. ROI - amplitude, peak_indices, base_indices, time_to_rise, max_slope, IEI
        cell_size: dict. ROI- cell size in um
        spk_times: dict. ROI- peak indices
        framerate: float. frames per second
        total_frames: int. total frames recorded

        return:
        -----------
        roi_data: Array. the cell_size, num_events, frequency, amplitude, time_to_rise, max_slope, IEI of each ROI in the recording
        '''
        num_roi = len(roi_analysis.keys())
        if len(cell_size.keys()) == num_roi and len(spk_times.keys()) == num_roi:
            roi_data = np.zeros((num_roi, 12))
            recording_time = total_frames/framerate

            for i, r in enumerate(roi_analysis):
                roi_data[i, 0] = r
                roi_data[i, 1] = cell_size[r]
                num_e = len(spk_times[r])
                roi_data[i, 2] = num_e
                roi_data[i, 3] = num_e / recording_time
                ## TODO: 3/4: how to best save these data on a time series
                roi_data[i, 4] = np.mean(roi_analysis[r]['amplitudes'])
                roi_data[i, 5] = stats.sem(roi_analysis[r]['amplitudes'])
                # roi_data[i, 5] = roi_analysis[r]['amplitudes']
                roi_data[i, 6] = np.mean(roi_analysis[r]['time_to_rise'])
                roi_data[i, 7] = stats.sem(roi_analysis[r]['time_to_rise'])
                # roi_data[i, 7] = roi_analysis[r]['time_to_rise']
                roi_data[i, 8] = np.mean(roi_analysis[r]['max_slope'])
                roi_data[i, 9] = stats.sem(roi_analysis[r]['max_slope'])
                # roi_data[i, 9] = roi_analysis[r]['max_slope']
                roi_data[i, 10] = np.mean(roi_analysis[r]['IEI'])
                roi_data[i, 11] = stats.sem(roi_analysis[r]['IEI'], nan_policy='omit')
        else:
            print('please make sure that the number of ROIs in each dictionary is the same')

        return roi_data

    # Taken from napari-calcium plugin by Federico Gasparoli
    def general_msg(self, message_1: str, message_2: str) -> None:
        '''
        Generate message for the viewer after analysis

        parameters:
        --------------
        message_1: str. message to show
        message_2: str. message to show

        returns:
        --------------
        None
        '''
        msg = QMessageBox()
        # msg.setStyleSheet("QLabel {min-width: 250px; min-height: 30px;}")
        msg_info_1 = f'<p style="font-size:18pt; color: #4e9a06;">{message_1}</p>'
        msg.setText(msg_info_1)
        msg_info_2 = f'<p style="font-size:15pt; color: #000000;">{message_2}</p>'
        msg.setInformativeText(msg_info_2)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def clear(self):
        '''
        to clear off the image and the analysis for the next image

        parameters:
        ---------------
        None.

        Returns:
        ---------------
        None
        '''
        i = len(self.viewer.layers)-1
        num_layer = -1 if self.blue_file is None else 0
        while i > num_layer:
            self.viewer.layers.pop(i)
            i -= 1

        if not self.batch_process:
            self.model_unet = None

        self.img_stack = None
        self.img_name = None
        self.img_size = None
        self.labels = None
        self.label_layer = None
        self.prediction_layer = None
        self.roi_dict = None
        self.roi_signal = None
        self.roi_dff = None
        self.median = None
        self.bg = None
        self.spike_times = None
        self.roi_analysis = None
        self.framerate = None
        self.binning = None
        self.mean_connect = None
        self.img_path = None
        self.colors = []
        self.binning = None
        self.objective = None
        self.magnification = None
        self.pixel_size = None

        self.axes.cla()
        self.canvas_traces.draw_idle()
