import csv
import importlib.resources
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tifffile as tff
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage import feature, filters, io, morphology, segmentation

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

        self.viewer.window.add_dock_widget(self._evk_batch_process)
        self._evk_batch_process()

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

        # batch process
        self.batch_process = False
        self.unet_init = False
        # batch process (evoked)
        self.blue_file = None
        self.ca_file = None
        self.st_roi_signal = None
        self.st_roi_dff = None

    def _select_folder(self) -> None:
        '''
        allow user to select a folder to analyze all the tif file in the folder

        parameters:
        ------------------
        None
        '''
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)

        # NOTE: trying for one experiemnt folder
        if dlg.exec_():
            folder_names = dlg.selectedFiles() # list of the path to the folder selected

        self.batch_process = True

        # traverse through all the ome.tif files in the selected folder
        for (folder_path, _, folder_list) in os.walk(folder_names[0]):
            # for file_name in Path.iterdir(folder):
            print("dir_path: ", folder_path)
            for file_name in os.listdir(folder_path):

                if file_name.endswith(".ome.tif"):
                    file_path = os.path.join(folder_path, file_name)
                    print("file_path: ", file_path)
                    img = tff.imread(file_path, is_ome=False, is_mmstack=False)
                    self.viewer.add_image(img, name=file_name)

                    self.img_stack = self.viewer.layers[0].data
                    self.img_path = file_path
                    self.img_name = file_name

                    # only initiate the trained model once
                    if not self.unet_init:
                        img_size = self.img_stack.shape[-1]
                        dir_path = os.path.dirname(os.path.realpath(__file__))
                        path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')
                        self.model_unet = tf.keras.models.load_model(path, custom_objects={"K": K})
                        self.unet_init = True

                    print("self img stack: ", self.img_stack.shape)
                    print("self img path ", self.img_path)
                    print("self img name: ", self.img_name)

                    self._on_click()
                    self.save_files()
                    self.clear()

            print(f'{folder_path} is done batch processing')

            if self.model_unet:
                self._compile_data(folder_path)
            # reset the model
            self.model_unet = None
            self.unet_init = False

        print('Batch Processing (spontaneous activity) Done')
        self.batch_process = False

    def _compile_data(self, base_folder,
                      file_name="summary.txt", variable=None):
        '''
        to compile all the data from different folders into one csv file
        options to include the line name and the variable name(s) to look for; 
        otherwise the programs finds the average amplitude in all the summary.txt

        parameters:
        ------------
        base_folder: str. the name of the base folder
        compile_name: str. optional.
            the name of the final file that has all the data
            default to compiled_file.txt
        folder_prefix: str. optional
            the prefix of the folders that has data to analyze
            e.g. NC230802
        file_name: str. optional
            the name of the file to pull data from
            default to summary.txt
        variable: list of str. optional. Be specific!
            a list of str that the user wants from each data file
            default to average amplitude

        returns:
        ------------
        None
        '''
        if variable is None:
            variable = ["Total ROI", "Percent Active ROI", "Average Amplitude", "Amplitude Standard Deviation",
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
            result = open(dir_name + "/" + file_name)
            data = {}
            data['name'] = dir_name.split(os.path.sep)[-1][:-4]

            # find the variable in the file
            for line in result:
                for old_var in variable:
                    if old_var.lower().strip() in line.lower():
                        items = line.split(":")
                        var = items[0].strip()

                        if var not in data:
                            data[var] = []

                        values = items[1].strip().split(" ")
                        num = values[0].strip("%")

                        if values[0] == "N/A":
                            num = 0
                        data[var] = float(num)

            if len(data) > 1:
                files.append(data)
            else:
                print(f'There is no {var} mentioned in the {dir_name}. Please check again.')

        if len(files) > 0:
            # write into a new csv file
            field_names = list(data.keys())

            compile_name = os.path.basename(base_folder) + "_compile_file.csv"

            with open(base_folder + "/" + compile_name, 'w', newline='') as c_file:
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
        if not self.batch_process:
            self.img_stack = self.viewer.layers[0].data
            self.img_name = self.viewer.layers[0].name
            self.img_path = self.viewer.layers[0].source.path

            img_size = self.img_stack.shape[-1]
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')
            self.model_unet = tf.keras.models.load_model(path, custom_objects={"K": K})

        background_layer = 0
        minsize = 100
        self.labels, self.label_layer, self.roi_dict = self.segment(self.img_stack, minsize, background_layer)

        if self.label_layer:
            self.roi_signal = self.calculate_ROI_intensity(self.roi_dict, self.img_stack)
            self.roi_dff, self.median, self.bg = self.calculateDFF(self.roi_signal)

            spike_templates_file = 'spikes.json'
            self.spike_times, self.max_correlations, self.max_cor_templates = self.find_peaks(self.roi_dff, spike_templates_file, 0.85, 0.80)
            self.roi_analysis, self.framerate = self.analyze_ROI(self.roi_dff, self.spike_times)
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
        roi_dict: dict. a dictionary of label-position pairs
        labels: ndarray. shape=() updated labels without the small ROIs
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

        # print("roi_dict len:", len(roi_dict))
        area_dict, roi_to_delete = self.get_ROI_area(roi_dict, 100)
        # print("area_dict:", area_dict)

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
        # print("new roi_dict len:", len(roi_dict))
        return roi_dict, labels

    def get_ROI_area(self, roi_dict, threshold):
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

    def calculateDFF(self, roi_signal):
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
            print('Active ROI:', roi_to_plot)
            self.axes.set_prop_cycle(color=colors_to_plot)
            self.axes_just_traces.set_prop_cycle(color=colors_to_plot)

            dff_max = np.zeros(len(roi_to_plot))
            for dff_index, dff_key in enumerate(roi_to_plot):
                dff_max[dff_index] = np.max(dff[dff_key])
            height_increment = max(dff_max)

            for height_index, d in enumerate(roi_to_plot):
                self.axes_just_traces.plot(dff[d] + height_index * (1.2 * height_increment))
                self.axes.plot(dff[d] + height_index * (1.2 * height_increment))
                if len(spike_times[d]) > 0:
                    self.axes.plot(spike_times[d], dff[d][spike_times[d]] + height_index * (1.2 * height_increment),
                                   ms=2, color='k', marker='o', ls='')
                self.canvas_traces.draw_idle()
                self.canvas_just_traces.draw_idle()
        else:
            self.general_msg('No activity', 'No calcium events were detected for any ROI')

    def find_peaks(self, roi_dff: dict, template_file: str, spk_threshold: float, reset_threshold: float):
        '''
        find the spikes from the fluorescence signals

        parameters:
        --------------
        roi_dff: dict. a dictionary of label (int)-dff (dff at each frame) pair
        template_file: str. the template file for spikes
        spk_threshold: float. threshold for determining if the spike \
            is correlated with the template
        reset_threshold: float. threshold for reseting the peak finding

        returns:
        --------------
        spike_times: dict. a dictionary of label (int) - the frame at which the peak occurs (int)

        '''
        # open the template file
        f = importlib.resources.open_text(__package__, template_file)
        spike_templates = json.load(f)
        spike_times = {}
        max_correlations = {}
        max_cor_templates = {}
        max_temp_len = max([len(temp) for temp in spike_templates.values()])

        # calculate the corelation between the dff of one label at each frame
        #   with each of the template
        # iterate through each label
        for r in roi_dff:
            # print("\n", r)
            m = np.zeros((len(roi_dff[r]), len(spike_templates)))
            roi_dff_pad = np.pad(roi_dff[r], (0, (max_temp_len - 1)), mode='constant')
            for spike_template_index, spk_temp in enumerate(spike_templates):
                for i in range(len(roi_dff[r])):
                    p = np.corrcoef(roi_dff_pad[i:(i + len(spike_templates[spk_temp]))],
                                    spike_templates[spk_temp])
                    m[i, spike_template_index] = p[0, 1]


            spike_times[r] = []
            spike_correlations = np.max(m, axis=1)
            max_correlations[r] = spike_correlations
            self.max_cor_templates[r] = np.argmax(m, axis=1) + 1

            j = 0
            # iterate through frame in one label
            while j < len(spike_correlations):
                if spike_correlations[j] > spk_threshold:
                    s_max = j
                    loop = True
                    # print(f'start loop at {j}')

                    # find the frame for the peak
                    # iterate through the correlation between each pixel with the temp
                    while loop:
                        while spike_correlations[j + 1] > reset_threshold:
                            if spike_correlations[j + 1] > spike_correlations[s_max]:
                                s_max = j + 1
                            j += 1
                        if spike_correlations[j + 2] > reset_threshold:
                            j += 1
                        else:
                            loop = False
                    # print(f'end loop at {j} with s_max of {s_max}')

                    # find the amplitude
                    window_start = max(0, (s_max - 5))
                    window_end = min((len(roi_dff[r]) - 1), (s_max + 15))
                    window = roi_dff[r][window_start:window_end]
                    peak_height = np.max(window) - np.min(window)
                    # print(peak_height)
                    if peak_height > 0.02:
                        spike_times[r].append(s_max)
                j += 1

            # the consecutive peaks should not be closer in frame
            if len(spike_times[r]) >= 2:
                for k in range(len(spike_times[r]) - 1):
                    if spike_times[r][k] is not None and \
                        (spike_times[r][k + 1] - spike_times[r][k]) <= 10:
                            spike_times[r][k + 1] = None
                spike_times[r] = [spk for spk in spike_times[r] if spk is not None]

        return spike_times, max_correlations, max_cor_templates

    def analyze_ROI(self, roi_dff: dict, spk_times: dict):
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
        metadata_file = self.img_path[0:-8] + '_metadata.txt'
        framerate = 0

        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                metadata = f.readlines()

            for line in metadata:
                line = line.strip()
                if line.startswith('"Exposure-ms": '):
                    exposure = float(line[15:-1]) / 1000  # exposure in seconds
                    framerate = 1 / exposure  # frames/second
                    break
        # print('framerate is:', framerate, 'frames/second')

        amplitude_info = self.get_amplitude(roi_dff, spk_times)
        time_to_rise = self.get_time_to_rise(amplitude_info, framerate)
        max_slope = self.get_max_slope(roi_dff, amplitude_info)
        IEI = self.analyze_IEI(spk_times, framerate)
        roi_analysis = amplitude_info

        for r in roi_analysis:
            roi_analysis[r]['spike_times'] = spk_times[r]
            roi_analysis[r]['time_to_rise'] = time_to_rise[r]
            roi_analysis[r]['max_slope'] = max_slope[r]
            roi_analysis[r]['IEI'] = IEI[r]

        # print(roi_analysis)
        return roi_analysis, framerate

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
                # print(f'ROI {r} spike times: {spk_times[r]}')

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
                                    if dff_deriv[start_index] < 0:
                                        negative_count += 1

                                    else:
                                        negative_count = 0

                                    if negative_count == neg_reset_num:
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
                    # print(f'ROI {r} spike {i} - start_index: {start_index}, end_index: {end_index}')

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

        # for r in amplitude_info:
        #     print('ROI', r)
        #     print('amp:', amplitude_info[r]['amplitudes'])
        #     print('peak:', amplitude_info[r]['peak_indices'])
        #     print('base:', amplitude_info[r]['base_indices'])
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

        # print('time to rise:', time_to_rise)
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
        IEI: dict. label (int)- IEI_time or IEI_frame (float) 
        '''
        IEI = {}
        for r in spk_times:
            IEI[r] = []

            if len(spk_times[r]) > 1:
                IEI_frames = np.mean(np.diff(np.array(spk_times[r])))
                if framerate:
                    IEI_time = IEI_frames / framerate # in seconds
                    IEI[r].append(IEI_time)
                else:
                    IEI[r].append(IEI_frames)
        return IEI

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
            save_path = self.img_path[0:-4]

            # create the folder
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

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
            with open(save_path + '/spike_times.json', 'w') as spike_file:
                json.dump(self.spike_times, spike_file, indent="")

            # dict. label (int) - dict[amplitude, peak_indices, base_indices]
            with open(save_path + '/roi_analysis.json', 'w') as analysis_file:
                json.dump(self.roi_analysis, analysis_file, indent="")

            # num_events for each labeled ROI
            # first column: ROI number
            # second column: num of events
            # third column: frequency (num events/frames or exposure )
            num_events = np.zeros((len(self.spike_times.keys()), 3))
            active_roi = 0
            frame_info = self.img_stack.shape[0]/self.framerate if self.framerate else len(self.img_stack)
            for i, r in enumerate(self.spike_times):
                num_e = len(self.spike_times[r])
                if num_e > 0:
                    active_roi += 1
                num_events[i, 0] = i
                num_events[i, 1] = num_e
                num_events[i, 2] = num_e / frame_info

            with open(save_path + '/num_events.csv', 'w', newline='') as num_event_file:
                writer = csv.writer(num_event_file, dialect="excel")
                if self.framerate:
                    fields = ['ROI', 'Num_events', 'Frequency (num of events/s)']
                else:
                    fields = ['ROI', 'Num_events', 'Frequency (num of events/frame)']
                writer.writerow(fields)
                writer.writerows(num_events)
                sum_text = [[f'Active ROIs: {str(active_roi)}'], ['']]
                sum_text.extend([[f'Framerate: {frame_info}']])
                writer.writerows(sum_text)

            # label with the maximum correlation withs one of the spike templates
            max_cor = np.zeros([len(self.max_correlations[list(self.max_correlations.keys())[0]]),
                                len(self.max_correlations)])
            for i, r in enumerate(self.max_correlations):
                max_cor[:, i] = self.max_correlations[r]

            with open(save_path + '/max_correlations.csv', 'w', newline='') as cor_file:
                writer = csv.writer(cor_file)
                writer.writerow(self.max_correlations.keys())
                for i in range(max_cor.shape[0]):
                    writer.writerow(max_cor[i, :])

            # label-index of the spike template with the maximum correlation pair
            max_cor_temps = np.zeros([len(self.max_cor_templates[list(self.max_cor_templates.keys())[0]]),
                                      len(self.max_cor_templates)])
            for i, r in enumerate(self.max_cor_templates):
                max_cor_temps[:, i] = self.max_cor_templates[r]

            with open(save_path + '/max_cor_templates.csv', 'w', newline='') as cor_temp_file:
                writer = csv.writer(cor_temp_file)
                writer.writerow(self.max_cor_templates.keys())
                for i in range(max_cor_temps.shape[0]):
                    writer.writerow(max_cor_temps[i, :])

            self.canvas_traces.print_png(save_path + '/traces.png')
            self.canvas_just_traces.print_png(save_path + '/traces_no_detections.png')

            label_array = np.stack((self.label_layer.data,) * 4, axis=-1).astype(float)
            for i in range(1, np.max(self.labels) + 1):
                i_coords = np.asarray(label_array == [i, i, i, i]).nonzero()
                label_array[(i_coords[0], i_coords[1])] = self.colors[i - 1]

            self.label_layer = self.viewer.add_image(label_array, name='roi_image', visible=False)
            im = Image.fromarray((label_array*255).astype(np.uint8))
            bk_im = Image.new(im.mode, im.size, "black")
            bk_im.paste(im, im.split()[-1])
            bk_im.save(save_path + '/ROIs.png')

            # the centers of each ROI
            roi_centers = {}
            for roi_number, roi_coords in self.roi_dict.items():
                center = np.mean(roi_coords, axis=0)
                roi_centers[roi_number] = (int(center[0]), int(center[1]))

            with open(save_path + '/roi_centers.json', 'w') as roi_file:
                json.dump(roi_centers, roi_file, indent="")

            # prediction layer
            self.prediction_layer.save(save_path + '/prediction.tif')

            self.generate_summary(save_path)

        else:
            self.general_msg('No ROI', 'Cannot save data')

    def generate_summary(self, save_path:str) -> None:
        '''
        to generate a summary of the graphs that include the average and std of 
        amplitudes, time_to_rise, IEI, and mean connectivity

        parameters:
        ------------
        save_path: str.  the prefix of the tif file name
        '''
        total_amplitude = []
        total_time_to_rise = []
        total_max_slope = []
        total_IEI = []
        total_num_events = []


        for r in self.roi_analysis:
            if len(self.roi_analysis[r]['amplitudes']) > 0:
                total_amplitude.extend(self.roi_analysis[r]['amplitudes'])
                total_time_to_rise.extend(self.roi_analysis[r]['time_to_rise'])
                total_max_slope.extend(self.roi_analysis[r]['max_slope'])
                # NOTE: add the number of spikes for each roi
                if len(self.spike_times[r]) > 0:
                    total_num_events.append(len(self.spike_times[r]))
            if len(self.roi_analysis[r]['IEI']) > 0:
                total_IEI.extend(self.roi_analysis[r]['IEI'])

        if any(self.spike_times.values()):
            avg_amplitude = np.mean(np.array(total_amplitude))
            std_amplitude = np.std(np.array(total_amplitude))
            avg_max_slope = np.mean(np.array(total_max_slope))
            std_max_slope = np.std(np.array(total_max_slope))
            # NOTE: get the average num of events
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
        percent_active = self.analyze_active(self.spike_times)

        with open(save_path + '/summary.txt', 'w') as sum_file:
            units = "seconds" if self.framerate else "frames"
            sum_file.write(f'File: {self.img_path}\n')
            if self.framerate:
                sum_file.write(f'Framerate: {self.framerate} fps\n')
            else:
                sum_file.write('No framerate detected\n')
            sum_file.write(f'Total ROI: {len(self.roi_dict)}\n')
            sum_file.write(f'Percent Active ROI (%): {percent_active}\n')
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
            # NOTE: num_events
            sum_file.write(f'Average Number of events: {avg_num_events}\n')
            if len(total_num_events) > 0:
                sum_file.write(f'\tNumber of events Standard Deviation: {std_num_events}\n')
                if self.framerate:
                    frequency = avg_num_events/(self.img_stack.shape[0]/self.framerate)
                    sum_file.write(f'\tAverage Frequency (num_events/s): {frequency}\n')
                else:
                    if len(self.img_stack) > 3:
                        frame = self.img_stack.shape[1]
                    else:
                        frame = self.img_stack.shape[0]
                    sum_file.write(f'\tFrequency (num_events/frame): {avg_num_events/frame}\n')

            sum_file.write(f'Global Connectivity: {self.mean_connect}')

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
        i = len(self.viewer.layers) - 1
        while i >= 0:
            self.viewer.layers.pop(i)
            i -= 1

        if not self.batch_process:
            self.model_unet = None

        self.img_stack = None
        self.img_name = None
        self.labels = None
        self.label_layer = None
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

        self.axes.cla()
        self.canvas_traces.draw_idle()

    @magicgui(call_button="batch process (evoked activity)",
              blue_file={"label": "Choose the stimulated area file:", "mode": "r"},
              ca_file={"label": "Choose the Calcium Imaging directory:", "mode": "d"})
    def _evk_batch_process(self, blue_file: Path, ca_file: Path) -> None:
        blue_file_path = str(blue_file)
        self.ca_file = str(ca_file)

        self.batch_proess = True

        # assuming the same blue area for all the input ca imaging file
        st_area = self.process_blue(blue_file_path, 80)
        old_parent = ''
        for file in Path(self.ca_file).glob('**/*.ome.tif'):
            img = tff.imread(file, is_ome=False, is_mmstack=False)
            self.viewer.add_image(img, name=file.stem)
            self.img_stack = self.viewer.layers[1].data
            self.img_path = file
            self.img_name = file.stem

            # if opening the file in a new experiment folder
            if old_parent != file.parent:
                # set the parent folder
                old_parent = file.parent

                # initiate the unet model
                img_size = self.img_stack.shape[-1]
                dir_path = os.path.dirname(os.path.realpath(__file__))
                path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')
                self.model_unet = tf.keras.models.load_model(path, custom_objects={"K": K})
                self.unet_init = True

            # produce the prediction and labeled layers
            background_layer = 0
            minsize = 100
            self.labels, self.label_layer, self.roi_dict = self.segment(self.img_stack, minsize, background_layer)

            # to group the cells in the stimulated area vs not in the stimulated area
            if self.label_layer:
                st_rois, nst_rois = self.group_st_cells(st_area, 0.1)
                spike_templates_file = 'spikes.json'
                # stimulated cells
                roi_signal_st = self.calculate_ROI_intensity(st_rois, self.img_stack)
                roi_dff_st, median_st, _ = self.calculateDFF(roi_signal_st)

                st_spike_times, st_max_correlation, st_max_cor_temp = self.find_peaks(roi_dff_st, spike_templates_file, 0.85, 0.8)
                roi_analysis_st, self.framerate = self.analyze_ROI(roi_dff_st, st_spike_times)

                # unstimulated cells
                roi_signal_nst = self.calculate_ROI_intensity(nst_rois, self.img_stack)
                roi_dff_nst, median_nst, _ = self.calculateDFF(roi_signal_nst)
                nst_spike_times, nst_max_correlation, nst_max_cor_temp = self.find_peaks(roi_dff_nst, spike_templates_file, 0.85, 0.8)
                roi_analysis_nst, _ = self.analyze_ROI(roi_dff_nst, nst_spike_times)

                # calculate connetivity
                self.roi_signal = self.calculate_ROI_intensity(self.roi_dict, self.img_stack)
                self.roi_dff, self.median, self.bg = self.calculateDFF(self.roi_signal)
                self.spike_times = self.find_peaks(self.roi_dff, spike_templates_file, 0.85, 0.8)
                self.mean_connect = self.get_mean_connect(self.roi_dff, self.spike_times)

                # save file
                self.save_evoked_files()

            # clear
            self.clear()

        self.model_unet = None
        self.batch_process = False
        self.blue_file = None
        self.ca_file = None
        self.unet_init = False

    def process_blue(self, blue_file_path: str, threshold: int):
        '''
        process the blue light file to get the position of the stimulatation area

        Parameter:
        -------------
        blue_file_path: str. the path to the blue file
        threshold: int. the cutoff-value of pixel brightness for the stimulated area

        Return:
        -------------
        st_area: ndarray. shape(number of pixel in the stimulated area, 1)
            an array of the position of the pixels in the stimulated area
        '''

        # img = io.imread(blue_file_path, as_gray=True)
        #using opencv
        blue_img = cv2.imread(blue_file_path, cv2.IMREAD_GRAYSCALE)
        # self.viewer.add_image(blue_img, name='test_blue img')
        ret,th = cv2.threshold(blue_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th,(5,5),0)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        # only include the pixels that is brighter than 80
        st_area = np.argwhere(closing>threshold)
        st_area_t = cv2.transpose(st_area) #(x,y)

        # # to visualize the epllipse
        # epllipse = cv2.fitEllipse(st_area)
        # (x, y), (d1, d2), angle = epllipse
        # print(f'center: {(x, y)}, diameters: {(d1, d2)}')
        # self.viewer.add_image(cv2.ellipse(blue_img, (int(x), int(y)), (int(d1/2), int(d2/2)), angle, 0, 360, (255, 255, 255), 3))
        self.viewer.add_image(closing, name="closing")
        # self.viewer.add_image(st_area_t, name="st_area")

        return st_area_t

    def group_st_cells(self, blue_area, overlap_th: float) -> dict:
        '''
        group the cells that are (mostly) in the stimulated area together

        parameter:
        -----------
        blue_area: ndarray. position of the pixels in the stimulated area
        overlap_th: float. the cutoff value for couting a cell as stimulated

        return:
        -----------
        st_roi: dict. label-position pair of stimulated cells
        nst_roi: dict. label-position pair of cells that are not stimulated
        '''
        # find rois that is in the stimulated area
        st_roi = {}
        for r in self.roi_dict:
            overlap = len(set(self.roi_dict).intersection(set(map(tuple, blue_area))))
            perc_overlap = overlap / len(self.roi_dict[r])

            if perc_overlap > overlap_th:
                st_roi[r] = self.roi_dict[r]

        # group those that are not in the stimulated area
        nst_roi = self.roi_dict.copy()

        for label in st_roi:
            del nst_roi[label]

        # regroup the labels
        st_label = np.zeros_like(self.labels)
        nst_label = np.zeros_like(self.labels)

        for r in st_roi:
            roi_coords = np.array(st_roi[r]).T.tolist()
            st_label[tuple(roi_coords)] = r

        for r in nst_roi:
            roi_coords = np.array(nst_roi[r]).T.tolist()
            nst_label[tuple(roi_coords)] = r

        # create label layers for each group
        st_layer = self.viewer.add_labels(st_label, name='Stimulated cells', opacity=1)
        nst_layer = self.viewer.add_labels(nst_label, name='Not stimulated cells', opacity=1)

        # NOTE: leave the code to save the roi files here for now. 
        # MOVE to save method later!!!
        label_array = np.stack((self.label_layer.data,) * 4, axis=-1).astype(float)
        st_label_array = np.stack((st_layer.data, ) * 4, axis=-1).astype(float)
        nst_label_array = np.stack((nst_layer.data, ) * 4, axis=-1).astype(float)

        for i in range(1, np.max(self.labels) + 1):
            i_coords = np.asarray(label_array == [i, i, i, i]).nonzero()
            label_array[(i_coords[0], i_coords[1])] = self.colors[i - 1]

        print(f'st_label array shape: {st_label_array.shape}')
        print(f'st_label_array: {st_label_array}')
        for i in range(1, st_label_array.shape[0]+1):
            i_coords = np.asarray(st_label_array == [i, i, i, i]).nonzeror()
            st_label_array[(i_coords[0], i_coords[1])] = self.colors[i-1]

        self.label_layer = self.viewer.add_image(label_array, name='roi_image', visible=False)
        im = Image.fromarray((label_array*255).astype(np.uint8))
        bk_im = Image.new(im.mode, im.size, "black")
        bk_im.paste(im, im.split()[-1])
        save_path = self.img_path[0:-4]
        bk_im.save(save_path + '/ROIs.png')

        return st_roi, nst_roi

    def save_evoked_files(self, st, roi_signal, roi_dff, median, spike_times,
                          roi_analysis, max_correlations, max_cor_templates):
        '''
        save the analysis files for evoked activity
        '''
        if self.roi_dict:
            save_path = self.img_path[0:-4]

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            if st:
                raw_signal_fname = '/raw_signal_st.csv'
                dff_fname = '/dff_st.csv'
                median_fname = '/medians_st.json'
                spike_fname = '/spike_times_st.json'
                roi_analysis_fname = '/roi_analysis_st.json'
                num_e_fname = '/num_events_st.csv'
                max_cor_fname = '/max_correlations_st.csv'
                max_cor_temp_fname = '/max_cor_templates_st.csv'
            else:
                raw_signal_fname = '/raw_signal_nst.csv'
                dff_fname = '/dff_nst.csv'
                median_fname = '/medians_nst.json'
                spike_fname = '/spike_times_nst.json'
                roi_analysis_fname = '/roi_analysis_nst.json'
                num_e_fname = '/num_events_nst.csv'
                max_cor_fname = '/max_correlations_nst.csv'
                max_cor_temp_fname = '/max_cor_templates_nst.csv'


            raw_signal = np.zeros([len(roi_signal[list(roi_signal.keys())[0]]), len(roi_signal)])
            for i, r in enumerate(roi_signal):
                raw_signal[:, i] = roi_signal[r]

            with open(save_path + raw_signal_fname, 'w', newline='') as st_signal_file:
                writer = csv.writer(st_signal_file, dialect='excel')
                writer.writerow(raw_signal.keys())
                for i in range(raw_signal.shape[0]):
                    writer.writerow(raw_signal[i, :])

            dff_signal = np.zeros([len(roi_dff[list(roi_dff.keys())[0]]), len(roi_dff)])
            for i, r in enumerate(roi_dff):
                dff_signal[:, i] = roi_dff[r]

            with open(save_path + dff_fname, 'w', newline='') as dff_file:
                writer = csv.writer(dff_file)
                writer.writerow(self.roi_dff.keys())
                for i in range(dff_signal.shape[0]):
                    writer.writerow(dff_signal[i, :])

            # the median background fluorescence
            with open(save_path + median_fname, 'w') as median_file:
                json.dump(median, median_file, indent="")

            # the label-frame of peaks pairs
            with open(save_path + spike_fname, 'w') as spike_file:
                json.dump(spike_times, spike_file, indent="")

            # dict. label (int) - dict[amplitude, peak_indices, base_indices]
            with open(save_path + roi_analysis_fname, 'w') as analysis_file:
                json.dump(roi_analysis, analysis_file, indent="")

            # num of events and frequency
            num_events = np.zeros((len(spike_times.keys()), 3))
            active_roi = 0
            frame_info = self.img_stack.shape[0]/self.framerate if self.framerate else len(self.img_stack)
            for i, r in enumerate(self.spike_times):
                num_e = len(spike_times[r])
                if num_e > 0:
                    active_roi += 1
                num_events[i, 0] = i
                num_events[i, 1] = num_e
                num_events[i, 2] = num_e / frame_info

            with open(save_path + num_e_fname, 'w', newline='') as num_event_file:
                writer = csv.writer(num_event_file, dialect="excel")
                fields = ['ROI', 'Num_events', 'Frequency (num of events/s)'] if self.framerate\
                      else ['ROI', 'Num_events', 'Frequency (num of events/frame)']
                # if self.framerate:
                #     fields = ['ROI', 'Num_events', 'Frequency (num of events/s)']
                # else:
                #     fields = ['ROI', 'Num_events', 'Frequency (num of events/frame)']
                writer.writerow(fields)
                writer.writerows(num_events)
                sum_text = [[f'Active ROIs: {str(active_roi)}'], ['']]
                sum_text.extend([[f'Framerate: {frame_info}']])
                writer.writerows(sum_text)

            max_cor = np.zeros([len(max_correlations[list(max_correlations.keys())[0]]),
                    len(max_correlations)])
            for i, r in enumerate(max_correlations):
                max_cor[:, i] = max_correlations[r]

            with open(save_path + max_cor_fname, 'w', newline='') as cor_file:
                writer = csv.writer(cor_file)
                writer.writerow(max_correlations.keys())
                for i in range(max_cor.shape[0]):
                    writer.writerow(max_cor[i, :])

            # label-index of the spike template with the maximum correlation pair
            max_cor_temps = np.zeros([len(max_cor_templates[list(max_cor_templates.keys())[0]]),
                                      len(max_cor_templates)])
            for i, r in enumerate(max_cor_templates):
                max_cor_temps[:, i] = max_cor_templates[r]

            with open(save_path + max_cor_temp_fname, 'w', newline='') as cor_temp_file:
                writer = csv.writer(cor_temp_file)
                writer.writerow(max_cor_templates.keys())
                for i in range(max_cor_temps.shape[0]):
                    writer.writerow(max_cor_temps[i, :])

            # save the traces of two groups

            # save the labels separately

            # save the ror_centers.json file

            # save the prediction layer

