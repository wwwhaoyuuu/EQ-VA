import pickle
import shutil

import numpy as np
import mne
import os
import scipy.io
from scipy.io import savemat
from scipy.signal.windows import hann

"""
    Extracting and Saving the PSD and DE features to mat and pkl files.
    data files are dict and label files are array.
"""


def _get_relative_psd(relative_energy_graph, freq_bands, sample_freq, stft_n=256):
    start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
    # print(start_index, end_index)
    psd = np.mean(relative_energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
    # print('psd:', psd.shape)
    return psd


def extract_psd_feature(data, freq_bands, window_size, sample_freq=200, stft_n=256):
    """
        data: 2D array, shape [n_channels, n_samples]
        window_size: float, window size in seconds
        freq_bands: list of tuples, frequency bands for feature extraction
        stft_n: int, number of points for FFT !!(its value should >= window_size*sample_freq and should be 2^x)
        sample_freq: int, sampling frequency
        To extract PSD feature from EEG data
    """
    # Ptr operation
    if len(data.shape) > 2:
        data = np.squeeze(data)
    n_channels, n_samples = data.shape
    point_per_window = int(sample_freq * window_size)
    window_num = int(n_samples // point_per_window)
    psd_feature = np.zeros((window_num, len(freq_bands), n_channels))
    # print('psd feature shape:', psd_feature.shape)

    for window_index in range(window_num):
        start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
        window_data = data[:, start_index:end_index]
        hdata = window_data * hann(point_per_window)
        fft_data = np.fft.fft(hdata, n=stft_n)
        # print('fft_data shape:', fft_data.shape)
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
        # print('energy_graph.shape:', energy_graph.shape)
        # 这里小小改了一下，因为原来是没有axis=1的，计算能量频谱图应该是针对每个通道的
        relative_energy_graph = energy_graph / np.sum(energy_graph, axis=1, keepdims=True)

        for band_index, band in enumerate(freq_bands):
            band_relative_psd = _get_relative_psd(relative_energy_graph, band, sample_freq, stft_n)
            psd_feature[window_index, band_index, :] = band_relative_psd

    return psd_feature


def SEED_mat2list(matfile_path):
    """
        matfile_path: the path of SEED mat file
        Convert SEED mat file to list
    """
    mat_dict = scipy.io.loadmat(matfile_path)
    list_data = []
    for idx, video in enumerate(mat_dict):
        if idx >= 3:
            list_data.append(mat_dict[video])
    return list_data


def label_mat2pkl(origin_labelmat, target_folder):
    """
        origin_labelmat: the path of label mat file
        target_folder: the folder to save the pkl file
        Convert label mat file to pkl file
    """
    _, filename = os.path.split(origin_labelmat)
    label = scipy.io.loadmat(origin_labelmat)['label'][0]
    with open(os.path.join(target_folder, filename.replace('.mat', '.pkl')), 'wb') as f:
        pickle.dump(label, f)


if __name__ == '__main__':
    """
        n_vids:number of trials or number of videos;freq:sampling rate;nsec:number of video second;nchn:Number of electrodes
        for 'SEED/Preprocessed_EEG' dataset,n_vids=15;freq=200;nsec=var(should be certain);nchn=62
    """
    n_vids = 15
    freq = 200
    nsec = 0  # should be certain
    nchn = 62
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

    data_path = 'E:\\1DomainCrossPI\AffectiveEEGPI\SEED\Preprocessed_EEG'
    file_names = [file for file in os.listdir(data_path) if file != 'label.mat']
    file_names.sort()

    savepaths = {
        'PSD_mat': './SEED_Features/PSD_mat',
        'DE_mat': './SEED_Features/DE_mat',
        'PSD_pkl': './SEED_Features/PSD_pkl',
        'DE_pkl': './SEED_Features/DE_pkl',
    }
    for savepath in savepaths:
        if not os.path.exists(savepaths[savepath]):
            os.makedirs(savepaths[savepath])
        if savepaths[savepath].endswith('mat'):
            shutil.copy(os.path.join(data_path, 'label.mat'), savepaths[savepath])
        else:
            label_mat2pkl(os.path.join(data_path, 'label.mat'), savepaths[savepath])

    for idx, file_name in enumerate(file_names):
        psd_dict = {}
        de_dict = {}
        # psd_list = []
        # de_list = []
        list_data = SEED_mat2list(os.path.join(data_path, file_name))
        # PSD feature extract and save, mat file save dict, pkl file save list
        for i, segment in enumerate(list_data):
            psd_data = extract_psd_feature(segment, freq_bands, 1, sample_freq=200, stft_n=256)
            psd_data = np.transpose(psd_data, (2, 0, 1))
            psd_dict[f'psd{i + 1}'] = psd_data
            # psd_list.append(psd_data)
        savemat(f"{savepaths['PSD_mat']}/{file_name}", psd_dict)
        with open(f"{savepaths['PSD_pkl']}/{file_name.replace('mat','pkl')}", 'wb') as f:
            pickle.dump(psd_dict, f)

        # DE feature extract and save,mat file save dict, pkl file save list
        for i, segment in enumerate(list_data):
            segment = segment[:, :int(segment.shape[1] // freq) * freq]
            segment_data = []
            for j in range(len(freq_bands)):
                low_freq = freq_bands[j][0]
                high_freq = freq_bands[j][1]
                segment_filt = mne.filter.filter_data(segment, freq, l_freq=low_freq, h_freq=high_freq)
                segment_filt = segment_filt.reshape(nchn, -1, freq)
                de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(segment_filt, 2)))
                segment_data.append(de_one)
            de_dict[f'de{i + 1}'] = np.array(segment_data).reshape(nchn, -1, len(freq_bands))
            # de_list.append(np.array(segment_data).reshape(nchn, -1, len(freq_bands)))
        savemat(f"{savepaths['DE_mat']}/{file_name}", de_dict)
        with open(f"{savepaths['DE_pkl']}/{file_name.replace('mat','pkl')}", 'wb') as f:
            pickle.dump(de_dict, f)
