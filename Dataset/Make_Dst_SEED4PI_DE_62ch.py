import os
import random
import mne
import numpy as np
import pandas as pd
import scipy.io


def band_filter(data, sfreq, freq_bands):
    # 输入的data是二维的脑电信号数据，shape为[n_channels, origin_n_samples]
    time_freq_data = []
    for idx, band in enumerate(freq_bands):
        low_freq = freq_bands[band][0]
        high_freq = freq_bands[band][1]
        data_filtered = np.expand_dims(mne.filter.filter_data(data, sfreq=sfreq, l_freq=low_freq, h_freq=high_freq), 2)
        time_freq_data.append(data_filtered)
    time_freq_data = np.concatenate(time_freq_data, axis=2)
    return time_freq_data


def de_extract(data, freq, n_freq_band):
    de_features = np.zeros((data.shape[0], data.shape[1] // freq, n_freq_band))
    # 输入的脑电信号是三维的，shape为[n_channels, spilt_n_samples, n_freq_band]
    for i in range(n_freq_band):
        # [n_channels, time, freq]
        band_data = data[..., i].reshape(data.shape[0], -1, freq)
        # [n_channels, time]
        de = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(band_data, 2)))
        de_features[..., i] = de
    return de_features


def slide_window_extract_de(data, de_window_size, step=5, freq=200):
    freq_bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 47), }
    n_windows = (data.shape[1] - de_window_size * freq) // (step * freq) + 1
    de_array = np.zeros((n_windows, data.shape[0], de_window_size, len(freq_bands)))
    data_filtered = band_filter(data, sfreq=freq, freq_bands=freq_bands)
    for window in range(n_windows):
        start = int(window * freq * step)
        end = int(start + de_window_size * freq)
        data_window = data_filtered[:, start:end]
        de_features = de_extract(data_window, freq, n_freq_band=len(freq_bands))
        de_array[window] = de_features
    # 返回的de_array的大小都是四维的[n_sample, n_channels, n_time, n_freq_band]
    return de_array


if __name__ == '__main__':
    # 选择数据中的部分数据
    data_folder_path = '/data/0shared/jinjiarui/run/PI/Dataset/SEED_select'
    # data_folder_path = './SEED_select'

    de_window_size = 5
    step = 5
    num_each_sub = None

    # 获取数据中每种情绪的idx
    label = scipy.io.loadmat(os.path.join(data_folder_path, 'label.mat'))['label'][0]
    positive_idx = label == 1
    neutral_idx = label == 0
    negative_idx = label == -1

    # 放个字典在这里保存每个被试的数据，相同被试会放在同一个list中，用于记录每个被试的数据
    sub_data_positive = {}
    sub_data_neutral = {}
    sub_data_negative = {}

    for i, file in enumerate(os.listdir(data_folder_path)):
        if file.endswith('.mat'):
            if 'label' not in file:
                subject_name = file.split('_')[0]
                data = scipy.io.loadmat(os.path.join(data_folder_path, file))

                for idx, key in enumerate(data.keys()):
                    if idx >= 3:
                        if positive_idx[idx - 3]:
                            sub_data_positive.setdefault(subject_name, []).append(
                                    slide_window_extract_de(data[key], de_window_size=de_window_size, step=step))
                        elif neutral_idx[idx - 3]:
                            sub_data_neutral.setdefault(subject_name, []).append(
                                    slide_window_extract_de(data[key], de_window_size=de_window_size, step=step))
                        elif negative_idx[idx - 3]:
                            sub_data_negative.setdefault(subject_name, []).append(
                                    slide_window_extract_de(data[key], de_window_size=de_window_size, step=step))

    sub_train_positive = {}
    sub_train_neutral = {}
    sub_train_negative = {}
    sub_eval_positive = {}
    sub_eval_neutral = {}
    sub_eval_negative = {}

    for label, key in enumerate(sub_data_positive.keys()):
        sub_train_positive[key] = np.vstack(sub_data_positive[key][:-1])
        sub_eval_positive[key] = sub_data_positive[key][-1]
        sub_train_neutral[key] = np.vstack(sub_data_neutral[key][:-1])
        sub_eval_neutral[key] = sub_data_neutral[key][-1]
        sub_train_negative[key] = np.vstack(sub_data_negative[key][:-1])
        sub_eval_negative[key] = sub_data_negative[key][-1]

    # 如果给定一个变量，就控制每个被试的数据量
    positive_resample = np.arange(len(sub_train_positive[next(iter(sub_train_positive))])) if num_each_sub is None \
        else random.sample(list(np.arange(len(sub_train_positive[next(iter(sub_train_positive))]), num_each_sub)))

    neutral_resample = np.arange(len(sub_train_neutral[next(iter(sub_train_neutral))])) if num_each_sub is None \
        else random.sample(list(np.arange(len(sub_train_neutral[next(iter(sub_train_neutral))]), num_each_sub)))

    negative_resample = np.arange(len(sub_train_negative[next(iter(sub_train_negative))])) if num_each_sub is None \
        else random.sample(list(np.arange(len(sub_train_negative[next(iter(sub_train_negative))]), num_each_sub)))

    train_data = {'positive': [], 'neutral': [], 'negative': []}
    train_label = {'positive': [], 'neutral': [], 'negative': []}
    eval_data = {'positive': [], 'neutral': [], 'negative': []}
    eval_label = {'positive': [], 'neutral': [], 'negative': []}

    for label, key in enumerate(sub_train_positive.keys()):
        train_data['positive'].append(sub_train_positive[key][positive_resample])
        train_data['neutral'].append(sub_train_neutral[key][neutral_resample])
        train_data['negative'].append(sub_train_negative[key][negative_resample])

        train_label['positive'] = train_label['positive'] + [label] * len(sub_train_positive[key][positive_resample])
        train_label['neutral'] = train_label['neutral'] + [label] * len(sub_train_neutral[key][neutral_resample])
        train_label['negative'] = train_label['negative'] + [label] * len(sub_train_negative[key][negative_resample])

        eval_data['positive'].append(sub_eval_positive[key])
        eval_data['neutral'].append(sub_eval_neutral[key])
        eval_data['negative'].append(sub_eval_negative[key])

        eval_label['positive'] = eval_label['positive'] + [label] * len(sub_eval_positive[key])
        eval_label['neutral'] = eval_label['neutral'] + [label] * len(sub_eval_neutral[key])
        eval_label['negative'] = eval_label['negative'] + [label] * len(sub_eval_negative[key])

    # 合并所有被试的数据
    for emo in train_data.keys():
        train_data[emo] = np.vstack(train_data[emo])
        train_label[emo] = np.array(train_label[emo])
        eval_data[emo] = np.vstack(eval_data[emo])
        eval_label[emo] = np.array(eval_label[emo])

        print(emo, train_data[emo].shape, train_label[emo].shape, eval_data[emo].shape, eval_label[emo].shape)

    # 保存一下数据
    os.makedirs(f'/data/0shared/jinjiarui/run/PI/Dataset/SEED_PI/window{de_window_size}_step{step}_ch62', exist_ok=True)
    for emo in train_data.keys():
        np.save(
                f'/data/0shared/jinjiarui/run/PI/Dataset/SEED_PI/window{de_window_size}_step{step}_ch62/{emo}_train_data.npy',
                train_data[emo])
        np.save(
            f'/data/0shared/jinjiarui/run/PI/Dataset/SEED_PI/window{de_window_size}_step{step}_ch62/{emo}_eval_data.npy',
            eval_data[emo])
        np.save(
                f'/data/0shared/jinjiarui/run/PI/Dataset/SEED_PI/window{de_window_size}_step{step}_ch62/{emo}_train_label.npy',
                train_label[emo])
        np.save(
                f'/data/0shared/jinjiarui/run/PI/Dataset/SEED_PI/window{de_window_size}_step{step}_ch62/{emo}_eval_label.npy',
                eval_label[emo])
        print(emo, train_data[emo].shape, eval_data[emo].shape, train_label[emo].shape, eval_label[emo].shape)

    # 保存一下通道名
    df = pd.read_excel(os.path.join(data_folder_path, 'Channel Order.xlsx'), header=None)
    channel_names = df.iloc[:, 0].values
    np.save(f'/data/0shared/jinjiarui/run/PI/Dataset/SEED_PI/window{de_window_size}_step{step}_ch62/channel_names.npy',
            channel_names)
