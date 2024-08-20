import fnmatch
import os
import pickle
from random import random
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample


def get_stimuli_info(excel_path):
    emo_stimuli = {}
    df = pd.read_excel(excel_path, usecols=["Targeted Emotion"]).dropna()
    for row in df.itertuples(index=True):
        print(row[0], row[1])
        emo_stimuli[row[1]] = emo_stimuli.get(row[1], []) + [row[0]]
    return emo_stimuli


def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data


def band_filter(data, sfreq, freq_bands):
    time_freq_data = []
    for idx, band in enumerate(freq_bands):
        low_freq = freq_bands[band][0]
        high_freq = freq_bands[band][1]
        data_filtered = np.expand_dims(
            mne.filter.filter_data(
                data, sfreq=sfreq, l_freq=low_freq, h_freq=high_freq
            ),
            3,
        )
        time_freq_data.append(data_filtered)
    time_freq_data = np.concatenate(time_freq_data, axis=3)
    return time_freq_data


def resample_data(data, origin_sfreq, target_sfreq):
    data_resampled = np.empty(
        (
            data.shape[0],
            data.shape[1],
            data.shape[2] // origin_sfreq * target_sfreq,
            data.shape[3],
        )
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[3]):
                data_resampled[i, j, :, k] = resample(
                    data[i, j, :, k], data.shape[2] // origin_sfreq * target_sfreq
                )

    return data_resampled


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
    n_windows = (data.shape[2] - de_window_size * freq) // (step * freq) + 1
    de_array = np.zeros(
        (data.shape[0], n_windows, data.shape[1], de_window_size, data.shape[3])
    )
    for video in range(data.shape[0]):
        data_video = data[video]
        for window in range(n_windows):
            start = int(window * freq * step)
            end = int(start + de_window_size * freq)
            data_window = data_video[:, start:end]
            de_features = de_extract(data_window, freq, n_freq_band=data.shape[3])
            de_array[video, window] = de_features
    return de_array


def get_valance_data(data, stimuli_dict, all=False):
    if all:
        select_emo = {
            "positive": ["Amusement", "Inspiration", "Joy", "Tenderness"],
            "neutral": ["Neutral"],
            "negative": ["Anger", "Disgust", "Fear", "Sadness"],
        }
    else:
        select_emo = {
            "positive": ["Joy"],
            "neutral": ["Neutral"],
            "negative": ["Anger"],
        }
    valance_data_dict = {}
    valance_train_data_dict = {}
    valance_test_data_dict = {}
    for valance in select_emo:
        for emo in select_emo[valance]:
            idx = stimuli_dict[emo]
            valance_data_dict[valance] = valance_data_dict.setdefault(valance, []) + [
                data[idx]
            ]
            valance_train_data_dict[valance] = valance_train_data_dict.setdefault(
                valance, []
            ) + [data[idx[:-1]]]
            valance_test_data_dict[valance] = valance_test_data_dict.setdefault(
                valance, []
            ) + [data[idx[-1]]]
        valance_data_dict[valance] = np.vstack(valance_data_dict[valance])
        valance_train_data_dict[valance] = np.vstack(valance_train_data_dict[valance])
        valance_test_data_dict[valance] = np.vstack(valance_test_data_dict[valance])
        valance_data_dict[valance] = valance_data_dict[valance].reshape(
            (-1, data.shape[2], data.shape[3], data.shape[4])
        )
        valance_train_data_dict[valance] = valance_train_data_dict[valance].reshape(
            (-1, data.shape[2], data.shape[3], data.shape[4])
        )
        valance_test_data_dict[valance] = valance_test_data_dict[valance].reshape(
            (-1, data.shape[2], data.shape[3], data.shape[4])
        )
    return valance_data_dict, valance_train_data_dict, valance_test_data_dict


if __name__ == "__main__":
    # base settings
    de_window_size = 5
    step = 1
    num_each_sub = None

    # base info
    origin_sfreq = 250
    target_sfreq = 200
    n_vids = 28

    freq_bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 14),
        "beta": (14, 30),
        "gamma": (30, 47),
    }
    data_folder = "/data/0shared/jinjiarui/run/PI/Dataset/FACED_Processed_data"
    save_folder = "/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI"
    os.makedirs(save_folder, exist_ok=True)
    info_excel_path = (
        "/data/0shared/jinjiarui/run/PI/Dataset/FACED_Processed_data/Stimuli_info.xlsx"
    )
    stimuli_dict = get_stimuli_info(info_excel_path)
    print("stimuli_dict: ", stimuli_dict)
    file_names = [
        name for name in os.listdir(data_folder) if fnmatch.fnmatch(name, "*.pkl")
    ]

    train_data = {}
    train_label = {}
    eval_data = {}
    eval_label = {}
    for name in file_names:
        label = int(name.split("sub")[1].split(".")[0])
        data = load_data(os.path.join(data_folder, name))

        time_freq_data = band_filter(data, origin_sfreq, freq_bands)
        data_resampled = resample_data(time_freq_data, origin_sfreq, target_sfreq)
        de_features = slide_window_extract_de(
            data_resampled, de_window_size=de_window_size, step=step, freq=target_sfreq
        )
        valance_data_dict, valance_train_data_dict, valance_test_data_dict = (
            get_valance_data(de_features, stimuli_dict=stimuli_dict, all=True)
        )

        for key in valance_data_dict.keys():
            print(key, "valance_data_dict: ", valance_data_dict[key].shape)
            print(key, "valance_train_data_dict: ", valance_train_data_dict[key].shape)
            print(key, "valance_test_data_dict: ", valance_test_data_dict[key].shape)

        positive_resample = (
            np.arange(len(valance_train_data_dict["positive"]))
            if num_each_sub is None
            else random.sample(
                list(np.arange(len(valance_train_data_dict["positive"]), num_each_sub))
            )
        )

        neutral_resample = (
            np.arange(len(valance_train_data_dict["neutral"]))
            if num_each_sub is None
            else random.sample(
                list(np.arange(len(valance_train_data_dict["neutral"]), num_each_sub))
            )
        )

        negative_resample = (
            np.arange(len(valance_train_data_dict["negative"]))
            if num_each_sub is None
            else random.sample(
                list(np.arange(len(valance_train_data_dict["negative"]), num_each_sub))
            )
        )

        valance_train_data_dict["positive"] = valance_train_data_dict["positive"][
            positive_resample
        ]
        valance_train_data_dict["neutral"] = valance_train_data_dict["neutral"][
            neutral_resample
        ]
        valance_train_data_dict["negative"] = valance_train_data_dict["negative"][
            negative_resample
        ]

        for valance in valance_train_data_dict.keys():
            train_data[valance] = train_data.setdefault(valance, []) + [
                valance_train_data_dict[valance]
            ]
            train_label[valance] = train_label.setdefault(valance, []) + [label] * len(
                valance_train_data_dict[valance]
            )
            eval_data[valance] = eval_data.setdefault(valance, []) + [
                valance_test_data_dict[valance]
            ]
            eval_label[valance] = eval_label.setdefault(valance, []) + [label] * len(
                valance_test_data_dict[valance]
            )

    for valance in train_data.keys():
        train_data[valance] = np.vstack(train_data[valance])
        train_label[valance] = np.array(train_label[valance])
        eval_data[valance] = np.vstack(eval_data[valance])
        eval_label[valance] = np.array(eval_label[valance])
        print(
            train_data[valance].shape,
            train_label[valance].shape,
            eval_data[valance].shape,
            eval_label[valance].shape,
        )

    os.makedirs(
        f"/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI/window{de_window_size}_step{step}",
        exist_ok=True,
    )
    for valance in train_data.keys():
        np.save(
            f"/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI/window{de_window_size}_step{step}/{valance}_train_data.npy",
            train_data[valance],
        )
        np.save(
            f"/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI/window{de_window_size}_step{step}/{valance}_train_label.npy",
            train_label[valance],
        )
        np.save(
            f"/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI/window{de_window_size}_step{step}/{valance}_eval_data.npy",
            eval_data[valance],
        )
        np.save(
            f"/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI/window{de_window_size}_step{step}/{valance}_eval_label.npy",
            eval_label[valance],
        )

    channel_names = [
        "FP1",
        "FP2",
        "FZ",
        "F3",
        "F4",
        "F7",
        "F8",
        "FC1",
        "FC2",
        "FC5",
        "FC6",
        "CZ",
        "C3",
        "C4",
        "T7",
        "T8",
        "CP1",
        "CP2",
        "CP5",
        "CP6",
        "PZ",
        "P3",
        "P4",
        "P7",
        "P8",
        "PO3",
        "PO4",
        "OZ",
        "O1",
        "O2",
        "A1",
        "A2",
    ]
    np.save(
        f"/data/0shared/jinjiarui/run/PI/Dataset/FACED_PI/window{de_window_size}_step{step}/channel_names.npy",
        channel_names,
    )
