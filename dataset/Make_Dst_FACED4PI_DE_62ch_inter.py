import scipy
from scipy.spatial import Delaunay
import fnmatch
import os
import pickle
from random import random
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample
from joblib import Parallel, delayed


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
    for i in range(n_freq_band):
        band_data = data[..., i].reshape(data.shape[0], -1, freq)
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


# def interpolate_missing_channels(data, existing_chs, target_chs, channel_positions):
#     """
#     根据给定的通道矩阵对缺失导联进行插值填充。

#     参数：
#     data (ndarray): 原始数据，形状为(samples, channels, timepoints)。
#     existing_chs (list): 当前数据集中的导联名称。
#     target_chs (list): 目标数据集中的导联名称。
#     channel_positions (dict): 各导联的位置字典。

#     返回：
#     interpolated_data (ndarray): 插值后的数据，形状为(samples, target_channels, timepoints)。
#     """
#     num_samples, _, num_timepoints = data.shape
#     interpolated_data = np.zeros((num_samples, len(target_chs), num_timepoints))

#     existing_positions = np.array([channel_positions[ch] for ch in existing_chs])
#     target_positions = np.array([channel_positions[ch] for ch in target_chs])

#     hull = Delaunay(existing_positions)

#     for sample in range(num_samples):
#         for timepoint in range(num_timepoints):
#             values = data[sample, :, timepoint]

#             # 使用 linear 插值
#             linear_interpolator = scipy.interpolate.LinearNDInterpolator(existing_positions, values)
#             linear_interpolated_values = linear_interpolator(target_positions)

#             # 使用 nearest 插值作为回退
#             nearest_interpolator = scipy.interpolate.NearestNDInterpolator(existing_positions, values)
#             nearest_interpolated_values = nearest_interpolator(target_positions)

#             # 对于在凸包内的点，使用 linear 插值；对于在凸包外的点，使用 nearest 插值
#             for i, target_position in enumerate(target_positions):
#                 if hull.find_simplex([target_position]) >= 0:
#                     interpolated_data[sample, i, timepoint] = linear_interpolated_values[i]
#                 else:
#                     interpolated_data[sample, i, timepoint] = nearest_interpolated_values[i]

#             # # 使用 cubic 插值
#             # cubic_interpolated_values = scipy.interpolate.griddata(
#             #         existing_positions, values, target_positions, method='cubic'
#             # )

#             # # 使用 RBF 插值作为回退
#             # rbf_interpolator = scipy.interpolate.Rbf(existing_positions[:, 0], existing_positions[:, 1], values,
#             #                                          function='thin_plate')
#             # rbf_interpolated_values = rbf_interpolator(target_positions[:, 0], target_positions[:, 1])

#             # # 对于在凸包内的点，使用 cubic 样条插值；对于在凸包外的点，使用 RBF 插值
#             # for i, target_position in enumerate(target_positions):
#             #     if hull.find_simplex([target_position]) >= 0:
#             #         interpolated_data[sample, i, timepoint] = cubic_interpolated_values[i]
#             #     else:
#             #         interpolated_data[sample, i, timepoint] = rbf_interpolated_values[i]

#     return interpolated_data


def interpolate_for_timepoint(existing_positions, target_positions, values, hull):
    linear_interpolator = scipy.interpolate.LinearNDInterpolator(
        existing_positions, values
    )
    linear_interpolated_values = linear_interpolator(target_positions)

    nearest_interpolator = scipy.interpolate.NearestNDInterpolator(
        existing_positions, values
    )
    nearest_interpolated_values = nearest_interpolator(target_positions)

    interpolated_values = np.zeros(len(target_positions))
    for i, target_position in enumerate(target_positions):
        if hull.find_simplex([target_position]) >= 0:
            interpolated_values[i] = linear_interpolated_values[i]
        else:
            interpolated_values[i] = nearest_interpolated_values[i]
    return interpolated_values


def interpolate_missing_channels(
    data, existing_chs, target_chs, channel_positions, n_jobs=-1
):
    num_samples, _, num_timepoints = data.shape
    interpolated_data = np.zeros((num_samples, len(target_chs), num_timepoints))

    existing_positions = np.array([channel_positions[ch] for ch in existing_chs])
    target_positions = np.array([channel_positions[ch] for ch in target_chs])
    hull = Delaunay(existing_positions)

    for sample in range(num_samples):
        results = Parallel(n_jobs=n_jobs)(
            delayed(interpolate_for_timepoint)(
                existing_positions, target_positions, data[sample, :, timepoint], hull
            )
            for timepoint in range(num_timepoints)
        )

        interpolated_data[sample] = np.array(results).T

    return interpolated_data


def get_channel_positions(channels_matrix):
    """
    获取通道的二维坐标位置。

    参数：
    channels_matrix (list of list): 通道位置矩阵。

    返回：
    positions (dict): 通道名称到二维坐标的映射字典。
    """
    positions = {}
    for i, row in enumerate(channels_matrix):
        for j, ch in enumerate(row):
            if ch != 0:
                positions[ch] = (i, j)
    return positions


if __name__ == "__main__":
    # base settings
    de_window_size = 1
    step = 1
    num_each_sub = None

    channels_matrix = [
        [0, 0, 0, "FP1", "FPZ", "FP2", 0, 0, 0],
        [0, 0, 0, "AF3", 0, "AF4", 0, 0, 0],
        ["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
        ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
        ["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"],
        ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"],
        ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"],
        [0, "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", 0],
        [
            0,
            0,
            "CB1",
            "O1",
            "OZ",
            "O2",
            "CB2",
            0,
            0,
        ],
    ]

    FACED_chs = [
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
    ]

    SEED_chs = [
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "CB1",
        "O1",
        "OZ",
        "O2",
        "CB2",
    ]
    channel_positions = get_channel_positions(channels_matrix)

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
    data_folder = "./dataset/FACED_Processed_data"
    save_folder = "./dataset/FACED_PI"
    os.makedirs(save_folder, exist_ok=True)
    info_excel_path = os.path.join(data_folder, "Stimuli_info.xlsx")
    stimuli_dict = get_stimuli_info(info_excel_path)
    print("stimuli_dict: ", stimuli_dict)
    file_names = [
        name for name in os.listdir(data_folder) if fnmatch.fnmatch(name, "*.pkl")
    ]

    train_data = {}
    train_label = {}
    eval_data = {}
    eval_label = {}
    for i, name in enumerate(file_names):
        label = int(name.split("sub")[1].split(".")[0])
        data = load_data(os.path.join(data_folder, name))
        data = data[:, :-2]
        # interpolate to 62 channels
        print("start interpolate :" + str(i))
        data = interpolate_missing_channels(
            data, FACED_chs, SEED_chs, channel_positions, n_jobs=32
        )

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
        f"./dataset/FACED_PI/window{de_window_size}_step{step}_ch62",
        exist_ok=True,
    )
    for valance in train_data.keys():
        np.save(
            f"./dataset/FACED_PI/window{de_window_size}_step{step}_ch62/{valance}_train_data.npy",
            train_data[valance],
        )
        np.save(
            f"./dataset/FACED_PI/window{de_window_size}_step{step}_ch62/{valance}_train_label.npy",
            train_label[valance],
        )
        np.save(
            f"./dataset/FACED_PI/window{de_window_size}_step{step}_ch62/{valance}_eval_data.npy",
            eval_data[valance],
        )
        np.save(
            f"./dataset/FACED_PI/window{de_window_size}_step{step}_ch62/{valance}_eval_label.npy",
            eval_label[valance],
        )

    channel_names = [
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "CB1",
        "O1",
        "OZ",
        "O2",
        "CB2",
    ]
    np.save(
        f"./dataset/FACED_PI/window{de_window_size}_step{step}_ch62/channel_names.npy",
        channel_names,
    )
