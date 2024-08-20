import os
import random
import mne
import numpy as np
import scipy.io


def band_filter(data, sfreq, freq_bands):
    time_freq_data = []
    for idx, band in enumerate(freq_bands):
        low_freq = freq_bands[band][0]
        high_freq = freq_bands[band][1]
        data_filtered = np.expand_dims(
            mne.filter.filter_data(
                data, sfreq=sfreq, l_freq=low_freq, h_freq=high_freq
            ),
            2,
        )
        time_freq_data.append(data_filtered)
    time_freq_data = np.concatenate(time_freq_data, axis=2)
    return time_freq_data


def de_extract(data, freq, n_freq_band):
    de_features = np.zeros((data.shape[0], data.shape[1] // freq, n_freq_band))
    for i in range(n_freq_band):
        band_data = data[..., i].reshape(data.shape[0], -1, freq)
        de = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(band_data, 2)))
        de_features[..., i] = de
    return de_features


def slide_window_extract_de(data, de_window_size, step=5, freq=200):
    freq_bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 14),
        "beta": (14, 30),
        "gamma": (30, 47),
    }
    n_windows = (data.shape[1] - de_window_size * freq) // (step * freq) + 1
    de_array = np.zeros((n_windows, data.shape[0], de_window_size, len(freq_bands)))
    data_filtered = band_filter(data, sfreq=freq, freq_bands=freq_bands)
    for window in range(n_windows):
        start = int(window * freq * step)
        end = int(start + de_window_size * freq)
        data_window = data_filtered[:, start:end]
        de_features = de_extract(data_window, freq, n_freq_band=len(freq_bands))
        de_array[window] = de_features
    # [n_sample, n_channels, n_time, n_freq_band]
    return de_array


if __name__ == "__main__":
    # select data part of the dataset
    data_folder_path = "./dataset/Preprocessed_EEG"
    # data_folder_path = './SEED_select'

    # delete reference channels A1 and A2 to align with the FACES dataset
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
        "A1",
        "A2",
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

    faced_chs_map = {ch: idx for idx, ch in enumerate(FACED_chs)}
    select_chs = [idx for idx, ch in enumerate(SEED_chs) if ch in faced_chs_map]
    aligned_chs = sorted(select_chs, key=lambda x: faced_chs_map[SEED_chs[x]])
    num_channels = len(aligned_chs)

    de_window_size = 1
    step = 1
    num_each_sub = None

    label = scipy.io.loadmat(os.path.join(data_folder_path, "label.mat"))["label"][0]
    positive_idx = label == 1
    neutral_idx = label == 0
    negative_idx = label == -1

    sub_data_positive = {}
    sub_data_neutral = {}
    sub_data_negative = {}

    for i, file in enumerate(os.listdir(data_folder_path)):
        if file.endswith(".mat"):
            if "label" not in file:
                subject_name = file.split("_")[0]
                data = scipy.io.loadmat(os.path.join(data_folder_path, file))

                for idx, key in enumerate(data.keys()):
                    if idx >= 3:
                        data[key] = data[key][aligned_chs]
                        print(data[key].shape)
                for idx, key in enumerate(data.keys()):
                    if idx >= 3:
                        if positive_idx[idx - 3]:
                            sub_data_positive.setdefault(subject_name, []).append(
                                slide_window_extract_de(
                                    data[key], de_window_size=de_window_size, step=step
                                )
                            )
                        elif neutral_idx[idx - 3]:
                            sub_data_neutral.setdefault(subject_name, []).append(
                                slide_window_extract_de(
                                    data[key], de_window_size=de_window_size, step=step
                                )
                            )
                        elif negative_idx[idx - 3]:
                            sub_data_negative.setdefault(subject_name, []).append(
                                slide_window_extract_de(
                                    data[key], de_window_size=de_window_size, step=step
                                )
                            )

    sub_train_positive = {}
    sub_train_neutral = {}
    sub_train_negative = {}
    sub_eval_positive = {}
    sub_eval_neutral = {}
    sub_eval_negative = {}

    for label, key in enumerate(sub_data_positive.keys()):
        # sub_train_positive[key] = np.vstack(sub_data_positive[key][:-1])
        # sub_eval_positive[key] = sub_data_positive[key][-1]
        # sub_train_neutral[key] = np.vstack(sub_data_neutral[key][:-1])
        # sub_eval_neutral[key] = sub_data_neutral[key][-1]
        # sub_train_negative[key] = np.vstack(sub_data_negative[key][:-1])
        # sub_eval_negative[key] = sub_data_negative[key][-1]

        train_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
        eval_indices = [4, 9, 14]

        sub_train_positive[key] = np.vstack(
            [sub_data_positive[key][i] for i in train_indices]
        )
        sub_eval_positive[key] = np.vstack(
            [sub_data_positive[key][i] for i in eval_indices]
        )

        sub_train_neutral[key] = np.vstack(
            [sub_data_neutral[key][i] for i in train_indices]
        )
        sub_eval_neutral[key] = np.vstack(
            [sub_data_neutral[key][i] for i in eval_indices]
        )

        sub_train_negative[key] = np.vstack(
            [sub_data_negative[key][i] for i in train_indices]
        )
        sub_eval_negative[key] = np.vstack(
            [sub_data_negative[key][i] for i in eval_indices]
        )

        print(
            key,
            sub_train_positive[key].shape,
            sub_eval_positive[key].shape,
            sub_train_neutral[key].shape,
            sub_eval_neutral[key].shape,
            sub_train_negative[key].shape,
            sub_eval_negative[key].shape,
        )

    positive_resample = (
        np.arange(len(sub_train_positive[next(iter(sub_train_positive))]))
        if num_each_sub is None
        else random.sample(
            list(
                np.arange(
                    len(sub_train_positive[next(iter(sub_train_positive))]),
                    num_each_sub,
                )
            )
        )
    )

    neutral_resample = (
        np.arange(len(sub_train_neutral[next(iter(sub_train_neutral))]))
        if num_each_sub is None
        else random.sample(
            list(
                np.arange(
                    len(sub_train_neutral[next(iter(sub_train_neutral))]), num_each_sub
                )
            )
        )
    )

    negative_resample = (
        np.arange(len(sub_train_negative[next(iter(sub_train_negative))]))
        if num_each_sub is None
        else random.sample(
            list(
                np.arange(
                    len(sub_train_negative[next(iter(sub_train_negative))]),
                    num_each_sub,
                )
            )
        )
    )

    train_data = {"positive": [], "neutral": [], "negative": []}
    train_label = {"positive": [], "neutral": [], "negative": []}
    eval_data = {"positive": [], "neutral": [], "negative": []}
    eval_label = {"positive": [], "neutral": [], "negative": []}

    for label, key in enumerate(sub_train_positive.keys()):
        train_data["positive"].append(sub_train_positive[key][positive_resample])
        train_data["neutral"].append(sub_train_neutral[key][neutral_resample])
        train_data["negative"].append(sub_train_negative[key][negative_resample])

        train_label["positive"] = train_label["positive"] + [label] * len(
            sub_train_positive[key][positive_resample]
        )
        train_label["neutral"] = train_label["neutral"] + [label] * len(
            sub_train_neutral[key][neutral_resample]
        )
        train_label["negative"] = train_label["negative"] + [label] * len(
            sub_train_negative[key][negative_resample]
        )

        eval_data["positive"].append(sub_eval_positive[key])
        eval_data["neutral"].append(sub_eval_neutral[key])
        eval_data["negative"].append(sub_eval_negative[key])

        eval_label["positive"] = eval_label["positive"] + [label] * len(
            sub_eval_positive[key]
        )
        eval_label["neutral"] = eval_label["neutral"] + [label] * len(
            sub_eval_neutral[key]
        )
        eval_label["negative"] = eval_label["negative"] + [label] * len(
            sub_eval_negative[key]
        )

    for emo in train_data.keys():
        train_data[emo] = np.vstack(train_data[emo])
        train_label[emo] = np.array(train_label[emo])
        eval_data[emo] = np.vstack(eval_data[emo])
        eval_label[emo] = np.array(eval_label[emo])

        print(
            emo,
            train_data[emo].shape,
            train_label[emo].shape,
            eval_data[emo].shape,
            eval_label[emo].shape,
        )

    os.makedirs(
        f"./dataset/SEED_PI/window{de_window_size}_step{step}_ch{num_channels}",
        exist_ok=True,
    )
    for emo in train_data.keys():
        np.save(
            f"./dataset/SEED_PI/window{de_window_size}_step{step}_ch{num_channels}/{emo}_train_data.npy",
            train_data[emo],
        )
        np.save(
            f"./dataset/SEED_PI/window{de_window_size}_step{step}_ch{num_channels}/{emo}_eval_data.npy",
            eval_data[emo],
        )
        np.save(
            f"./dataset/SEED_PI/window{de_window_size}_step{step}_ch{num_channels}/{emo}_train_label.npy",
            train_label[emo],
        )
        np.save(
            f"./dataset/SEED_PI/window{de_window_size}_step{step}_ch{num_channels}/{emo}_eval_label.npy",
            eval_label[emo],
        )
        print(
            "emotion: ",
            emo,
            "Train Set: ",
            train_data[emo].shape,
            train_label[emo].shape,
            "Eval Set: ",
            eval_data[emo].shape,
            eval_label[emo].shape,
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
    ]
    np.save(
        f"./dataset/SEED_PI/window{de_window_size}_step{step}_ch{num_channels}/channel_names.npy",
        channel_names,
    )
