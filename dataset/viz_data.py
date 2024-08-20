import pickle
from matplotlib import pyplot as plt


def visualize_eeg_channels(data, channels):
    plt.figure(figsize=(20, len(channels) * 3))

    for i, channel in enumerate(channels):
        plt.subplot(len(channels), 1, i + 1)
        plt.plot(data[channel])
        plt.title(f"Channel {channel + 1}")
        plt.xlabel("Time Points")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


data_path = "*.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

data = data[:, :-2, :]
print(data.shape)

eeg_data = data[2]

# channels_to_visualize = [i for i in range(eeg_data.shape[0])]
# visualize_eeg_channels(eeg_data, channels_to_visualize)
