import scipy.io
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import glob  # 用于查找文件路径
import re

# 设置数据和标签文件的路径
data_folder = 'E:\\1DomainCrossPI\AffectiveEEGPI\Features_Extract\SEED_Features\DE_mat/'
label_path = 'E:\\1DomainCrossPI\AffectiveEEGPI\Features_Extract\SEED_Features\DE_mat\label.mat'

# 加载包含标签的.mat文件
label_data = scipy.io.loadmat(label_path)
labels = label_data['label'][0]  # 提取标签数组

n_channels = 62  # 通道数
window_size = 5  # 窗口大小


# 函数：保存数据到指定路径
def save_data(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# 获取文件夹中所有的.mat文件
mat_files = glob.glob(data_folder + '*.mat')

# 标签
positive_label = []
neutral_label = []
negative_label = []

# 统计每个.mat文件中Total samples的最小值
min_total_samples = 1000000
for file_path in mat_files:
    if 'label.mat' in file_path:
        continue  # 跳过标签文件本身
    data = scipy.io.loadmat(file_path)  # 加载每个.mat文件中的EEG数据
    for key in data.keys():

        if re.match(r'de\d+', key):  # 匹配以任意单词字符开始，后接"_eeg"和数字的键
            eeg_data = data[key]
            total_samples = eeg_data.shape[1]
            # print(eeg_data.shape)
            if total_samples < min_total_samples:
                min_total_samples = total_samples
                # 计算可以整除的最大样本数
                min_total_samples = min_total_samples - min_total_samples % window_size
    data_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 把标签存储到positive_label、neutral_label、negative_label里面
    positive_label.append(data_file_name)
    neutral_label.append(data_file_name)
    negative_label.append(data_file_name)

print('Valid samples:', min_total_samples)
n_windows = min_total_samples // window_size
# 数据
positive_data = np.empty((len(positive_label), n_channels, (int)(min_total_samples / window_size), window_size, 5))
neutral_data = np.empty((len(neutral_label), n_channels, (int)(min_total_samples / window_size), window_size, 5))
negative_data = np.empty((len(negative_label), n_channels, (int)(min_total_samples / window_size), window_size, 5))

count = 0
# 遍历每个.mat文件
for file_path in mat_files:
    if 'label.mat' in file_path:
        continue  # 跳过标签文件本身
    data = scipy.io.loadmat(file_path)  # 加载每个.mat文件中的EEG数据

    # print(data_file_name)
    # 遍历文件中所有的键，寻找匹配 EEG 数据集的键
    positive_eeg = np.empty((n_channels, (int)(min_total_samples / window_size), window_size))
    neutral_eeg = np.empty((n_channels, (int)(min_total_samples / window_size), window_size))
    negative_eeg = np.empty((n_channels, (int)(min_total_samples / window_size), window_size))
    for key in data.keys():
        reshaped_data = None
        if re.match(r'de\d+', key):  # 匹配以de开始，后接数字的键
            eeg_data = data[key]
            index = int(re.findall(r'\d+', key)[0])  # 提取出数字部分，确定标签索引
            if index <= len(labels):  # 避免索引越界
                label_index = labels[index - 1]  # 提取对应的情感标签
                label_folder = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}[label_index]
                # 使用min_total_samples
                valid_samples = min_total_samples
                eeg_data = eeg_data[:, :valid_samples:]  # 舍弃不足以形成一个完整窗口的数据
                n_windows = valid_samples // window_size
                reshaped_data = eeg_data.reshape(n_channels, n_windows, window_size, 5)
                if label_folder == 'Positive':
                    positive_eeg = reshaped_data
                if label_folder == 'Neutral':
                    neutral_eeg = reshaped_data
                if label_folder == 'Negative':
                    negative_eeg = reshaped_data

    positive_data[count, :, :, :, :] = positive_eeg
    neutral_data[count, :, :, :, :] = neutral_eeg
    negative_data[count, :, :, :, :] = negative_eeg
    count += 1
#调整data格式
positive_data=positive_data.transpose(0, 2, 1, 3, 4)
positive_data=positive_data.reshape(count*n_windows,n_channels,window_size,5)
neutral_data=neutral_data.transpose(0, 2, 1, 3, 4)
neutral_data=neutral_data.reshape(count*n_windows,n_channels,window_size,5)
negative_data=negative_data.transpose(0, 2, 1, 3, 4)
negative_data=negative_data.reshape(count*n_windows,n_channels,window_size,5)

# label转为numpy格式并重复n_windows次
positive_label = np.array(positive_label)
positive_label=np.repeat(positive_label,n_windows)
neutral_label = np.array(neutral_label)
neutral_label=np.repeat(neutral_label,n_windows)
negative_label = np.array(negative_label)
negative_label=np.repeat(negative_label,n_windows)
# 切分数据集
positive_data_train, positive_data_test, positive_label_train, positive_label_test = train_test_split(positive_data,
                                                                                                      positive_label,
                                                                                                      test_size=0.1,
                                                                                                      random_state=42)
neutral_data_train, neutral_data_test, neutral_label_train, neutral_label_test = train_test_split(neutral_data,
                                                                                                  neutral_label,
                                                                                                  test_size=0.1,
                                                                                                  random_state=42)
negative_data_train, negative_data_test, negative_label_train, negative_label_test = train_test_split(negative_data,
                                                                                                      negative_label,
                                                                                                      test_size=0.1,
                                                                                                      random_state=42)
# 打印数据形状
print('positive_data_train:', positive_data_train.shape, 'positive_label_train:', positive_label_train.shape)
print('positive_data_test:', positive_data_test.shape, 'positive_label_test:', positive_label_test.shape)
print('neutral_data_train:', neutral_data_train.shape, 'neutral_label_train:', neutral_label_train.shape)
print('neutral_data_test:', neutral_data_test.shape, 'neutral_label_test:', neutral_label_test.shape)
print('negative_data_train:', negative_data_train.shape, 'negative_label_train:', negative_label_train.shape)
print('negative_data_test:', negative_data_test.shape, 'negative_label_test:', negative_label_test.shape)

# 为每个情绪保存数据
save_data(positive_data_train, 'DE\\Positive\\train_data.pkl')
save_data(positive_data_test, 'DE\\Positive\\val_data.pkl')
save_data(positive_label_train, 'DE\\Positive\\train_label.pkl')
save_data(positive_label_test, 'DE\\Positive\\val_label.pkl')
save_data(neutral_data_train, 'DE\\Neutral\\train_data.pkl')
save_data(neutral_data_test, 'DE\\Neutral\\val_data.pkl')
save_data(neutral_label_train, 'DE\\Neutral\\train_label.pkl')
save_data(neutral_label_test, 'DE\\Neutral\\val_label.pkl')
save_data(negative_data_train, 'DE\\Negative\\train_data.pkl')
save_data(negative_data_test, 'DE\\Negative\\val_data.pkl')
save_data(negative_label_train, 'DE\\Negative\\train_label.pkl')
save_data(negative_label_test, 'DE\\Negative\\val_label.pkl')
