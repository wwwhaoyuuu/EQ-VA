import os
import bisect
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List

list_path = List[Path]


class SingleShockDataset(Dataset):
    def __init__(self, folder_path: Path, valance, stage, sub_start: int):
        self.__folder_path = folder_path
        self.__valance = valance
        self.__stage = stage
        self.sub_start = sub_start

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__data = np.load(os.path.join(str(self.__folder_path), f'{self.__valance}_{self.__stage}_data.npy'))
        self.__label = np.load(os.path.join(str(self.__folder_path), f'{self.__valance}_{self.__stage}_label.npy'))
        self.__label = self.__label + self.sub_start
        self.ch_names = np.load(os.path.join(str(self.__folder_path), 'channel_names.npy'), allow_pickle=True)

        self.__length = self.__data.shape[0]

    def __len__(self):
        return self.__length

    def __getitem__(self, idx):
        return self.__data[idx], self.__label[idx]

    def get_ch_names(self):
        return self.ch_names

    def get_classes(self):
        return len(np.unique(self.__label))


class ShockDataset(Dataset):
    def __init__(self, folder_paths: list_path, valance: list, stage: str, sub_start: int):
        self.__folder_paths = folder_paths
        self.__valance = valance
        self.__stage = stage
        self.__sub_start = sub_start

        self.__datasets = []
        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, valance, self.__stage, self.__sub_start)
                           for file_path, valance in zip(self.__folder_paths, self.__valance)]

        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()

    def get_classes(self):
        return self.__datasets[0].get_classes()


if __name__ == '__main__':
    # using example
    dataset_train = [
        [Path("../Dataset/SEED_PI/window5_step1")],
        [Path("../Dataset/FACED_PI/window5_step1")]
    ]
    valance_train = [
        ['positive'],
        ['positive']
    ]