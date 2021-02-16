from __future__ import print_function

import os

import numpy as np
import scipy.io
import torch
import torch.utils.data as data

from classes.data.DataAugmenter import DataAugmenter


class ColorCheckerDataset(data.Dataset):

    def __init__(self, train: bool = True, folds_num: int = 1):

        self.__train = train
        self.__da = DataAugmenter()

        base_path_to_data = "dataset"
        path_to_folds = os.path.join(base_path_to_data, "folds.mat")
        path_to_metadata = os.path.join(base_path_to_data, "color_checker_metadata.txt")
        self.__path_to_data = os.path.join(base_path_to_data, "ndata")
        self.__path_to_label = os.path.join(base_path_to_data, "nlabel")

        folds = scipy.io.loadmat(path_to_folds)
        img_idx = folds['tr_split' if self.__train else 'te_split'][0][folds_num][0]

        metadata = open(path_to_metadata, 'r').readlines()
        self.__fold_data = [metadata[i - 1] for i in img_idx]

    def __getitem__(self, index: int) -> tuple:

        file_name = self.__fold_data[index].strip().split(' ')[1]
        img = np.array(np.load(os.path.join(self.__path_to_data, file_name + '.npy')), dtype='float32')
        illuminant = np.array(np.load(os.path.join(self.__path_to_label, file_name + '.npy')), dtype='float32')

        if self.__train:
            img, illuminant = self.__da.augment(img, illuminant)
        else:
            img = self.__da.crop(img)

        img = np.clip(img, 0.0, 65535.0) * (1.0 / 65535)

        # BGR to RGB
        img = img[:, :, ::-1]
        img = np.power(img, (1.0 / 2.2))

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        if not self.__train:
            img = img.type(torch.FloatTensor)

        return img, illuminant, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
