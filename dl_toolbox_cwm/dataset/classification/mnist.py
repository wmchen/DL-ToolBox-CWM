"""
MNIST Dataset
Programmer: Weiming Chen
Date: 2021.1
"""
import os
import gzip
import copy
import numpy as np
from dl_toolbox_cwm.dataset.classification.base_dataset_cls import BaseDataset_CLS


class MNIST(BaseDataset_CLS):
    CLASSES = [
        '0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9'
    ]
    train_image = 'train-images-idx3-ubyte.gz'
    train_label = 'train-labels-idx1-ubyte.gz'
    test_image = 't10k-images-idx3-ubyte.gz'
    test_label = 't10k-labels-idx1-ubyte.gz'

    def load_annotation(self):
        data_infos = {}
        if self.is_train:
            with gzip.open(os.path.join(self.data_prefix, self.train_label), 'rb') as f:
                data_infos['label'] = np.frombuffer(f.read(), np.uint8, offset=8)
            with gzip.open(os.path.join(self.data_prefix, self.train_image), 'rb') as f:
                data_infos['image'] = np.frombuffer(f.read(), np.uint8, offset=16).reshape((len(data_infos['label']), 28, 28))
        else:
            with gzip.open(os.path.join(self.data_prefix, self.test_label), 'rb') as f:
                data_infos['label'] = np.frombuffer(f.read(), np.uint8, offset=8)
            with gzip.open(os.path.join(self.data_prefix, self.test_image), 'rb') as f:
                data_infos['image'] = np.frombuffer(f.read(), np.uint8, offset=16).reshape((len(data_infos['label']), 28, 28))
        return data_infos

    def prepare_data(self, idx):
        img = copy.deepcopy(self.data_infos['image'][idx])
        img_to_3channel = np.array([img, img, img])  # 28x28 -> 3x28x28
        img_to_3channel = img_to_3channel.transpose((1, 2, 0))  # 3x28x28 -> 28x28x3
        img = self.transform(img_to_3channel)
        label = copy.deepcopy(self.data_infos['label'][idx])
        label = int(label)  # uint8 -> int
        return img, label

    def __len__(self):
        return len(self.data_infos['image'])
