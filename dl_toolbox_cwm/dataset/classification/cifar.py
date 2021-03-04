"""
CIFAR Dataset
Programmer: Weiming Chen
Date: 2020.12
"""
import os
import copy
import pickle
import numpy as np
from dl_toolbox_cwm.dataset.classification.base_dataset_cls import BaseDataset_CLS


class CIFAR10(BaseDataset_CLS):
    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def load_dataset(self):
        train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                       'data_batch_4', 'data_batch_5']
        data_info = {}
        if self.is_train:
            for i, batch in enumerate(train_batch):
                with open(os.path.join(self.data_prefix, batch), 'rb') as f:
                    data_info['train_{}'.format(i+1)] = pickle.load(f, encoding='bytes')
        else:
            with open(os.path.join(self.data_prefix, 'test_batch'), 'rb') as f:
                data_info['test'] = pickle.load(f, encoding='bytes')
        return data_info

    def load_annotation(self):
        data_info = self.load_dataset()
        img = []
        label = []
        if self.is_train:
            img = []
            label = []
            for i in range(5):
                img.append(data_info['train_{}'.format(i+1)][b'data'])
                label.append(data_info['train_{}'.format(i+1)][b'labels'])
        else:
            img.append(data_info['test'][b'data'])
            label.append(data_info['test'][b'labels'])
        img = np.concatenate(img)
        label = np.concatenate(label)
        data_infos = {
            'image': img,
            'label': label
        }
        return data_infos

    def prepare_data(self, idx):
        img_bak = copy.deepcopy(self.data_infos['image'][idx])
        img = img_bak.reshape((3, 32, 32))
        img = img.transpose((1, 2, 0))
        img = self.transform(img)
        label = copy.deepcopy(self.data_infos['label'][idx])
        return img, label

    def __len__(self):
        return len(self.data_infos['image'])
