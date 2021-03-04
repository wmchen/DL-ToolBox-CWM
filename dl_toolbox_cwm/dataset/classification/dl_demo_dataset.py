"""
DL Demo Dataset
Programmer: Weiming Chen
Date: 2021.1
"""
import os
import json
import copy
import cv2
from dl_toolbox_cwm.dataset.classification.base_dataset_cls import BaseDataset_CLS


class DL_Demo_Dataset(BaseDataset_CLS):
    CLASSES = [
        'fish',  # 1
        'chicken',  # 2
        'ostrich',  # 3
        'bird',  # 4
        'eagle',  # 5
        'owl',  # 6
        'gecko',  # 7
        'toad',  # 8
        'frog',  # 9
        'turtle',  # 10
        'lizard',  # 11
        'crocodile',  # 12
        'dinosaur',  # 13
        'snake',  # 14
        'fossil',  # 15
        'spider',  # 16
        'scorpion',  # 17
        'table',  # 18
        'centipede',  # 19
        'peacock',  # 20
        'duck',  # 21
        'goose',  # 22
        'elephant',  # 23
        'hedgehog',  # 24
        'platypus',  # 25
        'kangaroo',  # 26
        'koala',  # 27
        'marmot',  # 28
        'jellyfish',  # 29
        'coral',  # 30
        'sea snake',  # 31
        'conch',  # 32
        'snail',  # 33
        'chair',  # 34
        'lobster',  # 35
        'hermit crab',  # 36
        'penguin',  # 37
        'whale',  # 38
        'walrus',  # 39
        'sea lions',  # 40
        'dog',  # 41
        'wolf',  # 42
        'fox',  # 43
        'cat',  # 44
        'leopard',  # 45
        'lion',  # 46
        'tiger',  # 47
        'bear',  # 48
        'mongoose',  # 49
        'dragonfly',  # 50
        'butterfly',  # 51
        'starfish',  # 52
        'sea urchin',  # 53
        'rabbit',  # 54
        'squirrel',  # 55
        'zebra',  # 56
        'pig',  # 57
        'hippopotamus',  # 58
        'cow',  # 59
        'sheep',  # 60
        'camel',  # 61
        'alpaca',  # 62
        'watch',  # 63
        'pangolin',  # 64
        'sloth',  # 65
        'baboon',  # 66
        'chimpanzee',  # 67
        'monkey',  # 68
        'fire balloon',  # 69
        'Raccoon',  # 70
        'panda',  # 71
        'abacus',  # 72
        'accordion',  # 73
        'guitar',  # 74
        'aircraft carrier',  # 75
        'airplane',  # 76
        'airship',  # 77
        'ambulance',  # 78
    ]

    def load_annotation(self):
        data_infos = []
        with open(self.ann_file, 'r') as f:
            ann = json.load(f)
        paths = ann['path']
        labels = ann['label']
        for i in range(len(paths)):
            info = {
                'filename': paths[i],
                'label': labels[i]
            }
            data_infos.append(info)
        return data_infos

    def prepare_data(self, idx):
        data_bak = copy.deepcopy(self.data_infos[idx])
        data_path = os.path.join(self.data_prefix, data_bak['filename'])
        label = data_bak['label']
        img = cv2.imread(data_path)
        img = self.transform(img)
        return img, label
