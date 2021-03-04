"""
Base dataset for classification task
Programmer: Weiming Chen
Date: 2020.12
"""
import os
import copy
from abc import ABCMeta, abstractmethod
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset_CLS(Dataset, metaclass=ABCMeta):
    """
    Base dataset for classification task
    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of transform
                         Example:
                             >>> pipeline = [
                             >>>     transforms.RandomCrop(224, padding=4),
                             >>>     transforms.RandomHorizontalFlip(p=0.5)
                             >>> ]
        ann_file (str): the path of annotation file(json).
                        The structure of annotation file:
                        >>> ann = {
                        >>>     'filename': [],
                        >>>     'label': [],
                        >>> }
        is_train (bool): in train mode or test mode
    """
    CLASSES = None

    def __init__(self, data_prefix, pipeline, ann_file, is_train=True):
        super(BaseDataset_CLS, self).__init__()
        self.data_prefix = data_prefix
        self.transform = transforms.Compose(pipeline)
        self.ann_file = ann_file
        self.is_train = is_train
        self.data_infos = self.load_annotation()

    @abstractmethod
    def load_annotation(self):
        # method need to be re-writen
        pass

    @property
    def class_to_idx(self):
        # read-only method
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def prepare_data(self, idx):
        data_bak = copy.deepcopy(self.data_infos[idx])
        data_path = os.path.join(self.data_prefix, data_bak['filename'])
        label = data_bak['label']
        img = Image.open(data_path)
        img = self.transform(img)
        return img, label

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_infos)
