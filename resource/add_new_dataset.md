# Add new dataset

Author: Weiming Chen

Email: wm_chen@yeah.net

## The step of adding new dataset

### 1. Add dataset to directory

e.g.: Add 'custom_dataset' to the directory of data/custom_dataset

### 2. Registe dataset

e.g.:

Step1: Create dl_toolbox_cwm/classification/custom_dataset.py

```python
# dl_toolbox_cwm/classification/custom_dataset.py
import json
from dl_toolbox_cwm.dataset.classification.base_dataset_cls import BaseDataset_CLS

class Custom_Dataset(BaseDataset_CLS):
    CLASSES = ['cls1', 'cls2', 'cls3']
    
    def load_annotations(self):
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
```

Step2: Modify dl_toolbox_cwm/classification/\_\_init\_\_.py

```python
from .base_dataset_cls import BaseDataset_CLS
from .cifar import CIFAR10
from .mnist import MNIST
from .dl_img_dataset import DL_Demo_Dataset
from .custom_dataset import Custom_Dataset

__all__ = [
    'BaseDataset_CLS', 'CIFAR10', 'MNIST', 'DL_Demo_Dataset', 'Custom_Dataset'
]
```

### 3. Create config file

## Contact

This repository is currently maintained by Weiming Chen.

If you have some other method need to be supported, contact me through Email: wm_chen@yeah.net