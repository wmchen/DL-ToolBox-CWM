# Deep Learning ToolBox (DL-ToolBox-CWM)

Author: Weiming Chen

Email: wm_chen@yeah.net



## Introduction

The motivation of me to start this project is owing to the difficulty for some existing senior packages of PyTorch to adapt the new graphics card 'GeForce RTX 3090'. I just use the original code to build the project. And can compile on 'GeForce RTX 3090' successfully.

This project can be considered as a senior substitute of OpenMMLab projects due to current OpenMMLab projects can not compile on 'GeForce RTX 3090' successfully.

Still updating!



## License

This project is released under the [Apache 2.0 license](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/LICENSE).



## Baseline

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] ResNeSt



## Installation

Python 3.6+

Anaconda (recommended)

### 1 For stable graphics cards

- PyTorch

- Install dl_toolbox_cwm

  ```sh
  git clone https://github.com/Wei-ming-Chen/DL-ToolBox-CWM.git
  cd ./DL-ToolBox-CWM
  pip install -r requirements.txt
  python setup.py install
  ```

### 2 For GeForce RTX 3090

- PyTorch

  ```sh
  pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html
  ```

  or
  
  ```sh
  conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly
  ```
  
- Install dl_toolbox_cwm

  ```sh
  git clone https://github.com/Wei-ming-Chen/DL-ToolBox-CWM.git
  cd ./DL-ToolBox-CWM
  pip install -r requirements.txt
  python setup.py install
  ```



## Getting Start

The [naming_conventions](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/resource/naming_convention.md) of specific files.

Tutorial of [add_new_dataset](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/resource/add_new_dataset.md).

Tutorial of [training](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/resource/train.md).

Tutorial of [config_file](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/resource/config_file.md).



## Dataset available

See in [get_dataset](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/resource/get_dataset.md).



## Pretrained model available

See in [get_pretrained_model](https://github.com/Wei-ming-Chen/DL-ToolBox-CWM/blob/main/resource/get_pretrained_model.md).



## Acknowledgement

DL-ToolBox-CWM is an open project which is built to solve the failure of some senior package compiling on the new graphics card 'GeForce RTX 3090'. I sincerely appreciate the great work of OpenMMLab and the great researchers such as Kai Chen, Jintao Lin and so on.



## Reference projects

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.



## Contact

This repository is currently maintained by Weiming Chen.

If you have any questions, contact me through Email: wm_chen@yeah.net

