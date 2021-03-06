# Training

Author: Weiming Chen

Email: wm_chen@yeah.net



## Introduction

A uniform interface for training models was built for the convenience causes. By using that uniform interface, you just need to define a specific model in a configuration file, then you can train this specific model by a simple command line code.



## For classification

For the classification task of computer vision, the training interface is `tool/train_cls.py`.

### 1. Using GPU

#### 1.1 Single GPU mode

If you want train a ResNet18 model(take CIFAR10 dataset as an example) with a single GPU, you may code the following command line in terminal:

```bash
python tool/train_cls.py config/CIFAR-10/resnet/resnet18-pretrained_bs-64_e-100_lr-1e-2_lrconfig-cosine_loss-ce_SGD.py --work-dir result/CIFAR-10/resnet --top-k 5 --single-gpu --single-gpu-id 0 --num-workers 4 --val-bs 50
```

That is mean:

- You will train the model which was defined in configuration file `config/CIFAR-10/resnet/resnet18-pretrained_bs-64_e-100_lr-1e-2_lrconfig-cosine_loss-ce_SGD.py`
- The argument of `--work-dir result/CIFAR-10/resnet` means that the root working directory is `result/CIFAR-10/resnet`. Actually, a directory named `result/CIFAR-10/resnet/resnet18-pretrained_bs-64_e-100_lr-1e-2_lrconfig-cosine_loss-ce_SGD` will be created automatically if it is not already existed, and all the checkpoint files, logging file and the backup of the corresponding configuration file will save in this directory.
- The argument of `--top-k 5` means that top-1 to top-5 accuracy of the model will be evaluated during the training and validation process.
- The argument of `--single-gpu` means you will train the model by using a single GPU card.
- The argument of `--single-gpu-id 0` means the index of the using GPU is 0.
- The argument of `--num-workers 4` means the number workers you will use in the DataLoader. It is highly recommend that set `--num-workers` as two times of the number of GPU cards in your computer or server.
- The argument of `--val-bs` means the batch size of in validation process is 50.

#### 1.2 Multi GPU mode

Still updating.

### 2. Using CPU

If you want train a ResNet18 model(take CIFAR10 dataset as an example) with CPU, you may code the following command line in terminal:

```bash
python tool/train_cls.py config/CIFAR-10/resnet/resnet18-pretrained_bs-64_e-100_lr-1e-2_lrconfig-cosine_loss-ce_SGD.py --work-dir result/CIFAR-10/resnet --top-k 5 --num-workers 4 --val-bs 50
```



## Contact

This repository is currently maintained by Weiming Chen.

If you have some other method need to be supported, contact me through Email: wm_chen@yeah.net



