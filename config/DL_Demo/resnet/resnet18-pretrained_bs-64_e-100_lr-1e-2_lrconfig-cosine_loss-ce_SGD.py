import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dl_toolbox_cwm.dataset.classification import DL_Demo_Dataset
from dl_toolbox_cwm.model.classification import ResNet18


# root path
root_path = '/home/weiming/DL-ToolBox-CWM'  # TODO: rewrite as your root path

# dataset settings
dataset_name = 'DL_Demo_Dataset'

# Hyper-parameters
epoch = 100
batch_size = 64

# pipeline
train_pipeline = [
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
val_pipeline = [
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
test_pipeline = [
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
# dataset
train_config = dict(
    data_prefix='data/dl_demo/train',
    ann_file='data/dl_demo/train.json',
    shuffle=True
)
val_config = dict(
    data_prefix='data/dl_demo/test',
    ann_file='data/dl_demo/test.json',
    shuffle=True
)
test_config = dict(
    data_prefix='data/dl_demo/test',
    ann_file='data/dl_demo/test.json',
    shuffle=False
)
train_set = DL_Demo_Dataset(
    data_prefix=os.path.join(root_path, train_config['data_prefix']),
    pipeline=train_pipeline,
    ann_file=None if train_config['ann_file'] is None else os.path.join(root_path, train_config['ann_file']),
    is_train=True
)
val_set = DL_Demo_Dataset(
    data_prefix=os.path.join(root_path, val_config['data_prefix']),
    pipeline=val_pipeline,
    ann_file=None if train_config['ann_file'] is None else os.path.join(root_path, val_config['ann_file']),
    is_train=False
)
test_set = DL_Demo_Dataset(
    data_prefix=os.path.join(root_path, test_config['data_prefix']),
    pipeline=test_pipeline,
    ann_file=None if train_config['ann_file'] is None else os.path.join(root_path, test_config['ann_file']),
    is_train=False
)

# model
model = ResNet18(num_classes=78, pretrained='pretrained_model/ResNet/resnet18.pth')

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch*int(len(train_set)/batch_size))  # optional

# validate config (optional, default=1)
val_interval = 1

# checkpoint saving (optional, default=1)
checkpoint_interval = 1

# log config (optional, default=10)
log_interval = 10
