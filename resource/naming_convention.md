# Naming Convention

Author: Weiming Chen

Email: wm_chen@yeah.net



## For config file

The format of config file is '.py', and we highly recommend you to follow the naming convention.

#### 1. Naming convention

**{model}\_{other_setting}\_bs-{batchsize}\_e-{epoch}\_lr-{lr}\_lrconfig-{lr_policy}\_loss-{loss_function}\_{optimizer}.py**

For expamle, a config file named 'resnet18_bs-16_e-100_lr-1e-2_lrconfig-step_loss-ce_SGD.py'.

The corresponding meaning of this config file is: 

model name: resnet18

batch size: 16

epoch: 200

initial learining rate: 1e-2 **(Notice: do not name as 0.01)**

adjusting policy of learining rate: step

loss function: cross entropy

optimizer: SGD

#### 2. Other examples

resnet50_bs-64_e-100_lr-1e-2_lrconfig-cosine_loss-ce_SGD.py

resnet50_bs-32_e-200_lr-1e-2_loss-ce_Adam.py

......



## Contact

This repository is currently maintained by Weiming Chen.

If you have some other method need to be supported, contact me through Email: wm_chen@yeah.net