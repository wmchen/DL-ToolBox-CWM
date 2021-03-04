"""
Collect information from environment
Programmer: Weiming Chen
Date: 2020.12
"""
import os
import subprocess
import sys
from collections import defaultdict

import cv2
import torch
import torchvision

from dl_toolbox_cwm import __version__


def collect_env():
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    env_info['TorchVision'] = torchvision.__version__

    env_info['OpenCV'] = cv2.version.opencv_version

    env_info['DL-ToolBox-CWM'] = __version__

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')

