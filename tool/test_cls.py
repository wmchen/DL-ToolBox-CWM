"""
Train a model for classification task
Programmer: Weiming Chen
Date: 2021.1
"""
import argparse
import os
import shutil
import time
import warnings
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dl_toolbox_cwm.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model for classification task')
    parser.add_argument('config', help='path of training config file. (use relative path: config/...)')
    parser.add_argument('checkpoint', help='path of checkpoint file. (use relative path)')
    parser.add_argument('--work-dir', help='the direction to save logs and models.  (use relative path: result/...)')

    parser.add_argument('--top-k', type=int, default=5, help='evaluate top-1 to top-k accuracy.')

    parser.add_argument('--single-gpu', action='store_true', help='only use one gpu')
    parser.add_argument('--single-gpu-id', type=int, default=0, help='the id of gpu in single-gpu mode')
    parser.add_argument('--num-workers', type=int, default=4, help='The number of workers.')
    # TODO: multi-gpu

    parser.add_argument('--test-bs', default=20, type=int, help='the batch size of testing.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('DL-ToolBox-CWM') + len('DL-ToolBox-CWM')]

    # load config
    if not args.config.startswith('config'):
        warnings.warn('Relative path like config/... is recommended.')
    if not os.path.exists(args.config):
        cfg_path = os.path.join(root_path, args.config)
        if not os.path.exists(cfg_path):
            raise FileExistsError('error path: {}'.format(cfg_path))
    else:
        cfg_path = args.config
    Config = LoadConfig(cfg_path)
    cfg_dict = Config.load_cfg_file()

    # checkpoint info
    if not os.path.exists(args.checkpoint):
        ckpt_path = os.path.join(root_path, args.checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileExistsError('error path: {}'.format(ckpt_path))
    else:
        ckpt_path = args.checkpoint
    cfg_dict['checkpoint_path'] = ckpt_path

    # work-dir
    if args.work_dir is not None:
        if not os.path.exists(args.work_dir):
            work_dir = os.path.join(root_path, args.work_dir)
            if not os.path.exists(work_dir):
                raise FileExistsError('error path: {}'.format(work_dir))
        else:
            work_dir = args.work_dir
        if not os.path.isdir(work_dir):
            raise IsADirectoryError('--work-dir must get a directory, but got: {}'.format(work_dir))
        shutil.copy(cfg_path, os.path.join(work_dir, cfg_path.split('/')[-1]))
    else:
        work_dir = None
    cfg_dict['work_dir'] = work_dir

    # init logger
    if work_dir is not None:
        now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, now_time + '_test.log')
        json_log_file = os.path.join(work_dir, now_time + '_test.json')
    else:
        log_file = None
        json_log_file = None
    cfg_dict['log_file'] = log_file
    logger = get_root_logger('DL-ToolBox-CWM.tool.test_cls', log_file)
    json_log_info = []

    # log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    # GPU config
    if args.single_gpu:
        gpu_id = args.single_gpu_id
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        gpu_id = None
        device = None
    cfg_dict['gpu_id'] = gpu_id

    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True

    # log config info
    logger.info(Config.pretty_text(cfg_dict))

    # init dataloader
    test_loader = DataLoader(
        dataset=cfg_dict['test_set'],
        batch_size=args.test_bs,
        shuffle=cfg_dict['test_config']['shuffle'],
        num_workers=args.num_workers,
        pin_memory=True
    )

    # load checkpoint
    checkpoint = torch.load(ckpt_path)

    # model
    model = cfg_dict['model']
    model.load_state_dict(checkpoint['state_dict'])
    if device is not None:
        model.to(device)

    # loss function
    criterion = cfg_dict['criterion']

    model.eval()
    eval_loss = 0
    eval_correct = np.zeros((args.top_k, 1))
    eval_total = len(cfg_dict['test_set'])
    progress_bar = ProgressBar(eval_total)
    for i, data in enumerate(test_loader):
        progress_bar.start_timing()
        img, label = data
        if device is not None:
            img = img.to(device)
            label = label.to(device)
        label_resize = label.view(-1, 1)
        out = model(img)
        [_, num_classes] = out.shape
        loss = criterion(out, label)

        eval_loss += loss.item()
        for j in range(args.top_k):
            k = j + 1
            if k > num_classes:
                eval_correct[j] = -1
            else:
                _, pred = out.topk(k, 1, True, True)
                eval_correct[j] += torch.eq(pred, label_resize).cpu().sum().float().item()
        rest = progress_bar.total - progress_bar.current
        if rest >= args.test_bs:
            progress_bar.update_step(args.test_bs)
        else:
            progress_bar.update_step(rest)
    del progress_bar
    logger.info('loss: {:.4f}'.format(eval_loss / eval_total))
    for i in range(args.top_k):
        k = i + 1
        if eval_correct[i] != -1:
            logger.info('top-{}: {:.4f}'.format(k, int(eval_correct[i])/eval_total))
    if json_log_file is not None:
        json_log_info.append({
            'checkpoint': ckpt_path,
            'mode': 'test',
            'loss': eval_loss / eval_total,
            'accuracy': [int(eval_correct[i])/eval_total for i in range(args.top_k) if eval_correct[i] != -1]
        })
        logger.info('saving log info to json file...')
        with open(json_log_file, 'w') as f:
            json.dump(json_log_info, f)


if __name__ == '__main__':
    main()
