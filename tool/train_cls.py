"""
Train a model for classification task
Programmer: Weiming Chen
Date: 2020.12
"""
import argparse
import os
import shutil
import warnings
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dl_toolbox_cwm.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for classification task')
    parser.add_argument('config', help='path of training config file. (use relative path: config/...)')
    parser.add_argument('--work-dir', help='the direction to save logs and models.  (use relative path: result/...)')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from.  (use relative path)')

    parser.add_argument('--top-k', type=int, default=5, help='evaluate top-1 to top-k accuracy.')

    parser.add_argument('--single-gpu', action='store_true', help='only use one gpu')
    parser.add_argument('--single-gpu-id', type=int, default=0, help='the id of gpu in single-gpu mode')
    parser.add_argument('--num-workers', type=int, default=4, help='The number of workers.')
    # TODO: multi-gpu

    parser.add_argument('--val-bs', default=20, type=int, help='the batch size of validation.')

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
    cfg_file_name = cfg_path.split('/')[-1]

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
        work_dir = os.path.join(work_dir, os.path.splitext(cfg_file_name)[0])
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        shutil.copy(cfg_path, os.path.join(work_dir, cfg_file_name))
    else:
        work_dir = None
    cfg_dict['work_dir'] = work_dir

    # resume-from
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            resume_from = os.path.join(root_path, args.resume_from)
            if not os.path.exists(resume_from):
                raise FileExistsError('error path: {}'.format(resume_from))
        else:
            resume_from = args.resume_from
    else:
        resume_from = None
    cfg_dict['resume_from'] = resume_from

    # init logger
    if work_dir is not None:
        now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, now_time+'_trainval.log')
        json_log_file = os.path.join(work_dir, now_time+'_trainval.json')
    else:
        log_file = None
        json_log_file = None
    cfg_dict['log_file'] = log_file
    logger = get_root_logger('DL-ToolBox-CWM.tool.train_cls', log_file)
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
    train_loader = DataLoader(
        dataset=cfg_dict['train_set'],
        batch_size=cfg_dict['batch_size'],
        shuffle=cfg_dict['train_config']['shuffle'],
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=cfg_dict['val_set'],
        batch_size=args.val_bs,
        shuffle=cfg_dict['val_config']['shuffle'],
        num_workers=args.num_workers,
        pin_memory=True
    )
    train_iter_per_epoch = len(cfg_dict['train_set']) / cfg_dict['batch_size']
    if not train_iter_per_epoch % 1 == 0:
        train_iter_per_epoch = int(train_iter_per_epoch) + 1
    else:
        train_iter_per_epoch = int(train_iter_per_epoch)

    # model
    model = cfg_dict['model']
    if device is not None:
        model.to(device)

    # loss function
    criterion = cfg_dict['criterion']

    # optimizer
    optimizer = cfg_dict['optimizer']
    if 'scheduler' in cfg_dict:
        scheduler = cfg_dict['scheduler']
    else:
        scheduler = None

    # other parameters
    try:
        val_interval = cfg_dict['val_interval']
    except:
        val_interval = 1
    try:
        log_interval = cfg_dict['log_interval']
    except:
        log_interval = 1
    try:
        checkpoint_interval = cfg_dict['checkpoint_interval']
    except:
        checkpoint_interval = 1

    for epoch in range(cfg_dict['epoch']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        timer = True
        for i, data in enumerate(train_loader):
            if timer:
                timer = False
                start = time.time()
            optimizer.zero_grad()
            img, label = data
            if device is not None:
                img = img.to(device)
                label = label.to(device)
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            _, pred = torch.max(out.data, 1)
            train_total += label.size(0)
            train_correct += pred.eq(label.data).cpu().sum()
            if (i+1) % log_interval == 0:
                end = time.time()
                timer = True
                lr = optimizer.param_groups[0]['lr']
                logger.info('Epoch [{}][{}/{}]\tlr: {}, elapsed: {:.3f}, loss: {:.4f}, train_acc: {:.4f}'.format(
                    epoch+1,
                    i+1,
                    train_iter_per_epoch,
                    lr,
                    end-start,
                    train_loss/train_total,
                    int(train_correct)/train_total))
                if json_log_file is not None:
                    json_log_info.append({
                        'epoch': epoch+1,
                        'mode': 'train',
                        'iter': i+1,
                        'loss': train_loss/train_total,
                        'accuracy': int(train_correct)/train_total
                    })
        if work_dir is not None:
            if (epoch+1) % checkpoint_interval == 0:
                # save checkpoint
                logger.info('saving checkpoint...')
                checkpoint = {
                    'env_info': env_info,
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if scheduler is not None:
                    checkpoint['scheduler'] = scheduler.state_dict()
                checkpoint_path = os.path.join(work_dir, 'epoch_{}.pth'.format(epoch+1))
                torch.save(checkpoint, checkpoint_path)
        if val_interval != 0 and (epoch+1) % val_interval == 0:
            model.eval()
            eval_loss = 0
            eval_correct = np.zeros((args.top_k, 1))
            eval_total = len(cfg_dict['val_set'])
            progress_bar = ProgressBar(eval_total)
            for i, data in enumerate(val_loader):
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
                if rest >= args.val_bs:
                    progress_bar.update_step(args.val_bs)
                else:
                    progress_bar.update_step(rest)
            del progress_bar
            logger.info('Epoch(val) [{}]\tloss:{:.4f}'.format(epoch+1, eval_loss / eval_total))
            for i in range(args.top_k):
                k = i + 1
                if eval_correct[i] != -1:
                    logger.info('Epoch(val) [{}]\ttop-{}: {:.4f}'.format(epoch+1, k, int(eval_correct[i]) / eval_total))
            if json_log_file is not None:
                json_log_info.append({
                    'epoch': epoch+1,
                    'mode': 'val',
                    'loss': eval_loss/eval_total,
                    'accuracy': [int(eval_correct[i])/eval_total for i in range(args.top_k) if eval_correct[i] != -1]
                })

        if json_log_file is not None:
            logger.info('saving log info to json file...')
            with open(json_log_file, 'w') as f:
                json.dump(json_log_info, f)


if __name__ == '__main__':
    main()
