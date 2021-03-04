"""
Analyze results
Programmer: Weiming Chen
Date: 2021.1
"""
import argparse
import os
import json
import shutil
import time
import collections

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dl_toolbox_cwm.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze models for classification task')
    parser.add_argument('--ckpt-dir', required=True, type=str, nargs='+',
                        help='directory of checkpoint file. (use relative path)')
    parser.add_argument('--work-dir', required=True, help='the direction to save logs and models.  (use relative path: result/...)')

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
    now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # ckpt-dir
    ckpt_dir_list = args.ckpt_dir
    for i, ckpt_dir in enumerate(ckpt_dir_list):
        if not os.path.exists(ckpt_dir):
            ckpt_dir = os.path.join(root_path, ckpt_dir)
            if not os.path.exists(ckpt_dir):
                raise FileExistsError('error path: {}'.format(ckpt_dir))
            else:
                ckpt_dir_list[i] = ckpt_dir

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
        work_dir = os.path.join(work_dir, now_time)
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
    else:
        work_dir = None

    # init logger
    if work_dir is not None:
        log_file = os.path.join(work_dir, now_time + '_analyze.log')
        json_log_file = os.path.join(work_dir, now_time + '_analyze.json')
    else:
        log_file = None
        json_log_file = None
    logger = get_root_logger('DL-ToolBox-CWM.tool.analyze_cls', log_file)
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

    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True

    trainval_log_info = {}
    test_info = {}
    for ckpt_dir in ckpt_dir_list:
        logger.info('checkpoint directory: {}'.format(ckpt_dir))
        config_file = None
        log_json_file = None
        ckpt_file = []
        for filename in os.listdir(ckpt_dir):
            if os.path.splitext(filename)[1] == '.pth':
                ckpt_file.append(filename)
            if os.path.splitext(filename)[1] == '.py':
                config_file = filename
            if os.path.splitext(filename)[1] == '.json':
                log_json_file = filename
        if config_file is None:
            raise FileExistsError('cannot find config file in: {}'.format(ckpt_dir))
        if log_json_file is None:
            raise FileExistsError('cannot find log file in: {}'.format(ckpt_dir))
        if len(ckpt_file) == 0:
            raise FileExistsError('cannot find checkpoint file in: {}'.format(ckpt_dir))
        ckpt_file.sort(key=lambda x: int(x[6:-4]))
        # load log file
        logger.info('load train-val log file: {}'.format(log_json_file))
        config_name = os.path.splitext(config_file)[0]
        trainval_log = read_trainval_log_file(os.path.join(ckpt_dir, log_json_file))
        trainval_log_info[config_name] = trainval_log
        if json_log_file is not None:
            statistic = {
                'log_type': 'analyze.trainval_log',
                'config_name': config_name,
            }
            trainval_log_statistic_result = trainval_log_statistic(trainval_log)
            for key in trainval_log_statistic_result.keys():
                statistic[key] = trainval_log_statistic_result[key]
            json_log_info.append(statistic)
            logger.info('saving log info to json file...')
            with open(json_log_file, 'w') as f:
                json.dump(json_log_info, f)
        # load config
        logger.info('load config file: {}'.format(config_file))
        shutil.copy(os.path.join(ckpt_dir, config_file), os.path.join(work_dir, config_file))  # copy config file
        Config = LoadConfig(os.path.join(work_dir, config_file))
        cfg_dict = Config.load_cfg_file()
        # init dataloader
        logger.info('load dataset')
        test_loader = DataLoader(
            dataset=cfg_dict['test_set'],
            batch_size=args.test_bs,
            shuffle=cfg_dict['test_config']['shuffle'],
            num_workers=args.num_workers,
            pin_memory=True
        )
        # model
        logger.info('load model')
        model = cfg_dict['model']
        if device is not None:
            model.to(device)
        # loss function
        logger.info('load loss function')
        criterion = cfg_dict['criterion']
        # test
        logger.info('start to test checkpoint file')
        test_info[config_name] = {}
        for ckpt in ckpt_file:
            logger.info('test: {}'.format(ckpt))
            test_info[config_name][ckpt] = {}
            checkpoint = torch.load(os.path.join(ckpt_dir, ckpt))
            model.load_state_dict(checkpoint['state_dict'])
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
            logger.info('[{}][{}] loss: {:.4f}'.format(config_name, ckpt, eval_loss/eval_total))
            for i in range(args.top_k):
                k = i + 1
                if eval_correct[i] != -1:
                    logger.info('[{}][{}] top-{}: {:.4f}'.format(config_name, ckpt, k, int(eval_correct[i])/eval_total))
            test_info[config_name][ckpt]['loss'] = eval_loss / eval_total
            test_info[config_name][ckpt]['accuracy'] = [int(eval_correct[i])/eval_total for i in range(args.top_k) if eval_correct[i] != -1]
            if json_log_file is not None:
                json_log_info.append({
                    'log_type': 'analyze.test_checkpoint',
                    'config_name': config_name,
                    'checkpoint': ckpt,
                    'mode': 'test',
                    'loss': eval_loss / eval_total,
                    'accuracy': [int(eval_correct[i])/eval_total for i in range(args.top_k) if eval_correct[i] != -1]
                })
                logger.info('saving log info to json file...')
                with open(json_log_file, 'w') as f:
                    json.dump(json_log_info, f)

    # Draw figures
    # train-val log, loss
    plt.title('Loss')
    plt.xlabel('epoch')
    for config_name in trainval_log_info.keys():
        trainval_log = trainval_log_info[config_name]
        epoch = trainval_log['epoch']
        loss = trainval_log['loss']
        model_name = config_name.split('_')[0]
        plt.plot(epoch, loss, label=model_name)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(work_dir, 'loss.jpg'))
    plt.show()
    # train-val log, val_accuracy
    for k in range(args.top_k):
        plt.title(f'Top-{k+1} validation accuracy')
        plt.xlabel('epoch')
        for config_name in trainval_log_info.keys():
            trainval_log = trainval_log_info[config_name]
            epoch = trainval_log['epoch']
            model_name = config_name.split('_')[0]
            max_k = len(trainval_log['val_accuracy'][0])
            if k < max_k:
                val_acc = []
                for i in range(len(epoch)):
                    val_acc.append(trainval_log['val_accuracy'][i][k])
                plt.plot(epoch, val_acc, label=model_name)
            else:
                continue
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(work_dir, f'top-{k+1}_validation-accuracy.jpg'))
        plt.show()
    # test, accuracy
    for k in range(args.top_k):
        plt.title(f'Top-{k+1} test accuracy')
        plt.xlabel('epoch')
        for config_name in test_info.keys():
            model_name = config_name.split('_')[0]
            checkpoint_list = []
            test_accuracy = []
            for ckpt in test_info[config_name].keys():
                checkpoint_list.append(int(ckpt.split('.')[0].split('_')[-1]))
                max_k = len(test_info[config_name][ckpt]['accuracy'])
                if k < max_k:
                    test_accuracy.append(test_info[config_name][ckpt]['accuracy'][k])
                else:
                    break
            if len(test_accuracy) > 0:
                plt.plot(checkpoint_list, test_accuracy, label=model_name)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(work_dir, f'top-{k+1}_test-accuracy.jpg'))
        plt.show()


def read_trainval_log_file(path):
    """
    Read train & validation log file(json)
    Args:
        path (str)
    :return:
    >>> result = {
    >>>     'epoch': [],
    >>>     'loss': [],
    >>>     'val_accuracy': [],  # [[top1, top2, ...], ...]
    >>>}
    """
    assert os.path.splitext(path)[1] == '.json', 'Only \'json\' file can be read.'
    assert os.path.exists(path), FileExistsError(path)
    with open(path, 'r') as f:
        log = json.load(f)
    result = {
        'epoch': [],
        'loss': [],
        'val_accuracy': []
    }
    current_epoch = 0
    loss = 0
    counter = 0
    for i in log:
        if i['epoch'] not in result['epoch']:
            result['epoch'].append(i['epoch'])
            current_epoch = i['epoch']
            loss = 0
            counter = 0
        if i['mode'] == 'train' and current_epoch == i['epoch']:
            loss += i['loss']
            counter += 1
        if i['mode'] == 'val':
            result['loss'].append(round(loss/counter, 4))
            result['val_accuracy'].append(i['accuracy'])
    return result


def trainval_log_statistic(trainval_log):
    """
    Args:
        trainval_log (dict): output of function: read_trainval_log_file
    :return:
    e.g.:
    >>> result = {
    >>>     'total_epoch': 100,
    >>>     'last_top-1_val_accuracy': 0.87,
    >>>     'last_top-2_val_accuracy': 0.95,
            ...
    >>>     'max_top-1_val_accuracy': 0.88,
    >>>     'last_top-1_epoch': 65,
    >>>     'max_top-2_val_accuracy': 0.96,
    >>>     'last_top-2_epoch': 45,
            ...
    >>> }
    """
    result = {'total_epoch': len(trainval_log['epoch'])}
    # last top-k validation accuracy
    for k in range(len(trainval_log['val_accuracy'][0])):
        result[f'last_top-{k + 1}_val_accuracy'] = trainval_log['val_accuracy'][-1][k]
    # max top-k validation accuracy
    top_k_acc = collections.OrderedDict()
    for k in range(len(trainval_log['val_accuracy'][0])):
        top_k_acc[str(k + 1)] = []
        for epoch in trainval_log['epoch']:
            top_k_acc[str(k + 1)].append(trainval_log['val_accuracy'][epoch-1][k])
    for key in top_k_acc.keys():
        max_acc = max(top_k_acc[key])
        max_acc_epoch = trainval_log['epoch'][top_k_acc[key].index(max_acc)]
        result[f'max_top-{key}_val_accuracy'] = max_acc
        result[f'max_top-{key}_epoch'] = max_acc_epoch
    return result


if __name__ == '__main__':
    main()
