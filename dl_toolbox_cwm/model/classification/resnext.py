"""
ResNeXt for classification
https://arxiv.org/abs/1611.05431
Programmer: Weiming Chen
Date: 2021.3
"""
import torch
import torch.nn as nn
from dl_toolbox_cwm.model.backbone import ResNeXt
from dl_toolbox_cwm.model.neck import GlobalAveragePooling
from dl_toolbox_cwm.utils import validate_ckpt

__all__ = [
    'ResNeXt50_32x4d', 'ResNeXt101_32x4d', 'ResNeXt101_32x8d', 'ResNeXt152_32x4d'
]


class ResNeXt_CLS(ResNeXt):
    def __init__(self, num_classes, **kwargs):
        super(ResNeXt_CLS, self).__init__(**kwargs)
        self.gap = GlobalAveragePooling()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            outs = outs[0]
        else:
            outs = tuple(outs)
        outs = self.gap(outs)
        outs = self.fc(outs)
        return outs


def ResNeXt50_32x4d(num_classes, pretrained=None):
    model = ResNeXt_CLS(num_classes, depth=50, groups=32, width_per_group=4)
    if pretrained is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained)
        matched_dict, unmatched = validate_ckpt(model_dict, pretrained_dict)
        if len(unmatched) > 0:
            print('unmatched key ({}): '.format(len(unmatched)))
            for i in range(len(unmatched)):
                unmatched_key = unmatched[i]
                if unmatched_key in model_dict.keys():
                    print(f'model_dict[{unmatched_key}].shape={model_dict[unmatched_key].shape} vs '
                          f'pretrained[{unmatched_key}].shape={pretrained_dict[unmatched_key].shape}')
                else:
                    print(f'key \'{unmatched_key}\' is not in model_dict')
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
    return model


def ResNeXt101_32x4d(num_classes, pretrained=None):
    model = ResNeXt_CLS(num_classes, depth=101, groups=32, width_per_group=4)
    if pretrained is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained)
        matched_dict, unmatched = validate_ckpt(model_dict, pretrained_dict)
        if len(unmatched) > 0:
            print('unmatched key ({}): '.format(len(unmatched)))
            for i in range(len(unmatched)):
                unmatched_key = unmatched[i]
                if unmatched_key in model_dict.keys():
                    print(f'model_dict[{unmatched_key}].shape={model_dict[unmatched_key].shape} vs '
                          f'pretrained[{unmatched_key}].shape={pretrained_dict[unmatched_key].shape}')
                else:
                    print(f'key \'{unmatched_key}\' is not in model_dict')
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
    return model


def ResNeXt101_32x8d(num_classes, pretrained=None):
    model = ResNeXt_CLS(num_classes, depth=101, groups=32, width_per_group=8)
    if pretrained is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained)
        matched_dict, unmatched = validate_ckpt(model_dict, pretrained_dict)
        if len(unmatched) > 0:
            print('unmatched key ({}): '.format(len(unmatched)))
            for i in range(len(unmatched)):
                unmatched_key = unmatched[i]
                if unmatched_key in model_dict.keys():
                    print(f'model_dict[{unmatched_key}].shape={model_dict[unmatched_key].shape} vs '
                          f'pretrained[{unmatched_key}].shape={pretrained_dict[unmatched_key].shape}')
                else:
                    print(f'key \'{unmatched_key}\' is not in model_dict')
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
    return model


def ResNeXt152_32x4d(num_classes, pretrained=None):
    model = ResNeXt_CLS(num_classes, depth=152, groups=32, width_per_group=4)
    if pretrained is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained)
        matched_dict, unmatched = validate_ckpt(model_dict, pretrained_dict)
        if len(unmatched) > 0:
            print('unmatched key ({}): '.format(len(unmatched)))
            for i in range(len(unmatched)):
                unmatched_key = unmatched[i]
                if unmatched_key in model_dict.keys():
                    print(f'model_dict[{unmatched_key}].shape={model_dict[unmatched_key].shape} vs '
                          f'pretrained[{unmatched_key}].shape={pretrained_dict[unmatched_key].shape}')
                else:
                    print(f'key \'{unmatched_key}\' is not in model_dict')
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
    return model
