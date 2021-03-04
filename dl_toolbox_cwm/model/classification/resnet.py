"""
ResNet for classification
Programmer: Weiming Chen
Date: 2021.3
"""
import torch
import torch.nn as nn
from dl_toolbox_cwm.model.backbone import ResNet
from dl_toolbox_cwm.model.neck import GlobalAveragePooling
from dl_toolbox_cwm.utils import validate_ckpt

__all__ = [
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'
]


class ResNet_CLS(ResNet):
    fc_channel = {
        18: 512,
        34: 512,
        50: 2048,
        101: 2048,
        152: 2048
    }

    def __init__(self, num_classes, **kwargs):
        super(ResNet_CLS, self).__init__(**kwargs)
        self.gap = GlobalAveragePooling()
        self.fc = nn.Linear(self.fc_channel[self.depth], num_classes)

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


def ResNet18(num_classes, pretrained=None):
    """
    ResNet18 for classification
    Args:
        num_classes (int): the number of classes
        criterion: loss function
        pretrained (str | None):
    :return: model
    """
    model = ResNet_CLS(num_classes, depth=18)
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


def ResNet34(num_classes, pretrained=None):
    """
    ResNet34 for classification
    Args:
        num_classes (int): the number of classes
        criterion: loss function
        pretrained (str | None):
    :return: model
    """
    model = ResNet_CLS(num_classes, depth=34)
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


def ResNet50(num_classes, pretrained=None):
    """
    ResNet50 for classification
    Args:
        num_classes (int): the number of classes
        criterion: loss function
        pretrained (str | None):
    :return: model
    """
    model = ResNet_CLS(num_classes, depth=50)
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


def ResNet101(num_classes, pretrained=None):
    """
    ResNet101 for classification
    Args:
        num_classes (int): the number of classes
        criterion: loss function
        pretrained (str | None):
    :return: model
    """
    model = ResNet_CLS(num_classes, depth=101)
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


def ResNet152(num_classes, pretrained=None):
    """
    ResNet152 for classification
    Args:
        num_classes (int): the number of classes
        criterion: loss function
        pretrained (str | None):
    :return: model
    """
    model = ResNet_CLS(num_classes, depth=152)
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
