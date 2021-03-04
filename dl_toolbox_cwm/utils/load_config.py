"""
Load config file
Programmer: Weiming Chen
Date: 2020.12
"""
import sys
import os
import ast
from importlib import import_module


class LoadConfig(object):
    """
    A facility for loading config file.
    The supported file format is: python
    """

    support_format = ['.py']
    all_variables = [
        'epoch', 'batch_size',
        'dataset_name', 'train_pipeline', 'test_pipeline',
        'train_config', 'val_config', 'test_config',
        'train_set', 'val_set', 'test_set',
        'model', 'criterion', 'optimizer', 'scheduler',
        'val_interval', 'checkpoint_interval', 'log_interval'
    ]
    necessary_variables = [
        'epoch', 'batch_size',
        'dataset_name', 'train_pipeline', 'test_pipeline',
        'train_config', 'val_config', 'test_config',
        'train_set', 'val_set', 'test_set',
        'model', 'criterion', 'optimizer'
    ]

    def __init__(self, cfg_path):
        assert os.path.exists(cfg_path), 'Config file \'{}\' is not exist'.format(cfg_path)
        fmt = os.path.splitext(cfg_path)[1]
        assert fmt in self.support_format, 'Unsupported file format {}, only .py/.json are supported.'.format(fmt)
        self.cfg_path = cfg_path
        self.fmt = fmt

    def validate_py_syntax(self):
        # validate the syntax of python file
        with open(self.cfg_path, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config 'f'file {self.cfg_path}: {e}')

    def validate_config_dict(self, cfg_dict):
        for i in self.necessary_variables:
            if i not in cfg_dict.keys():
                raise Exception('Config file should include variable: {}'.format(i))
            if i.endswith('_config'):
                if 'shuffle' not in cfg_dict[i].keys():
                    raise Exception('Variable \'{}\' should include key: shuffle'.format(i))

    def clean_config_dict(self, cfg_dict):
        clean_dict = dict()
        for k, v in cfg_dict.items():
            if k in self.all_variables:
                clean_dict[k] = v
        return clean_dict

    def file_to_dict(self, use_predefined_variables=True):
        # Convert config file to config dict
        self.validate_py_syntax()
        cfg_dir = self.cfg_path.split('/')
        cfg_name = cfg_dir[-1]
        cfg_module_name = cfg_name.split('.')[0]
        del cfg_dir[-1]
        cfg_dir = '/'.join(cfg_dir)
        sys.path.insert(0, cfg_dir)
        mod = import_module(cfg_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        del sys.modules[cfg_module_name]
        cfg_text = self.cfg_path + '\n'
        with open(self.cfg_path, 'r') as f:
            cfg_text += f.read()
        return cfg_dict, cfg_text

    def load_cfg_file(self, use_predefined_variables=True):
        cfg_dict, cfg_text = self.file_to_dict(use_predefined_variables)
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        self.validate_config_dict(cfg_dict)
        return self.clean_config_dict(cfg_dict)

    @staticmethod
    def pretty_text(cfg_dict):
        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        text = _format_dict(cfg_dict, outest_level=True)
        return text
