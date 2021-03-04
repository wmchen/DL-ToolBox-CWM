"""
Validate checkpoint
Programmer: Weiming Chen
Date: 2021.1
"""


def validate_ckpt(model_dict, ckpt_dict):
    matched_dict = {}
    unmatched_key = []
    for key, value in ckpt_dict.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                matched_dict[key] = value
            else:
                unmatched_key.append(key)
        else:
            unmatched_key.append(key)
    return matched_dict, unmatched_key
