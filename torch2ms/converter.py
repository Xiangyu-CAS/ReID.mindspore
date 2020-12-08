from mindspore.train.serialization import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore import Tensor
from mindspore.common.parameter import Parameter

import torch

import sys
sys.path.append('../')
from lib.modeling.build_model import resnet50, Encoder
from lib.config import _C as cfg

bn_map = {'running_mean': 'moving_mean',
          'running_var': 'moving_variance',
          'weight': 'gamma',
          'bias': 'beta'}


def torchTensor2MSTensor(key, param):
    param = param.detach().numpy()
    param = Parameter(Tensor(param), name=key)
    return param


def convert_param(torch_state_dict, ms_state_dict):
    new_param = {}
    torch_keys = torch_state_dict.keys()
    ms_keys = ms_state_dict.keys()
    for key, value in torch_state_dict.items():
        if 'num_batches_tracked' in key:
            continue

        prefix, suffix = '.'.join(key.split('.')[:-1]), key.split('.')[-1]
        if prefix + '.running_mean' in torch_keys and \
                prefix + '.running_var' in torch_keys:
            suffix = bn_map[suffix]
            key = prefix + '.' + suffix
        if key in ms_keys:
            new_param[key] = torchTensor2MSTensor(key, value)
        else:
            print('skip key: {}'.format(key))
    return new_param


def main():
    out_path = 'encoder.ckpt'
    torch_model_path = '/media/xiangyuzhu/940AC3310AC30F64/project/NAIC-baseline/output/market1501/r50/best.pth'
    torch_state_dict = torch.load(torch_model_path, map_location='cpu')

    config_file = '../configs/naic20.yml'
    cfg.merge_from_file(config_file)
    cfg.MODEL.PRETRAIN_CHOICE = 'scratch'
    ms_model = Encoder(cfg)
    save_checkpoint(ms_model, out_path)
    ms_state_dict = load_checkpoint(out_path)

    # convert
    new_param = convert_param(torch_state_dict, ms_state_dict)
    load_param_into_net(ms_model, new_param)
    print('saving model to {}'.format(out_path))
    save_checkpoint(ms_model, out_path)


if __name__ == '__main__':
    main()
