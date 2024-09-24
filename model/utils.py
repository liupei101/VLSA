from typing import List, Optional
import torch
import torch.nn as nn
import math

from .deepmil import DeepMIL, DSMIL, TransMIL, ILRA, DeepAttnMISL, PatchGCN
from .vlsa import VLSA


##########################################
# Functions for loading models 
##########################################
def load_model(arch:str, dims:Optional[List] = None, **kws):
    if arch == 'DeepMIL':
        assert 'network' in kws, "Please specify a network for a DeepMIL arch."
        network = kws['network']
        if network == 'ABMIL':
            return Deep_ABMIL(dims, **kws)
        elif network == 'MaxMIL':
            return Deep_MaxMIL(dims, **kws)
        elif network == 'MeanMIL':
            return Deep_MeanMIL(dims, **kws)
        elif network == 'DSMIL':
            return Deep_DSMIL(dims, **kws)
        elif network == 'TransMIL':
            return Deep_TransMIL(dims, **kws)
        elif network == 'ILRA':
            return Deep_ILRA(dims, **kws)
        elif network == 'DeepAttnMISL':
            return Deep_AttnMISL(dims, **kws)
        elif network == 'PatchGCN':
            return Deep_PatchGCN(dims, **kws)
    elif arch == 'VLSA':
        return Deep_VLSA(**kws) 
    else:
        raise NotImplementedError("Backbone {} cannot be recognized".format(arch))

def Deep_VLSA(**kws):
    assert 'text_encoder_cfg'   in kws
    assert 'image_encoder_cfg'  in kws
    assert 'prompt_learner_cfg' in kws

    model = VLSA(**kws)

    return model

def Deep_PatchGCN(dims, **kws):
    model = PatchGCN(dims[0], dims[1], dims[2], **kws)

    return model

def Deep_AttnMISL(dims, **kws):
    model = DeepAttnMISL(dims[0], dims[1], dims[2], **kws)

    return model

def Deep_ILRA(dims, **kws):
    model = ILRA(dims[0], dims[1], dims[2], **kws)

    return model

def Deep_TransMIL(dims, **kws):
    model = TransMIL(dims[0], dims[1], dims[2], **kws)

    return model

def Deep_DSMIL(dims, **kws):
    model = DSMIL(dims[0], dims[1], dims[2], **kws)

    return model
    
def Deep_ABMIL(dims, **kws):
    if 'pooling' in kws:
        assert kws['pooling'] in ['attention', 'gated_attention']
    model = DeepMIL(dims[0], dims[1], dims[2], **kws)

    return model

def Deep_MaxMIL(dims, **kws):
    if 'pooling' in kws:
        assert kws['pooling'] == 'max'
    model = DeepMIL(dims[0], dims[1], dims[2], pooling='max', **kws)

    return model

def Deep_MeanMIL(dims, **kws):
    if 'pooling' in kws:
        assert kws['pooling'] == 'mean'
    model = DeepMIL(dims[0], dims[1], dims[2], pooling='mean', **kws)

    return model

##########################################
# Model weight initialization functions
##########################################
@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

@torch.no_grad()
def general_init_weight(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

def init_pytorch_defaults(m, version='041'):
    '''
    copied from AMDIM repo: https://github.com/Philip-Bachman/amdim-public/
    note from me: haven't checked systematically if this improves results
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False

