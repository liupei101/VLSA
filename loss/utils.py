import inspect
from functools import partial
import torch
import torch.nn as nn

from . import loss_surv as SurvLoss
from . import loss_clf as ClfLoss
from .loss_surv_ext import SurvEMD, SurvT2I
from .loss_clf import BinaryCrossEntropy, SoftTargetCrossEntropy


def load_loss(task, *args, **kws):
    if task in ['clf', 'sa', 'vlsa']:
        assert 'loss_type' in kws, "The key of `loss_type` is not found in kws."
        loss_fn = dict()
        func_load_loss = load_clf_loss_func if task == 'clf' else load_surv_loss_func
        for loss_name in kws['loss_type']:
            loss_fn[loss_name] = func_load_loss(loss_name, **kws[loss_name])
        return loss_fn
    else:
        raise NotImplementedError(f"cannot recognize the task {task}.")

def loss_reg_l1(coef):
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func

def load_clf_loss_func(loss_type:str, **loss_cfg):
    """
    loss_type (str): The name of the classification loss functions or classes defined in loss_clf.py
    loss_cfg (dict): The arguments to be specified for the loss function. 
    """
    if loss_type == 'BCE':
        target_loss_func = BinaryCrossEntropy(loss_cfg['smoothing'], target_threshold=loss_cfg['target_thresh'])
    elif loss_type == 'CE':
        target_loss_func = SoftTargetCrossEntropy(loss_cfg['smoothing'])
    else:
        loss_protype = getattr(ClfLoss, loss_type)
        if inspect.isclass(loss_protype):
            target_loss_func = loss_protype(**loss_cfg)
        elif inspect.isfunction(loss_protype):
            target_loss_func = partial(loss_protype, **loss_cfg)
        else:
            raise ValueError(f"{loss_type} is not found.")

    return target_loss_func

def load_surv_loss_func(loss_type:str, **loss_cfg):
    """
    loss_type (str): The name of the survival loss functions or classes defined in loss_surv.py
    loss_cfg (dict): The arguments of the survival function to be loaded. 
    """
    if loss_type == 'CE':
        target_loss_func = nn.CrossEntropyLoss()
    elif loss_type == 'KL':
        target_loss_func = nn.KLDivLoss(reduction="sum")
    elif loss_type == 'SurvEMD':
        target_loss_func = SurvEMD(**loss_cfg)
    elif loss_type == 'SurvT2I':
        target_loss_func = SurvT2I(**loss_cfg)
    elif loss_type == 'QueryDiv':
        target_loss_func = None
    else:
        loss_protype = getattr(SurvLoss, loss_type)
        if inspect.isclass(loss_protype):
            target_loss_func = loss_protype(**loss_cfg)
        elif inspect.isfunction(loss_protype):
            target_loss_func = partial(loss_protype, **loss_cfg)
        else:
            raise ValueError(f"{loss_type} is not found.")

    return target_loss_func
