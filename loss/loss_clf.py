""" Copied from pytorch image models (timm).
Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropy(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self, smoothing=0.1, target_threshold: Optional[float] = None, weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, ret_mean = True) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        cur_weight = weight if weight is not None else self.weight
        reduction_way = self.reduction if ret_mean else 'none'
        return F.binary_cross_entropy_with_logits(
            x, target,
            cur_weight,
            pos_weight=self.pos_weight,
            reduction=reduction_way)


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, ret_mean = True) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if weight is not None:
            loss = loss * weight
        if ret_mean:
            return loss.mean()
        else:
            return loss


class SoftTargetCrossEntropy(nn.Module):
    """CE loss with one-hot soft labels.
    """
    def __init__(self, smoothing=0.1):
        super(SoftTargetCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, ret_mean = True) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        # if it's class indices, then convert it to one-hot with label_smoothing
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        if weight is not None:
            loss = torch.sum(-target * weight * F.log_softmax(x, dim=-1), dim=-1)
        else:
            loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        if ret_mean:
            return loss.mean()
        else:
            return loss
