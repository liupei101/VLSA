"""Earth Mover Distance (Wasserstein distance p=1)
Notes:
- Adapted from https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss.
- See more details in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def wasserstein_loss(pred_dist, target_dist):
    return cdf_loss(pred_dist, target_dist, p=1)

def cdf_loss(pred_dist, target_dist, p=1, normalize_dist=True, ret_raw=False):
    """
    See proof in Ramdas, Garcia, Cuturi. 
      "On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests". arXiv:1509.02237, 2015.
    """
    assert pred_dist.shape == target_dist.shape, "Two input shapes do not match."
    if normalize_dist:
        pred_dist = pred_dist / (torch.sum(pred_dist, dim=-1, keepdim=True) + 1e-14)
        target_dist = target_dist / (torch.sum(target_dist, dim=-1, keepdim=True) + 1e-14)

    # make cdf with cumsum
    cdf_pred_dist = torch.cumsum(pred_dist, dim=-1)
    cdf_target_dist = torch.cumsum(target_dist, dim=-1)

    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_pred_dist - cdf_target_dist)), dim=-1)
    elif p == 2:
        if not ret_raw:
            cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_pred_dist - cdf_target_dist), 2), dim=-1))
        else:
            cdf_distance = torch.sum(torch.pow((cdf_pred_dist - cdf_target_dist), 2), dim=-1)
    else:
        if not ret_raw:
            cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_pred_dist - cdf_target_dist), p), dim=-1), 1 / p)
        else:
            cdf_distance = torch.sum(torch.pow(torch.abs(cdf_pred_dist - cdf_target_dist), p), dim=-1)

    return cdf_distance

def convert_survival_label(t, e, n_bins):
    t, e = t.view(-1, 1), e.view(-1, 1)
    bsz = t.shape[0]
    t_vector = torch.full(
        (bsz, n_bins), 0,
        device=t.device, dtype=t.dtype
    ).scatter_(1, t, 1)
    
    for i in range(bsz):
        loc = t[i, 0] + 1
        if loc < n_bins:
            t_vector[i, loc:] = t_vector[i, loc:] + (1 - e[i, 0])

    return t_vector


class SurvEMD(nn.Module):
    """
    Earth Mover Distance^2 (Wasserstein distance p=2) for ordinal survival analysis.
    """
    def __init__(self, p=2, raw_distance=True, reduction='mean', **kws):
        super().__init__()
        self.p = p
        self.raw_distance = raw_distance
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none']
        print(f"[SurvEMD] initialized a SurvEMD loss with p = {p}, raw_distance = {raw_distance}, and reduction = {reduction}.")

    def forward(self, y_hat, t, e, cur_logit_scale=10.0):
        """
        y_hat: torch.FloatTensor() with shape of [B, MAX_T], converted by softmax.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            e = 1 for uncensored samples (with event), 
            e = 0 for censored samples (without event).
        cur_logit_scale: should be the value after applying self.logit_scale.exp().
        """
        bsz, n_bins = y_hat.shape[0], y_hat.shape[-1]
        
        if isinstance(cur_logit_scale, torch.Tensor):
            _logit_scale = cur_logit_scale.detach()
        else:
            _logit_scale = cur_logit_scale

        t = t.view(-1, 1).long()
        e = e.view(-1, 1).long()
        # convert time-to-event label
        target = convert_survival_label(t, e, n_bins) # [bsz, n_bins]
        target_dist = torch.softmax((2 * target - 1) * _logit_scale, dim=-1)

        # convert the predicted y_hat
        pred = (1 - e) * ((1 - target) * y_hat + target * _logit_scale) + e * y_hat
        pred_dist = torch.softmax(pred, dim=-1) # [bsz, n_bins]

        # [bsz, n_bins] <==> [bsz, n_bins]
        loss = cdf_loss(
            pred_dist, target_dist, 
            p=self.p, 
            normalize_dist=False, 
            ret_raw=self.raw_distance
        ) # [bsz, ]

        if self.reduction == 'mean': # default
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def SupConLoss(logits, targets):
    """
    Contrastive loss
    """
    assert logits.shape == targets.shape, "logits and targets do not match in shape."
    # for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach() 
    exp_logits = torch.exp(logits) 
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (targets * log_prob).sum(1) / targets.sum(1)
    loss = - mean_log_prob_pos.mean()
    return loss


class SurvT2I(nn.Module):
    """
    Text2Image loss used for survival prediction models.
    """
    def __init__(self, loss='CL', reduction='mean', **kws):
        super().__init__()
        print("[SurvT2I] got additional kws:", kws)
        self.reduction = reduction
        self.loss = loss

        if loss == 'CL':
            self.loss_func = SupConLoss            
        elif loss == 'KL':
            self.loss_func = nn.KLDivLoss(reduction="sum")
        else:
            raise NotImplementedError(f"Expected loss = CL or KL, but got {loss}.")
        
        print(f"[SurvT2I] initialized a Text2Image loss with loss = {loss}.")

    def forward(self, raw_y_hat, t, e, cur_logit_scale=10.0):
        """
        raw_y_hat: torch.FloatTensor() with shape of [B, MAX_T], unnormalized raw prediction.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            e = 1 for uncensored samples (with event), 
            e = 0 for censored samples (without event).
        cur_logit_scale: should be the value after applying self.logit_scale.exp().
        """
        # transposed raw prediction: logits output by the model
        logits = raw_y_hat.T # [MAX_T, B]
        
        n_bins, bsz = logits.shape[0], logits.shape[-1]
        
        if isinstance(cur_logit_scale, torch.Tensor):
            _logit_scale = cur_logit_scale.detach()
        else:
            _logit_scale = cur_logit_scale

        # convert time-to-event label
        t, e = t.view(-1, 1).long(), e.view(-1, 1).long()
        targets = convert_survival_label(t, e, n_bins).T # [n_bins, bsz]
        # indicator to select from target & y_hat
        sel_ind = torch.logical_not(torch.logical_and(targets == 1, e.T == 0))

        # cal loss for each text
        total_loss, num_slot = 0.0, 0
        for i in range(n_bins):
            if not sel_ind[i].any():
                continue
            
            c_target = torch.masked_select(targets[i], sel_ind[i]).unsqueeze(0).to(logits)
            if c_target.sum() == 0:
                continue

            c_logit = torch.masked_select(logits[i], sel_ind[i]).unsqueeze(0)

            if self.loss == 'CL':
                cur_loss = self.loss_func(c_logit, c_target)

            elif self.loss == 'KL':
                c_target = ((2 * c_target - 1) * _logit_scale).softmax(dim=-1)
                cur_loss = self.loss_func(F.log_softmax(c_logit, dim=-1), c_target)

            num_slot += 1
            total_loss += cur_loss

        if self.reduction == 'mean' and num_slot != 0:
            total_loss /= num_slot

        return total_loss
