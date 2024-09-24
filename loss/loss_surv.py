import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################
# General Loss for Survival Analysis Models, 
# including continuous output and discrete output.
##################################################

def recon_loss(pred_t, t, e, alpha=0.0, gamma=1.0, norm='l1', cur_alpha=None, **kws):
    """Continuous Survival Model

    Reconstruction loss for pred_t and labels.
    recon_loss = l2 + l3
    if e = 0, l2 = max(0, t - pred_t)
    if e = 1, l3 = |t - pred_t|
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    loss_obs = e * torch.abs(pred_t - t)
    loss_cen = (1 - e) * F.relu(gamma - (pred_t - t))
    if norm == 'l2':
        loss_obs = loss_obs * loss_obs
        loss_cen = loss_cen * loss_cen
    loss_recon = loss_obs + loss_cen
    _alpha = alpha if cur_alpha is None else cur_alpha
    loss = (1.0 - _alpha) * loss_recon + _alpha * loss_obs
    loss = loss.mean()
    return loss

def rank_loss(pred_t, t, e, gamma=1, norm='l1', add_weight=False, **kws):
    """Continuous Survival Model

    Ranking loss for preditions and observations.
    for pairs (i, j) conditioned on e_i = 1 & t_i < t_j:
        diff_ij = (-pred_t_i) - (-pred_t_j)
        rank_loss = ||max(0, gamma - diff_ij)||_norm
                  = ||max(0, gamma + pred_t_i - pred_t_j)||_norm
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    pair_mask = (t.view(-1, 1) < t.view(1, -1)) * (e.view(-1, 1) == 1)
    if not torch.any(pair_mask):
        return torch.Tensor([0.0]).to(pred_t.device)
    pair_diff = pred_t.view(-1, 1) - pred_t.view(1, -1) # the lower, the best
    pair_loss = F.relu(gamma + pair_diff)
    pair_mask = pair_mask.float()
    if add_weight:
        # masked_log_softmax
        x = pair_diff
        maxx = (x * pair_mask + (1 - 1 / (pair_mask + 1e-5))).max()
        log_ex = x - maxx
        log_softmax = log_ex - (torch.exp(log_ex * pair_mask) * pair_mask).sum().log()
        normed_weight = (log_softmax * pair_mask).exp() * pair_mask
    else:
        weight = pair_mask
        normed_weight = weight / weight.sum()

    if norm == 'l2':
        pair_loss = pair_loss * pair_loss
    elif norm == 'l1':
        pass
    else:
        raise NotImplementedError('Arg. `norm` expected l1/l2, but got {}'.format(norm))

    rank_loss = (pair_loss * normed_weight).sum()
    return rank_loss

def MSE_loss(pred_t, t, e, include_censored=False, **kws):
    """Continuous Survival Model.

    MSE loss for pred_t and labels, used for reproducing ESAT (shen et al., ESAT, AAAI, 2022).
    Please refer to its official repo: https://github.com/notbadforme/ESAT/blob/main/esat/trainforesat.py#L111
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    loss = e * (pred_t - t) * (pred_t - t)
    if include_censored:
        # loss += (1 - e) * F.relu(t - pred_t) * F.relu(t - pred_t)
        loss += (1 - e) * (pred_t - t) * (pred_t - t)
    loss = loss.mean()
    return loss


class SurvMLE(nn.Module):
    """A maximum likelihood estimation function in Survival Analysis.
    As suggested in '10.1109/TPAMI.2020.2979450',
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances 
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...
    The larger the alpha, the bigger the importance of event_loss.
    If alpha = 0, event loss and censored loss are viewed equally. 
    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py
    """
    def __init__(self, alpha=0.0, eps=1e-7, **kws):
        super(SurvMLE, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, hazards_hat, t, e, cur_alpha=None):
        """
        y: torch.FloatTensor() with shape of [B, 2] for a discrete model, converted by Sigmoid.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            e = 1 for uncensored samples (with event), 
            e = 0 for censored samples (without event).
        hazards_hat: torch.FloatTensor() with shape of [B, MAX_T]
        """
        batch_size = len(t)
        t = t.view(batch_size, 1).long() # ground truth bin, 0 [0,a_1), 1 [a_1,a_2),...,k-1 [a_k-1,inf)
        c = 1 - e.view(batch_size, 1).float() # convert it to censorship status, 0 or 1
        S = torch.cumprod(1 - hazards_hat, dim=1) # survival is cumulative product of 1 - hazards
        S_padded = torch.cat([torch.ones_like(c), S], 1) # s[0] = 1.0 to avoid for t = 0
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, t).clamp(min=self.eps)) + torch.log(torch.gather(hazards_hat, 1, t).clamp(min=self.eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, t+1).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss
        

class SurvIFMLE(nn.Module):
    """A maximum likelihood estimation function in Survival Analysis, with Incidence Function as its input.
    As adopted in DeepHit (Lee et al. AAAI, 2018)
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances 
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...
    The larger the alpha, the bigger the importance of event_loss.
    If alpha = 0, event loss and censored loss are viewed equally. 
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean', **kws):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        assert reduction in ['sum', 'mean', 'none']
        self.reduction = reduction
        print("[SurvIFMLE] initialized a SurvIFMLE loss with reduction = {}.".format(reduction))

    def forward(self, incidence_hat, t, e, cur_alpha=None):
        """
        y: torch.FloatTensor() with shape of [B, 2] for a discrete model, converted by softmax.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            e = 1 for uncensored samples (with event), 
            e = 0 for censored samples (without event).
        incidence_hat: torch.FloatTensor() with shape of [B, MAX_T], having been processed by `softmax`.
        """
        batch_size = len(t)
        t = t.view(batch_size, 1).long() # ground truth bin, 0 [0,a_1), 1 [a_1,a_2),...,k-1 [a_k-1,inf)
        c = 1 - e.view(batch_size, 1).float() # convert it to censorship status, 0 or 1
        CIF_hat = torch.cumsum(incidence_hat, dim=1) # compute cumulative incidence function (CIF)
        uncensored_loss = -(1 - c) * torch.log(torch.gather(incidence_hat, 1, t).clamp(min=self.eps))
        censored_loss = - c * torch.log((1 - torch.gather(CIF_hat, 1, t)).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


class SurvPLE(nn.Module):
    """A partial likelihood estimation (called Breslow estimation) function in Survival Analysis.

    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    Note that it only suppurts survival data with no ties (i.e., event occurrence at same time).
    
    Args:
        y_hat (Tensor): Predictions given by the survival prediction model.
        T (Tensor): The last observed time. 
        E (Tensor): An indicator of event observation.
            if E = 1, uncensored one (with event)
            else E = 0, censored one (without event)
    """
    def __init__(self, **kws):
        super(SurvPLE, self).__init__()
        self.CONSTANT = torch.tensor(10.0)

    def forward(self, y_hat, T, E):
        device = y_hat.device
        # avoid numerical overflow
        cont = self.CONSTANT.to(device)
        y_hat = torch.where(y_hat > cont, cont, y_hat)

        n_batch = len(T)
        R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
        for i in range(n_batch):
            for j in range(n_batch):
                R_matrix_train[i, j] = T[j] >= T[i]

        train_R = R_matrix_train.float().to(device)
        train_ystatus = E.float().to(device)

        theta = y_hat.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

        return loss_nn
