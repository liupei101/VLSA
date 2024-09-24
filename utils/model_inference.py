import os.path as osp
import math
import numpy as np
import torch
import torch.nn.functional as F

from runner.vlsa_handler import VLSAHandler
from utils.func import get_model_cfg


def load_vlsa_model(run_path, cuda_id=0, return_cfg=False):
    model_cfg = get_model_cfg(run_path)
    model_cfg['cuda_id'] = cuda_id
    model = VLSAHandler.func_load_model(model_cfg)
    model = model.cuda(cuda_id)
    net_ckpt = torch.load(osp.join(run_path, 'train_model-last.pth'), map_location=lambda storage, loc: storage.cuda(cuda_id))
    model.load_state_dict(net_ckpt['model'], strict=False)

    if return_cfg:
        return model, model_cfg
    return model

def evaluate_prototype_shap_imp(decoupled_similarity, logit_scale, verbose=False):
    # decoupled_similarity: [P, num_cls], text-to-image similarity decoupled over text prototypes
    if isinstance(decoupled_similarity, np.ndarray):
        decoupled_similarity = torch.from_numpy(decoupled_similarity)

    num_p, num_cls = decoupled_similarity.shape

    def calc_survival_risk(pred_sim):
        # pred_sim: [P', num_cls] \in [-1, 1]
        prob_IF = F.softmax(logit_scale * pred_sim.mean(dim=0), dim=0) # [num_ranks, ]
        survival_risk = torch.sum((num_cls - torch.arange(0, num_cls)) * prob_IF)
        return survival_risk.item()

    def int2bin(intx):
        idx = []
        for i in range(num_p):
            if intx % 2 == 1:
                idx.append(i)
            intx = intx // 2
        assert intx == 0
        return idx

    n_cases = 2 ** num_p
    V = torch.zeros(n_cases)
    for i in range(n_cases):
        if i == 0:
            # V[i] = (num_p + 1) / 2
            V[i] = 1.0
            continue

        sel_idx = int2bin(i)
        V[i] = calc_survival_risk(decoupled_similarity[sel_idx])

    if verbose:
        print("[SHAP] Survival risk (base) =", V[0])
        print("[SHAP] Survival risk (full) =", V[n_cases - 1])

    Fac = [math.factorial(i) for i in range(1 + num_p)]
    W = [Fac[i] * Fac[num_p - i - 1] / Fac[num_p] for i in range(num_p)]
    # calculate contribution for each prototype
    shap_imp = torch.zeros(num_p)
    for i in range(num_p):
        # for the state not containing i
        sum_i = 0.0
        for j in range(n_cases):
            sel_idx = int2bin(j)
            if i in sel_idx:
                continue
            
            side_effect = V[j + 2 ** i] - V[j]
            sum_i += W[len(sel_idx)] * side_effect
        shap_imp[i] = sum_i
    
    if verbose:
        print("[SHAP] Sum over SHAP values =", shap_imp.sum())
    
    return shap_imp

def calc_text_img_similarity(model, X_feats, axis_softmax='V', verbose=False):
    # X_feats, all patch features of one patient
    assert axis_softmax in ['L', 'V']

    model.eval()
    
    if X_feats.shape[0] == 1 and len(X_feats.shape) == 3:
        X_feats = X_feats[0]
    X = X_feats.to(next(model.parameters()).device)

    with torch.no_grad():
        logit_scale = model.get_logit_scale()
        text_features = model.forward_text_only()
        norm_text_features = F.normalize(text_features, dim=-1) # [num_ranks, emb_dim]
    
    text_features = text_features.detach()
    norm_text_features = norm_text_features.detach()
    logit_scale = logit_scale.detach().item()
    coattn_logit_scale = model.mil_encoder.get_coattn_logit_scale().item() # with exp

    if verbose:
        print("pred_logit_scale:", logit_scale)
        print("coattn_logit_scale:", coattn_logit_scale)

    _dim = 0 if axis_softmax == 'L' else 1
    
    with torch.no_grad():
        # prototype's transformed text features
        transformed_text_features = model.mil_encoder.get_query()
        norm_transformed_text_features = F.normalize(transformed_text_features, dim=-1)
        
        norm_X = F.normalize(X, dim=-1) # [N, d]
        
        # calc similarity
        _A = torch.matmul(norm_transformed_text_features, norm_X.transpose(0, 1)) # [P, N]
        _A = coattn_logit_scale * _A
        A = F.softmax(_A, dim=_dim)  # [P, N]

        # calc the contribution of each prototype
        # Approach 1: in a direct way like model's forward process 
        X = X.unsqueeze(0) # [1, N, d]
        image_feature, cottn_score = model.mil_encoder(X, ret_with_attn=True)
        L_image_feature = image_feature.norm(dim=-1) # [1, emb_dim]
        norm_image_feature = image_feature / L_image_feature # [1, emb_dim]
        img_txt_similarities = norm_image_feature @ norm_text_features.t() # [1, num_ranks]
        probs = F.softmax(logit_scale * img_txt_similarities, dim=-1) # [1, num_ranks]
        
        # Approach 2: in a decoupled way
        enc_X = model.mil_encoder.visual_adapter(X) # [1, N, d]
        norm_enc_X = enc_X.squeeze(0) / L_image_feature # [N, d]
        cottn_score = cottn_score.squeeze(0) # [P, N]
        decoupled_img_txt_similarities = cottn_score @ (norm_enc_X @ norm_text_features.t()) # [P, N] @ [N, num_ranks] -> [P, num_ranks]
        decoupled_imp = F.softmax(logit_scale * decoupled_img_txt_similarities, dim=0) # [P, num_ranks]
        probs_2 = F.softmax(logit_scale * decoupled_img_txt_similarities.mean(dim=0, keepdims=True), dim=-1) # [1, num_ranks]

    A = A.detach().cpu()
    cottn_score = cottn_score.detach().cpu()
    probs = probs.detach().cpu()
    probs_2 = probs_2.detach().cpu()
    decoupled_img_txt_similarities = decoupled_img_txt_similarities.detach().cpu()
    decoupled_shap_imp = evaluate_prototype_shap_imp(decoupled_img_txt_similarities, logit_scale, verbose=verbose)
    decoupled_imp = decoupled_imp.detach().cpu()
    # data = {"prob": probs, "decoupled_prob": imp_matrix}
    return None, A, cottn_score, probs, probs_2, decoupled_imp, decoupled_shap_imp

def calc_abmil_text_img_similarity(model, X_feats, verbose=False, **kws):
    # X_feats, all patch features of one patient
    model.eval()
    
    if X_feats.shape[0] == 1 and len(X_feats.shape) == 3:
        X_feats = X_feats[0]
    X = X_feats.to(next(model.parameters()).device)

    with torch.no_grad():
        logit_scale = model.get_logit_scale()
        text_features = model.forward_text_only()
        norm_text_features = F.normalize(text_features, dim=-1) # [num_ranks, emb_dim]
    
    text_features = text_features.detach()
    norm_text_features = norm_text_features.detach()
    logit_scale = logit_scale.detach().item()

    if verbose:
        print("pred_logit_scale:", logit_scale)
    
    with torch.no_grad():
        # in a direct way like model's forward process 
        X = X.unsqueeze(0) # [1, N, d]
        image_feature, attn_score = model.mil_encoder(X, ret_with_attn=True)
        attn_score = F.softmax(attn_score, dim=-1)
        L_image_feature = image_feature.norm(dim=-1) # [1, emb_dim]
        norm_image_feature = image_feature / L_image_feature # [1, emb_dim]
        img_txt_similarities = norm_image_feature @ norm_text_features.t() # [1, num_ranks]
        probs = F.softmax(logit_scale * img_txt_similarities, dim=-1) # [1, num_ranks]
        
    attn_score = attn_score.detach().cpu()
    probs = probs.detach().cpu()
    return attn_score, probs
