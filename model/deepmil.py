import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import *

EPS = 1e-6
__all__ = [
    "logit_pooling", "FeatMIL", "VLFAN", "DeepMIL", "DSMIL", 
    "TransMIL", "ILRA", "DeepAttnMISL", "PatchGCN"
]

# Note:
#   MI-Zero adapted from https://github.com/mahmoodlab/CONCH/blob/main/conch/downstream/zeroshot_path.py#L44C1-L56C1
def logit_pooling(logits, method):
    """
    logits: N x C logit for each patch
    method:
        - logit_topk: the top number of patches to use for pooling
        - logit_max: mean logits
        - logit_mean: max logits (act as logit_top1)
    """
    if method[:9] in ['logit_max', 'logit_top']:
        topk = 1 if method == 'logit_max' else int(method.split('top')[-1])
        # Sums logits across topk patches for each class, to get class prediction for each topk
        maxk = min(topk, logits.size(0)) # Ensures k is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
        values, _ = logits.topk(maxk, 0, True, True) # maxk x C
        pooled_logits = values.mean(dim=0, keepdim=True) # 1 x C logit scores
    elif method == 'logit_mean':
        pooled_logits = logits.mean(dim=0, keepdim=True) # 1 x C logit scores
    else:
        raise NotImplementedError(f"The pooling ({method}) is not implemented.")
    
    preds = pooled_logits.argmax(dim=1) # predicted class indices
    
    return preds, pooled_logits


class FeatMIL(nn.Module):
    """
    A simple network only handles the aggregation of features.
    """
    def __init__(self, pooling='mean', **kwargs):
        super().__init__()
        self.network = nn.Identity(**kwargs)
        self.pooling = pooling
        print(f"[FeatMIL] specified pooling = {pooling}.")
        print("[FeatMIL] Note: only max or mean pooling affects the inference; otherwise, directly return the input.")

    def forward(self, X):
        """
        X: initial bag features, with shape of [b, K, d]
           where b = 1 for batch size, K is the instance size of this bag, and d is feature dimension.
        """
        assert X.shape[0] == 1

        if self.pooling == 'mean':
            X_vec = torch.mean(X, dim=1)
        elif self.pooling == 'max':
            X_vec, _ = torch.max(X, dim=1)
        else:
            # identity network
            X = X.squeeze(0)
            X_vec = self.network(X)

        return X_vec

#####################################################################################
#      Language-guided Visual Feature Aggregation Networks (with feature adapter)
#####################################################################################


class VLFAN(nn.Module):
    def __init__(self, dim_in=1024, dim_hid=256, use_feat_proj=True, drop_rate=0.25, query='Parameter', num_query=10, 
        gated_query=False, query_pooling='mean', pred_head='default', dim_reduction=4, keep_ratio=0.8, **kwargs):
        super().__init__()

        self._pos_gated_query = -1

        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        
        self.num_query = num_query
        self.query_type = query
        self.gated_query = gated_query
        assert self.query_type in ['Parameter', 'Text']
        if self.query_type != "Parameter":
            self.Q = None
            print(f"[LVFAN] got {self.query_type} for query_type; please call `reset_query` later to reset it.")
        else:
            _n_query = num_query + 1 if self.gated_query else num_query
            self.Q = nn.Parameter(torch.randn(_n_query, dim_in))
            print(f"[LVFAN] Q (query) is initialized as nn.Parameter ({_n_query} x {dim_in}).")

        if self.gated_query:
            print(f"[LVFAN] gated query will be used.")

        assert query_pooling in ['mean', 'max', 'weight', 'attention', 'gated_attention']
        if query_pooling == 'attention':
            self.query_pooling = Attention_Pooling(dim_in, dim_hid)
        elif query_pooling == 'gated_attention':
            self.query_pooling = Gated_Attention_Pooling(dim_in, dim_hid, dropout=drop_rate)
        elif query_pooling == 'weight':
            self.query_pooling = nn.Parameter(torch.randn(1, num_query))
        else:
            self.query_pooling = query_pooling

        self.pred_head = pred_head
        if self.pred_head == 'Identity':
            self.visual_adapter = nn.Identity()
            print("[LVFAN] nn.Identity is used in the end projection.")

        else:
            self.visual_adapter = nn.Linear(dim_in, dim_in)
            print("[LVFAN] nn.Linear is used in the end projection.")
        
        self.use_custom_coattn = True
        if self.use_custom_coattn:
            self.coattn_logit_scale = torch.ones([]) * np.log(100)
            print("[VLFAN] warning: use a simple Cross-Attention Layer.")

    def get_coattn_logit_scale(self):
        return self.coattn_logit_scale.exp()

    def reset_query(self, query_network):
        assert self.query_type != "Parameter", f"Cannot override Q (query) for query_type ({self.query_type})."
        self.Q = query_network
        print("[LVFAN] Q (query) is reset to a network.")

    def forward_query_pooling(self, X):
        # X: [B, N, C] -> [B, C]
        if self.query_pooling == 'mean':
            pooled_out = torch.mean(X, dim=1)
            return pooled_out, None

        elif self.query_pooling == 'max':
            pooled_out, _ = torch.max(X, dim=1)
            return pooled_out, None

        else:
            if callable(self.query_pooling):
                pooled_out, attn_score = self.query_pooling(X)
                return pooled_out, attn_score
            else:
                weight = F.softmax(self.query_pooling, dim=-1).unsqueeze(0) # [1, 1, N]
                pooled_out = torch.matmul(weight, X).squeeze(1) # [B, C]
                return pooled_out, None

    def get_query(self):
        assert self.Q is not None, f"You have to call `reset_query` to reset query for query_type ({self.query_type})."
        Q = self.Q() if callable(self.Q) else self.Q # [P, C]
        return Q

    def query_div_loss(self, last_div=True, **kws):
        Q = self.get_query()
        norm_Q = F.normalize(Q, dim=-1)
        if len(Q) == self.num_query + 1 and last_div:
            # in this case, with additional one full prototype prompt
            sim = norm_Q[-1:] @ norm_Q[:-1].T
        else:
            sim = norm_Q @ norm_Q.T
            sim = sim[~torch.eye(len(Q), dtype=torch.bool, device=sim.device)]

        loss = sim.abs().mean()
        return loss

    def forward(self, X, ret_with_attn=False):
        """
        X: initial bag features, with shape of [B, N, C]
           where B = 1 for batch size, N is the instance size of this bag, and C is feature dimension.
        """
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        
        # cross attention for instance aggregation
        # [B, N, C] -> [B, P, C] --mean--> [B, C]
        Q = self.get_query()

        if self.use_custom_coattn:
            # Normalize the query
            Q = F.normalize(Q.unsqueeze(0), dim=-1) # [B, P, C]

            norm_X = F.normalize(X, dim=-1) # [B, N, C]
            A_ = torch.matmul(Q, norm_X.transpose(1, 2)) # [B, P, N] \in [-1, 1]

            if self.gated_query:
                assert self._pos_gated_query == -1, "The gated query is placed at the end by default."
                assert A_.shape[1] == self.num_query + 1, f"Query number is expected to be {self.num_query + 1}."
                A_ = A_[:, :self._pos_gated_query, :] - A_[:, self._pos_gated_query:, :] # [B, P, N]
            
            A_ = self.coattn_logit_scale.exp() * A_ # logits, with shape of [B, P, N]
            A = F.softmax(A_, dim=-1)  # [B, P, N]
            # print("Max coattn scores:", A.max().item())
            out = torch.matmul(A, X)  # [B, P, N] bmm [B, N, C] = [B, P, C]

            # (mean)-pooling over coattn outputs
            pooled_out, pooled_ext = self.forward_query_pooling(out) # [B, P, C] -> [B, C]
            visual_features = self.visual_adapter(pooled_out)

        if ret_with_attn:
            
            if pooled_ext is not None:
                attn = (A.detach(), pooled_ext.detach()) # cross-attn scores and query attn-pooling scores
            else:
                attn = A.detach()

            return visual_features, attn # [B, dim_in], [B, P, N]
        else:
            return visual_features

#####################################################################################
#  Common deep MIL networks: Max-pooling, Mean-pooling, ABMIL, DSMIL, and TransMIL 
#####################################################################################


class DeepMIL(nn.Module):
    """
    Deep Multiple Instance Learning for Bag-level Task.

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """
    def __init__(self, dim_in=1024, dim_hid=256, num_cls=2, use_feat_proj=True, drop_rate=0.25, 
        pooling='attention', pred_head='default', dim_reduction=4, keep_ratio=0.8, **kwargs):
        super(DeepMIL, self).__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']
        assert pred_head in ['default', 'Adapter']

        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        
        if pooling == 'gated_attention':
            self.sigma = Gated_Attention_Pooling(dim_in, dim_hid, dropout=drop_rate)
        elif pooling == 'attention':
            self.sigma = Attention_Pooling(dim_in, dim_hid)
        else:
            self.sigma = pooling
        
        self.pred_head = pred_head
        if self.pred_head == 'Adapter':
            assert keep_ratio >= 0 and keep_ratio <= 1.0
            self.keep_ratio = keep_ratio
            self.visual_adapter = Adapter(dim_in, dim_reduction)
            print("[DeepMIL] Adapter is used in the end projection.")

        else:
            self.g = nn.Linear(dim_in, num_cls)
            print("[DeepMIL] nn.Linear is used in the end projection.")

    def forward(self, X, ret_with_attn=False):
        """
        X: initial bag features, with shape of [b, K, d]
           where b = 1 for batch size, K is the instance size of this bag, and d is feature dimension.
        """
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        
        # global pooling function sigma
        # [B, N, C] -> [B, C]
        if self.sigma == 'mean':
            out_feat = torch.mean(X, dim=1)
        elif self.sigma == 'max':
            out_feat, _ = torch.max(X, dim=1)
        else:
            out_feat, raw_attn = self.sigma(X)
        
        if self.pred_head == 'Adapter':
            adapted_out = self.visual_adapter(out_feat)
            logit = self.keep_ratio * out_feat + (1 - self.keep_ratio) * adapted_out
        
        else:
            logit = self.g(out_feat)

        if ret_with_attn:
            attn = raw_attn.detach()
            return logit, attn # [B, num_cls], [B, N]
        else:
            return logit


################################################
# TransMIL, Shao et al., NeurlPS, 2021.
################################################
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn.detach()
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, dim_in=512, dim_hid=256, num_cls=2, **kwargs):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=dim_hid)
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hid), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hid))
        self.num_cls = num_cls
        self.layer1 = TransLayer(dim=dim_hid)
        self.layer2 = TransLayer(dim=dim_hid)
        self.norm = nn.LayerNorm(dim_hid)
        self._fc2 = nn.Linear(dim_hid, self.num_cls)

    def forward(self, X, **kwargs):

        assert X.shape[0] == 1 # [1, n, 512], single bag
        
        h = self._fc1(X) # [B, n, dim_hid]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, dim_hid]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1) # token: 1 + H + add_length
        n1 = h.shape[1] # n1 = 1 + H + add_length

        #---->Translayer x1
        h = self.layer1(h) #[B, N, dim_hid]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, dim_hid]
        
        #---->Translayer x2
        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            h, attn = self.layer2(h, return_attn=True) # [B, N, dim_hid]
            # attn shape = [1, n_heads, n2, n2], where n2 = padding + n1
            if add_length == 0:
                attn = attn[:, :, -n1, (-n1+1):]
            else:
                attn = attn[:, :, -n1, (-n1+1):(-n1+1+H)]
            attn = attn.mean(1).detach()
            assert attn.shape[1] == H
        else:
            h = self.layer2(h) # [B, N, dim_hid]
            attn = None

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, num_cls]

        if attn is not None:
            return logits, attn

        return logits


###################################################################
#  Official implementation of ILRA (Jinxi Xiang et al. ICLR 2023)  
###################################################################
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):
        Q0 = Q
        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)
        A, _ = self.multihead_attn(Q, K, V)
        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if self.gate is not None:
            O = O.mul(self.gate(Q0))
        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat


class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret = self.mha(global_embedding, X)
        return ret


class ILRA(nn.Module):
    def __init__(self, dim_in=512, dim_hid=256, num_cls=2, 
        num_layers=2, num_heads=8, topk=1, ln=False, **kwargs
    ):
        super().__init__()
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(num_layers):
            block = GAB(
                dim_in=dim_in if idx == 0 else dim_hid,
                dim_out=dim_hid,
                num_heads=num_heads,
                num_inds=topk,
                ln=ln
            )
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=dim_hid, num_heads=num_heads, num_seeds=topk, ln=ln)
        # classifier
        self.classifier = nn.Linear(in_features=dim_hid, out_features=num_cls)

        initialize_weights(self)
        print("[setup] initialized an ILRA model.")

    def forward(self, X):
        for block in self.gab_blocks:
            X = block(X)

        feat = self.pooling(X)
        logits = self.classifier(feat)
        logits = logits.squeeze(1) # [1, num_cls]

        return logits

################################################
#    DeepAttnMISL (Yao et al., MedIA, 2020)
################################################


class DeepAttnMISL(nn.Module):
    """
        Adapted from the official implementation: 
        - DeepAttnMISL/blob/master/DeepAttnMISL_model.py
    """
    def __init__(self, dim_in=512, dim_hid=256, num_cls=1, num_clusters=8, dropout=0.25, **kwargs):
        super().__init__()
        print("[setup] got irrelevant kwargs:", kwargs)
        self.dim_hid = dim_hid
        self.num_clusters = num_clusters
        self.phis = nn.Sequential(*[nn.Conv2d(dim_in, dim_hid, 1), nn.ReLU()]) # It's equivalent to FC + ReLU
        self.pool1d = nn.AdaptiveAvgPool1d(1)    
        
        # attention pooling layer for clusters
        self.attention_net = nn.Sequential(*[
            nn.Linear(dim_hid, dim_hid), nn.ReLU(), nn.Dropout(dropout),
            Gated_Attention_Pooling(dim_hid, dim_hid, dropout=dropout)
        ])
        # output layer
        self.output_layer = nn.Linear(in_features=dim_hid, out_features=num_cls)

        print("[setup] initialized a DeepAttnMISL model.")

    def forward(self, X, cluster_id, *args):
        if cluster_id is not None:
            cluster_id = cluster_id.detach().cpu().numpy()
        X = X.squeeze(0) # assert batch_size = 1
        # FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            x_cluster_i = X[cluster_id==i].T.unsqueeze(0).unsqueeze(2) # [N, d] -> [1, d, 1, N]
            h_cluster_i = self.phis(x_cluster_i) # [1, d, 1, N] -> [1, d', 1, N]
            if h_cluster_i.shape[-1] == 0: # no any instance in this cluster
                h_cluster_i = torch.zeros((1, self.dim_hid, 1, 1), device=X.device)
            h_cluster.append(self.pool1d(h_cluster_i.squeeze(2)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0) # [num_clusters, d']
        H, A = self.attention_net(h_cluster) # [1, d'], [1, num_clusters]
        out = self.output_layer(H)
        return out

################################################
#    PatchGCN (Chen et al., MICCAI, 2021)
################################################
from torch_geometric.nn import GENConv, DeepGCNLayer


class PatchGCN(nn.Module):
    """
        Adapted from the official implementation: 
        - https://github.com/mahmoodlab/Patch-GCN/blob/master/models/model_graph_mil.py#L116
    """
    def __init__(self, dim_in=512, dim_hid=256, num_cls=4, num_layers:int=3, edge_agg:str='spatial', dropout:float=0.25, **kwargs):
        super().__init__()
        self.edge_agg = edge_agg
        self.num_layers = num_layers
        self.fc = nn.Sequential(*[nn.Linear(dim_in, dim_hid), nn.ReLU(), nn.Dropout(dropout)])
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv = GENConv(dim_hid, dim_hid, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(dim_hid, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=(i+1)%3)
            self.layers.append(layer)
        dim_sum = dim_hid * (1 + self.num_layers)
        self.path_phi = nn.Sequential(*[nn.Linear(dim_sum, dim_hid), nn.ReLU(), nn.Dropout(dropout)])
        # attention pooling layer for graph nodes
        self.path_attention_head = Gated_Attention_Pooling(dim_hid, dim_hid, dropout=dropout)
        # output layer
        self.output_layer = nn.Linear(in_features=dim_hid, out_features=num_cls)

        print("[setup] initialized a PatchGCN model.")

    def forward(self, x_path, *args):
        data = x_path
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        edge_attr = None
        x = self.fc(data.x)
        x_ = x 
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        h_path = x_ # [N, dim_sum], dim_sum = dim_hid * (1 + num_layers)
        h_path = self.path_phi(h_path) 
        H, A = self.path_attention_head(h_path) # [1, d'], [1, N]
        out = self.output_layer(H)
        return out

################################################
# DSMIL, li et al., CVPR, 2021.
################################################

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
    
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)
        
    def forward(self, feats, c, **kwargs): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 


class DSMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_hid=256, num_cls=2, use_feat_proj=True, drop_rate=0.25, **kwargs):
        super(DSMIL, self).__init__()
        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        self.i_classifier = FCLayer(in_size=dim_in, out_size=num_cls)
        self.b_classifier = BClassifier(dim_in, dim_hid, num_cls, dropout_v=drop_rate)
        
    def forward(self, X, **kwargs):
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        X = X.squeeze(0) # to [N, C] for input to i and b classifier
        feats, classes = self.i_classifier(X)
        prediction_bag, A, B = self.b_classifier(feats, classes) # bag = [1, C], A = [N, C]
        
        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction) # logits = [1, C]

        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            # average over class heads
            attn = A.detach()
            attn = attn.mean(dim=1).unsqueeze(0)
            return logits, attn
        
        return logits

