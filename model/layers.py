import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import Nystromformer


def Instance_Embedding_Net(dim_in=1024, dim_out=1024, dropout=0.5, arch='default'):
    if arch == 'default':
        f = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )
    elif arch == 'nonlinear':
        f = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )
    else:
        f = nn.Identity()

    return f

def Instance_Dependency_Learning_Net(feat_dim, dropout=0.5, arch='default'):
    if arch == 'default':
        s = nn.Identity()
    elif arch == 'sa-tf':
        # self-attention layer
        patch_encoder_layer = nn.TransformerEncoderLayer(
            feat_dim, 8, dim_feedforward=feat_dim, 
            dropout=dropout, activation='relu', batch_first=True
        )
        s = nn.TransformerEncoder(patch_encoder_layer, num_layers=1)
    elif arch == 'sa-nf':
        s = Nystromformer(
            dim=feat_dim, depth=1, heads=8,
            attn_dropout=dropout
        )
    else:
        s = nn.Identity()

    return s


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(Feat_Projecter, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x = [B, N, C] or [N, C]
        if len(x.shape) == 3:
            L1, L2, L3 = x.shape[0], x.shape[1], x.shape[2]
            x = x.view(-1, L3)
            x = self.projecter(x) # for BatchNorm
            x = x.view(L1, L2, -1)
        else:
            x = self.projecter(x)
        return x


class Gated_Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by 
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim, hid_dim, dropout=0.5):
        super(Gated_Attention_Pooling, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x, ret_raw_attn=False):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        emb = self.fc1(x) # [B, N, d']
        scr = self.score(x) # [B, N, d'] \in [0, 1]
        new_emb = emb.mul(scr)
        A_ = self.fc2(new_emb) # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1) # [B, 1, N]
        A = F.softmax(A_, dim=2) # [B, 1, N]
        out = torch.matmul(A, x).squeeze(1) # [B, 1, d]
        if ret_raw_attn:
            A_ = A_.squeeze(1) # [B, N]
            return out, A_
        else:
            A = A.squeeze(1) # [B, N]
            return out, A


class Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by 
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim=1024, hid_dim=512):
        super(Attention_Pooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x, ret_raw_attn=True):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        A_ = self.attention(x)  # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1)  # [B, 1, N]
        attn = F.softmax(A_, dim=2)  # [B, 1, N]
        out = torch.matmul(attn, x).squeeze(1)  # [B, 1, N] bmm [B, N, d] = [B, 1, d]
        if ret_raw_attn:
            A_ = A_.squeeze(1)
            return out, A_
        else:
            A = A.squeeze(1)
            return out, A
