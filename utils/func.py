from typing import Union
import ast
import sys
import os.path as osp
import yaml
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as dist
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def check_list_consistency(A, B):
    _A, _B = list(A), list(B)
    assert len(_A) == len(_B), "Consistency check failed."
    for a in _A:
        assert a in _B, "Consistency check failed."
    for b in _B:
        assert b in _A, "Consistency check failed."

def fill_placeholder(target, fill, ind='{}'):
    _target = str(target)
    _fill = str(fill)
    if ind not in _target:
        return target

    new_target = _target.replace(ind, _fill)
    
    if isinstance(target, int):
        return int(new_target)
    elif isinstance(target, float):
        return float(new_target)

    return new_target

def create_output_converter(converter:Union[None,str]=None):
    if converter == 'sigmoid':
        func_converter = torch.sigmoid
    elif converter == 'softmax':
        func_converter = lambda x: F.softmax(x, dim=-1)
    else:
        func_converter = lambda x: x
    print(f"[setup] the converter of network output: {converter}.")
    return func_converter

def random_mask_instance(bag:Tensor, mask_ratio:float, scale=1, mask_way='mask_zero'):
    if mask_ratio <= 0 or mask_ratio > 1:
        return bag

    N = bag.shape[0]
    n_square = scale * scale
    assert N % n_square == 0, 'bag must consist of square instances.'
    N_scaled = N // n_square

    # calculate under the scaled version
    n_keep = max(1, int(N_scaled * (1 - mask_ratio)))
    idxs = np.random.permutation(N_scaled)
    idxs_keep = np.sort(idxs[:n_keep])
    
    # restore to the original scale
    idxs_keep = idxs_keep.reshape(-1, 1) * np.array([n_square] * n_square).reshape(1, -1) + \
        np.array([_ for _ in range(n_square)]).reshape(1, -1)
    idxs_keep = idxs_keep.reshape(-1).tolist()

    if mask_way == 'discard':
        return bag[idxs_keep]
    elif mask_way == 'mask_zero':
        new_bag = torch.zeros_like(bag)
        new_bag[idxs_keep] = bag[idxs_keep]
        return new_bag
    else:
        raise NotImplementedError("Not support for mask_way={}.".format(mask_way))

def add_prefix_to_filename(path, prefix=''):
    dir_name, file_name = osp.split(path)
    file_name = prefix + '_' + file_name
    return osp.join(dir_name, file_name)

def get_kfold_pids(pids, num_fold=5, keep_pids=None, random_state=42):
    kfold_pids = []
    cur_pids = [] if keep_pids is None else keep_pids
    if num_fold <= 1:
        kfold_pids.append(cur_pids + pids)
    else:
        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=random_state)
        X = np.ones((len(pids), 1))
        for _, fold_index in kfold.split(X):
            kfold_pids.append(cur_pids + [pids[_i] for _i in fold_index])
    return kfold_pids

def get_label_mask(t, c, bins):
    n = t.shape[0]
    z = (torch.arange(bins).view(1, -1) * torch.ones((n, 1))).to(t.device)
    label = torch.where(c.to(torch.bool), z > t, z == t).to(torch.float)
    label_mask = (z <= t).to(torch.int) # we ignore the location whose value is greater than t
    return label, label_mask

def get_patient_data(df:pd.DataFrame, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    pat_df = df.loc[df_idx, :]
    pat_df = pat_df.reset_index(drop=True)
    return pat_df

def sampling_data(data, num:Union[int,float]):
    total = len(data)
    if isinstance(num, float):
        assert num < 1.0 and num > 0.0
        num = int(total * num)
    assert num < total
    idxs = np.random.permutation(total)
    idxs_sampled = idxs[:num]
    idxs_left = idxs[num:]
    data_sampled = [data[i] for i in idxs_sampled]
    data_left = [data[i] for i in idxs_left]
    return data_sampled, data_left

def rename_keys(d, prefix_name, sep='/'):
    newd = dict()
    for k, v in d.items():
        newd[prefix_name + sep + k] = v
    return newd

def agg_tensor(collector, data):
    for k in data.keys():
        if k not in collector or collector[k] is None:
            collector[k] = data[k]
        else:
            collector[k] = torch.cat([collector[k], data[k]], dim=0)
    return collector

def fetch_kws(d, prefix:str=''):
    if prefix == '':
        return d
    else:
        ret = dict()
        for k in d.keys():
            if k.startswith(prefix):
                new_key = k.split(prefix)[1]
                if len(new_key) < 2:
                    continue
                ret[new_key[1:]] = d[k]
        return ret

def parse_str_dims(s, sep='-', dtype=int):
    if type(s) != str:
        return [s]
    else:
        return [dtype(_) for _ in s.split(sep)]

def set_requires_grad(module, requires_grad):
    if isinstance(module, torch.nn.Module):
        for param in module.parameters():
            param.requires_grad = requires_grad
    elif isinstance(module, torch.nn.parameter.Parameter):
        module.requires_grad = requires_grad
    else:
        raise TypeError(f"The type of the module is wrong: {type(module)}")

    return None

def freeze_param(module):
    if module is None:
        print("model is not instantialized.")
        return
    set_requires_grad(module, False)

def unfreeze_param(module):
    if module is None:
        print("model is not instantialized.")
        return
    set_requires_grad(module, True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))

def setup_device(cuda_id):
    if cuda_id is not None and cuda_id >= 0:
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_id)
            print(f'[setup] device: CUDA {cuda_id}.')
        else:
            raise RuntimeError(f'[setup] CUDA is not found.')
    else:
        raise RuntimeError(f'[setup] got invalid cuda_id ({cuda_id}).')

# worker_init_fn = seed_worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# generator = g
def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def sparse_str(s):
    res = None
    try:
        res = ast.literal_eval(s)
    except Exception as e:
        res = s
    return res

def get_model_cfg(path_run_log):
    full_path_run_log = osp.join(path_run_log, 'config.yaml')
    if osp.exists(full_path_run_log):
        with open(full_path_run_log, "r") as setting:
            cfg = yaml.load(setting, Loader=yaml.FullLoader)
        print("[Model CFG] loaded config from", full_path_run_log)
        return cfg

    full_path_run_log = osp.join(path_run_log, 'print_config.txt')
    if osp.exists(full_path_run_log):
        cfg = dict()
        with open(full_path_run_log, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if '-->' not in line:
                    continue
                k, v = line.split('-->')
                k, v = k.strip(), v.strip()
                cfg[k] = sparse_str(v)
        print("[Model CFG] loaded config from", full_path_run_log)
        return cfg
    
    raise RuntimeError(f"[Model CFG] Model configuration is not found in {path_run_log}.")

def print_config(config, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL CONFIGURATION ****************", file=f)
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val), file=f)
    print("**************** MODEL CONFIGURATION ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def save_config(config, path_to_save):
    with open(path_to_save, "w") as f:
        yaml.dump(config, f)

def print_metrics(metrics, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL METRICS ****************", file=f)
    for key in sorted(metrics.keys()):
        val = metrics[key]
        for v in val:
            cur_key = key + '/' + v[0]
            keystr  = "{}".format(cur_key) + (" " * (20 - len(cur_key)))
            valstr  = "{}".format(v[1])
            if isinstance(v[1], list):
                valstr = "{}, avg/std = {:.5f}/{:.5f}".format(valstr, np.mean(v[1]), np.std(v[1]))
            print("{} -->   {}".format(keystr, valstr), file=f)
    print("**************** MODEL METRICS ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def args_grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, start_epoch=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            start_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_checkpoint = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):

        self.save_checkpoint = False

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.update_score(val_loss)
        elif score - 1e-6 < self.best_score:
            self.counter += 1
            print(f'[early-stopping] counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.start_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_score(val_loss)
            self.counter = 0

    def stop(self, **kws):
        return self.early_stop

    def save_ckpt(self, **kws):
        return self.save_checkpoint

    def update_score(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'[early-stopping] validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss
        self.save_checkpoint = True
