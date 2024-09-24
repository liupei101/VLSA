import os
import os.path as osp
import torch
from torch import Tensor
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import random
import h5py
import json


#############################################
#           General IO functions
#############################################
def read_patch_data(path: str, dtype:str='torch', key='features'):
    r"""Read patch data from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
        key (string): Key of return data, default 'features'.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            pdata = hf[key][:]
    elif ext == '.pt':
        pdata = torch.load(path, map_location=torch.device('cpu'))
    elif ext == '.npy':
        pdata = np.load(path)
    else:
        raise ValueError(f'Not support {ext}')

    if isinstance(pdata, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(pdata)
    elif isinstance(pdata, Tensor) and dtype == 'numpy':
        return pdata.numpy()
    else:
        return pdata

def load_patch_feats(slide_ids, cfg, verbose=False):
    all_feats = []
    for sid in slide_ids:
        if verbose:
            print(f"read feats from {sid}...")
        path_patch = osp.join(cfg['path_patch'], sid + '.pt')
        feats = read_patch_data(path_patch, dtype='torch')
        all_feats.append(feats)
    return all_feats

def read_patch_feats_from_uid(uid: str, cfg):
    full_path = osp.join(cfg['path_patch'], uid + '.' + cfg['feat_format'])
    feats = read_patch_data(full_path, dtype='torch').to(torch.float)
    return feats

def read_patch_coord(path: str, dtype:str='torch'):
    r"""Read patch coordinates from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']

    with h5py.File(path, 'r') as hf:
        coords = hf['coords'][:]

    if isinstance(coords, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(coords)
    else:
        return coords 

def load_patch_coords(slide_ids, cfg, verbose=False):
    all_coords = []
    for sid in slide_ids:
        if verbose:
            print(f"read coords from {sid}...")
        path_coord = osp.join(cfg['path_coord'], sid + '.h5')
        coors = read_patch_coord(path_coord, dtype='torch')
        all_coords.append(coors)
    return all_coords

def infer_columns_for_splitting(available_columns):
    columns_with_key_words = ('train', 'test', 'val')

    ret_columns = []
    for c in columns_with_key_words:
        target_c = None
        for a_c in available_columns:
            if c in a_c:
                target_c = a_c
        ret_columns.append(target_c)

    train_col, test_col, val_col = ret_columns
    if test_col is None:
        test_col = val_col
        val_col  = None

    assert train_col is not None, "The column corresponding to `train` is not found."
    assert test_col is not None, "The column corresponding to `test` is not found."
    
    return train_col, test_col, val_col

def read_file_data_splitting(path: str):
    _, ext = osp.splitext(path)

    data_split = dict()
    if ext == '.npz':
        data_npz = np.load(path)
        column_train, column_test, column_validation = infer_columns_for_splitting([_ for _ in data_npz.keys()])
        
        pids_train = [str(s) for s in data_npz[column_train]]
        data_split['train'] = pids_train
        print(f"[data split] there are {len(pids_train)} cases for train.")
        
        pids_test = [str(s) for s in data_npz[column_test]]
        data_split['test'] = pids_test
        print(f"[data split] there are {len(pids_test)} cases for test.")

        if column_validation is not None:
            pids_val = [str(s) for s in data_npz[column_validation]]
            data_split['validation'] = pids_val
            print(f"[data split] there are {len(pids_val)} cases for validation.")

    elif ext == '.csv':
        data_csv = pd.read_csv(path)
        column_train, column_test, column_validation = infer_columns_for_splitting([_ for _ in data_csv.columns])

        pids_train = [str(s) for s in data_csv[column_train].dropna()]
        data_split['train'] = pids_train
        print(f"[data split] there are {len(pids_train)} cases for train.")

        pids_test = [str(s) for s in data_csv[column_test].dropna()]
        data_split['test'] = pids_test
        print(f"[data split] there are {len(pids_test)} cases for test.")

        if column_validation is not None:
            pids_val = [str(s) for s in data_csv[column_validation].dropna()]
            data_split['validation'] = pids_val
            print(f"[data split] there are {len(pids_val)} cases for validation.")

    return data_split

def read_maxt_from_table(path: str, at_column='t'):
    df = pd.read_csv(path)
    return df[at_column].max()

def load_init_prompt(prompt_path, context_idx=0, rank_idx=0, replace=False):
    """
    prompt_path (str): it should be a JSON file. 
        Its content would like {"context_templates": [xx, xx], "class_names": {'0':[xx, xx], '1':[xx, xx]}}.
    context_idx (int): The index of the list with "context_templates" to choose.
    rank_idx (int): The index of the list with all class names to choose.
    replace (bool): if replace the CLASSNAME in context with rank names
    """
    if prompt_path is None:
        prompt_context, prompt_rank_names = None, None

    else:
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        prompt_context = prompts['context_templates'][context_idx]
        prompt_rank_names = []
        for k in prompts['class_names'].keys():
            rank_name = prompts['class_names'][k][rank_idx]
            if replace:
                rank_name = prompt_context.replace("CLASSNAME", rank_name)
            prompt_rank_names.append(rank_name)

    return prompt_context, prompt_rank_names

def load_init_text(path, key=None):
    with open(path, 'r') as f:
        texts = json.load(f)

    if key is None:
        return texts
    else:
        return texts[str(key)]

#############################################
#  IO functions for classification models
#############################################
def retrieve_from_table_clf(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id'):
    """Get info from table, oriented to classification tasks"""
    assert level in ['slide', 'patient']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 'label']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    pid2loc = dict()
    for i in df.index:
        _p = df.loc[i, pid_column]
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} not found in table {}.'.format(p, table_path))
            continue
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _label = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = _label

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label[_sid] = _label

    if shuffle:
        if level == 'patient':
            pid = random.shuffle(pid)
        else:
            sid = random.shuffle(sid)

    res = []
    for r in ret:
        res.append(eval(r))
    return res

def save_prediction_clf(uids, y_true, y_pred, save_path, binary=True, **kws):
    r"""Save classification prediction.

    Args:
        y_true (Tensor or ndarray): true labels, typically with shape (N, ).
        y_pred (Tensor or ndarray): final predicted probabilities, typically with shape (N, num_cls).
        save_path (string): path to save.
        binary (boolean): if it is a binary prediction.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.squeeze().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    assert ((y_pred >= 0.0) & (y_pred <= 1.0)).all(), "Prediction must be probabilities in [0, 1]."

    assert len(uids) == len(y_true)
    assert len(uids) == len(y_pred)

    save_data = {'uids': uids, 'y': y_true}
    cols = ['uids', 'y']
    if binary:
        save_data['y_hat'] = y_pred[:, 1]
        cols.append('y_hat')
    else:
        for i in range(y_pred.shape[-1]):
            _col = 'y_hat_' + str(i)
            save_data[_col] = y_pred[:, i]
            cols.append(_col)

    df = pd.DataFrame(save_data, columns=cols)
    df.to_csv(save_path, index=False)

#########################################
#    IO functions for survival models
#########################################
def retrieve_from_table_surv(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id', time_format='origin', time_bins=4):
    assert level in ['slide', 'patient']
    assert time_format in ['origin', 'ratio', 'quantile']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 't', 'e']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    pid2loc = dict()
    max_time = 0.0
    for i in df.index:
        max_time = max(max_time, df.loc[i, 't'])
        _p = df.loc[i, pid_column]
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    # process time format
    from utils.func import compute_discrete_label
    if time_format == 'ratio':
        df.loc[:, 't'] = 1.0 * df.loc[:, 't'] / max_time
    elif time_format == 'quantile':
        df, new_columns = compute_discrete_label(df, bins=time_bins)
        assert_columns  = [pid_column, 'pathology_id'] + new_columns
    else:
        pass

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} not found in table {}.'.format(p, table_path))
            continue
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _t, _ind = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = (_t, _ind)

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label = (_t, _ind)

    if shuffle:
        if level == 'patient':
            pid = random.shuffle(pid)
        else:
            sid = random.shuffle(sid)

    res = []
    for r in ret:
        res.append(eval(r))
    return res

def save_prediction_surv(patient_id, y_true, y_pred, save_path, **kws):
    r"""Save surival prediction.

    Args:
        y_true (Tensor or ndarray): true labels, typically with shape [N, 2].
        y_pred (Tensor or ndarray): final predicted values, typically with shape [N, 1].
        save_path (string): path to save.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    assert len(patient_id) == len(y_true)
    assert len(patient_id) == len(y_pred)
    
    if y_pred.shape[1] == 1: # continuous model
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)
        t, e = y_true[:, 0], y_true[:, 1]
        df = pd.DataFrame(
            {'patient_id': patient_id, 't': t, 'e': e, 'pred': y_pred}, 
            columns=['patient_id', 't', 'e', 'pred']
        )
    else:
        bins = y_pred.shape[1]
        y_t, y_e = y_true[:, [0]], y_true[:, [1]]
        if 'type_pred' in kws and ('IF' in kws['type_pred'] or kws['type_pred'] == 'incidence'):
            survival = 1.0 - np.cumsum(y_pred, axis=1)
        else:
            survival = np.cumprod(1.0 - y_pred, axis=1)
        risk = np.sum(survival, axis=1, keepdims=True)
        arr = np.concatenate((y_t, y_e, risk, survival), axis=1) # [B, 3+BINS]
        df = pd.DataFrame(arr, columns=['t', 'e', 'risk'] + ['surf_%d' % (_ + 1) for _ in range(bins)])
        df.insert(0, 'patient_id', patient_id)
    df.to_csv(save_path, index=False)
    print("[info] saved survival prediction to {} with kws = {}.".format(save_path, kws))
