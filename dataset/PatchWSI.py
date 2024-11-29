"""
Class for bag-style dataloader
"""
from typing import Union
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.io import retrieve_from_table_clf
from utils.io import retrieve_from_table_surv
from utils.io import read_patch_data, read_patch_coord
from utils.func import sampling_data, random_mask_instance
from .label_converter import MetaSurvData
from .label_converter import calculate_uncensored_time_bins


class WSIPatchClf(Dataset):
    r"""A patch dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, label_path:Union[None,str]=None,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, ratio_mask=None, mode='patch', **kws):
        super(WSIPatchClf, self).__init__()
        if ratio_sampling is not None:
            assert ratio_sampling > 0 and ratio_sampling < 1.0
            print("[dataset] patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))
        if ratio_mask is not None and ratio_mask > 1e-5:
            assert ratio_mask <= 1, 'The argument ratio_mask must be not greater than 1.'
            assert mode == 'patch', 'Only support a patch mode for instance masking.'
            self.ratio_mask = ratio_mask
            print("[dataset] masking instances with ratio_mask = {}".format(ratio_mask))
        else:
            self.ratio_mask = None
        # used for randomly choosing slides to perform slide-level augmentation
        self.patch_path_random = kws['random_patch_path']
        self.patch_path_choice = ['feat-x20-RN50-B-color_norm-vflip', 'feat-x20-RN50-B-color_hed_light']

        self.read_path = patch_path
        self.label_path = label_path
        self.has_patch_label = (label_path is not None) and len(label_path) > 0
        
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')
        self.uid = self.sids
        
        assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        self.new_sid2label = None
        self.flag_use_corrupted_label = False
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws
        self.summary()

    def summary(self):
        print(f"[dataset] in {self.mode} mode, avaiable WSIs count {self.__len__()}")
        if self.patch_path_random:
            print("[dataset] randomly load patch features.")
        if not self.has_patch_label:
            print("[dataset] the patch-level label is not avaiable, derived by slide label.")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid] if not self.flag_use_corrupted_label else self.new_sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        if self.mode == 'patch':
            if self.patch_path_random:
                prob = np.random.rand()
                if prob <= 0.5:
                    cur_read_path = self.read_path
                else:
                    if prob <= 0.75:
                        cur_sub_path = self.patch_path_choice[0]
                    elif prob <= 1.00:
                        cur_sub_path = self.patch_path_choice[1]
                    else:
                        cur_sub_path = ""
                    temp_paths = self.read_path.split('/')
                    temp_paths[-2] = cur_sub_path
                    cur_read_path = "/".join(temp_paths)
                full_path = osp.join(cur_read_path, sid + '.' + self.read_format)
            else:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
            feats = read_patch_data(full_path, dtype='torch').to(torch.float)
            # if masking patches
            if self.ratio_mask:
                feats = random_mask_instance(feats, self.ratio_mask, scale=1, mask_way='mask_zero')
            full_coord = osp.join(self.kws['coord_path'],  sid + '.h5')
            coors = read_patch_coord(full_coord, dtype='torch')
            if self.has_patch_label:
                path = osp.join(self.label_path, sid + '.npy')
                patch_label = read_patch_data(path, dtype='torch', key='label').to(torch.long)
            else:
                patch_label = label * torch.ones(feats.shape[0]).to(torch.long)
            assert patch_label.shape[0] == feats.shape[0]
            assert coors.shape[0] == feats.shape[0]
            return index, (feats, coors), (label, patch_label)
        else:
            pass
            return None

    def corrupt_labels(self, corrupt_prob):
        """
        The implementation roughly follows https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
        """
        labels = np.array([self.sid2label[_sid] for _sid in self.sids])
        mask = np.random.rand(len(labels)) <= corrupt_prob
        labels[mask] = np.random.choice(labels.max() + 1, mask.sum()) # replaced with random labels
        # change the label
        cnt_wlabel = 0
        self.new_sid2label = dict()
        for i, _sid in enumerate(self.sids):
            if labels[i] != self.sid2label[_sid]:
                cnt_wlabel += 1
            self.new_sid2label[_sid] = int(labels[i])
        self.flag_use_corrupted_label = True
        print("[dataset] info: {:.2f}% corrupted labels with corrupt_prob = {}".format(cnt_wlabel / len(labels) * 100, corrupt_prob))

    def resume_labels(self):
        if self.flag_use_corrupted_label:
            self.flag_use_corrupted_label = False
            print("[dataset] info: the corrupted labels have been resumed.")


class WSIPatchSurv(Dataset):
    r"""A patch dataset class for classification tasks (patient-level in general).

    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    Return:
        index: The index of current item in the whole dataset.
        (feats, extra_data): Patch features and extra data.
        label: It contains typical survival labels, 'last follow-up time' and 'censorship status';
            censorship = 0 ---> uncersored, w/ event; censorship = 1 ---> cersored, w/o event.  
    """
    def __init__(self, patient_ids: list, patch_path: str, mode:str, meta_data:MetaSurvData,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, **kws):
        super().__init__()
        if ratio_sampling is not None:
            print("[dataset] Patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))

        assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws
        self.kws = kws

        self.pids, self.pid2sids, self.pid2label = meta_data.collect_info_by_pids(patient_ids)

        self.meta_data = meta_data
        self.uid = self.pids
        self.read_path = patch_path
        self.read_format = read_format
        self.summary()

    def summary(self):
        print(f"[Dataset] WSIPatchSurv: in {self.mode} mode, avaiable patients count {self.__len__()}.")

    def get_meta_data(self):
        return self.meta_data

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid   = self.pids[index]
        sids  = self.pid2sids[pid]
        label = self.pid2label[pid]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                if not osp.exists(full_path):
                    print(f"[WSIPatchSurv] warning: not found slide {sid}.")
                    continue
                feats.append(read_patch_data(full_path, dtype='torch'))

            feats = torch.cat(feats, dim=0).to(torch.float)
            return index, (feats, torch.Tensor([0])), label

        elif self.mode == 'cluster':
            cids = np.load(osp.join(self.kws['cluster_path'], '{}.npy'.format(pid)))
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                if not osp.exists(full_path):
                    print(f"[WSIPatchSurv] warning: not found slide {sid}.")
                    continue
                feats.append(read_patch_data(full_path, dtype='torch'))
            feats = torch.cat(feats, dim=0).to(torch.float)
            cids = torch.Tensor(cids)
            assert cids.shape[0] == feats.shape[0]
            return index, (feats, cids), label

        elif self.mode == 'graph':
            feats, graphs = [], []
            import torch_geometric
            from .GraphBatchWSI import GraphBatch
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                if not osp.exists(full_path):
                    print(f"[WSIPatchSurv] warning: not found slide {sid}.")
                    continue
                feats.append(read_patch_data(full_path, dtype='torch'))
                full_graph = osp.join(self.kws['graph_path'],  sid + '.pt')
                if not osp.exists(full_graph):
                    print(f"[WSIPatchSurv] warning: not found the graph of slide {sid}.")
                    continue
                graphs.append(torch.load(full_graph))
            feats = torch.cat(feats, dim=0).to(torch.float)
            graphs = GraphBatch.from_data_list(graphs, update_cat_dims={'edge_latent': 1})
            assert isinstance(graphs, torch_geometric.data.Batch)
            return index, (feats, graphs), label

        else:
            pass
            return None


class FewShot_WSIPatchSurv(Dataset):
    """
    WSIPatchSurv dataset for few shot learning
    """
    def __init__(self, dataset, num_shot, seed=0):
        super().__init__()
        self._dataset = dataset
        self.num_shot = num_shot
        self.seed = seed
        self.uid = dataset.uid
        self.meta_data = dataset.meta_data

        self.uncensored_time_bins = calculate_uncensored_time_bins(self.uid, self.meta_data, ret_continuous_time=False)
        event_labels = [dataset.pid2label[u][1] for u in self.uid] # label = ['t', 'e']

        self.few_shot_idx = self.get_few_shot_samples(self.uncensored_time_bins, event_labels, seed=seed)

        if num_shot > 0:
            print("[WSIPatchSurv] initialized {}-shot dataset with seed = {}.".format(num_shot, seed))
        else:
            print("[WSIPatchSurv] dataset remains as before for num_shot = {}.".format(num_shot))

    def get_few_shot_samples(self, discrete_time_labels, event_labels, preserve_order=True, seed=0):
        if not isinstance(discrete_time_labels, np.ndarray):
            discrete_time_labels = np.array(discrete_time_labels)
        if not isinstance(event_labels, np.ndarray):
            event_labels = np.array(event_labels)

        time_bins = np.array([_ for _ in range(self.meta_data.num_bins)])
        seed_rng = np.random.default_rng(seed)

        is_valid_sampling = False # the samples must have as least one uncensored patient
        while not is_valid_sampling:
            print("[FewShot_WSIPatchSurv] starting sampling...")
            few_shot_idx = []
            for t in time_bins:
                idx_of_t = np.where(discrete_time_labels == t)[0]
                if self.num_shot <= 0:
                    idx_of_t_few_shot = idx_of_t.tolist()
                else:
                    num_sample = min(self.num_shot, len(idx_of_t))
                    idx_of_t_few_shot = seed_rng.choice(idx_of_t, num_sample, replace=False).tolist()
                    if len(idx_of_t_few_shot) != self.num_shot:
                        print(f"[warning] only select {num_sample} samples from the {t}-th time for {self.num_shot}-shot.")
                few_shot_idx = few_shot_idx + idx_of_t_few_shot
            cnt_event = event_labels[few_shot_idx].sum()
            is_valid_sampling = cnt_event >= 1 and cnt_event < len(few_shot_idx)

        if preserve_order:
            few_shot_idx.sort()

        return few_shot_idx

    def get_meta_data(self):
        return self.meta_data

    def __getitem__(self, index):
        idx = self.few_shot_idx[index] # map index to the index in original dataset
        return self._dataset.__getitem__(idx)

    def __len__(self):
        return len(self.few_shot_idx)
