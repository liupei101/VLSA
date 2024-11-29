import pandas as pd
import numpy as np
import math

from eval.SurvivalEVAL.Evaluations.MeanError import get_best_guess_from_training_data

EPS = 1e-5

def calculate_uncensored_time_bins(patient_ids, meta_data, ret_continuous_time=False):
    actual_label = meta_data.get_patient_data(patient_ids, ret_columns=['t', 'e'])
    # num_bins = math.ceil(math.sqrt(actual_label.e.values.sum()))
    actual_label['uncensored_t'] = get_best_guess_from_training_data(actual_label.t.values, actual_label.e.values, method="KM")
    
    if 'discrete' in meta_data.label_format: # for discrete time label
        time_bins = meta_data.time_bins

    else: # for continuous time label (with its default behavior)
        print("[info] discrete label is not found or not valid, so discrete time_bins will be calculated here.")
        time_bins = calculate_discrete_time_bins(
            actual_label, column_t='t', column_e='e', 
            num_bins=None, use_quantiles=False, max_time=meta_data.max_t
        )

    # prevent from NAN output by pd.cut
    actual_label['uncensored_t'].clip(time_bins[0] + EPS, time_bins[-1] - EPS, inplace=True)
    
    uncensored_time_bins = pd.cut(
        actual_label['uncensored_t'], bins=time_bins, retbins=False, 
        labels=False, right=False, include_lowest=True
    )
    uncensored_time_bins = uncensored_time_bins.values.astype(int)

    if ret_continuous_time:
        uncensored_time_bins, actual_label.uncensored_t.values

    return uncensored_time_bins

def calculate_discrete_time_bins(patient_data, column_t='t', column_e='e', num_bins=None, use_quantiles=False, max_time=None):
    df_events = patient_data[patient_data[column_e] == 1]
    event_times = df_events[column_t]

    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(event_times)))
        print('[time-to-event converter] found null `time_bins`; generated a new one.')
    print('[time-to-event converter] current length of `time_bins` = {}'.format(num_bins))
    
    if use_quantiles:
        _, qbins = pd.qcut(event_times, q=num_bins, retbins=True, labels=False)
    else:
        qbins = np.linspace(0, event_times.max(), num_bins + 1)

    if max_time is None:
        max_time = patient_data[column_t].max()
    # set the first bin to [0, the_second_cutpoint)
    # set the last bin to [the_last_cutpoint, +inf), where MAX_TIME + EPS -> +inf
    qbins[0] = 0
    qbins[-1] = max_time + 1e-5

    return qbins

def to_patient_data(df, at_column='patient_id'):
    df_gps = df.groupby(at_column).groups
    df_idx = [i[0] for i in df_gps.values()]
    return df.loc[df_idx, :]

def get_index_by_values(df, values, at_column='patient_id', select_element='first'):
    assert select_element in ['first', 'last', 'all']
    ret_idxs = []
    for v in values:
        sel_idxs = df[df[at_column] == v].index
        if len(sel_idxs) > 0:
            if len(sel_idxs) > 1:
                if select_element != 'all':
                    print("[warning] found {} candidates for v = {}; will select the {} one.".format(len(sel_idxs), v, select_element))
                if select_element == 'first':
                    ret_idxs.append(sel_idxs[0])
                elif select_element == 'last':
                    ret_idxs.append(sel_idxs[-1])
                else:
                    ret_idxs += [_idx for _idx in sel_idxs]
            else:
                ret_idxs.append(sel_idxs[0])
        else:
            print("[warning] v = {} is not found.".format(v))
    return ret_idxs


class MetaSurvData(object):
    """
    MetaSurvData: handling survival data for training.
    """
    def __init__(self, path_label, column_t='t', column_e='e', verbose=True, **kws):
        super().__init__()
        self.path_label = path_label
        self.column_t = column_t
        self.column_e = column_e
        self.column_label = None
        self.label_format = None
        self.time_bins = None

        self.full_data = pd.read_csv(
            path_label, 
            dtype={'patient_id': str, 'pathology_id': str}
        )
        self.pat_data = to_patient_data(self.full_data, at_column='patient_id')

        if 'data_split' in kws:
            assert isinstance(kws['data_split'], dict), "Please use a dict to specify `data_split`."
            self.data_split = kws['data_split']
        else:
            self.data_split = None

        self.min_t = self.pat_data[column_t].min()
        self.max_t = self.pat_data[column_t].max()
        
        if verbose:
            print('[time-to-event converter] at patient level')
            print('\tmeta data is loaded from {}'.format(self.path_label))
            print('\tmin/avg/median/max time = {}/{:.2f}/{}/{}'.format(self.min_t, 
                self.pat_data[column_t].mean(), self.pat_data[column_t].median(), self.max_t))
            print('\tratio of event = {}'.format(self.pat_data[column_e].sum() / len(self.pat_data)))

    def get_patient_data(self, pids=None, split=None, ret_columns=None):
        if pids is None:
            if split is not None:
                assert split in self.data_split.keys(), f"split ({split}) cannot be found in the key of `data_split`."
                pids = self.data_split[split]

        if ret_columns is None:
            ret_columns = list(self.pat_data.columns)
        
        if pids is not None:
            pat_idxs = get_index_by_values(self.pat_data, pids, select_element='first')
            return self.pat_data.loc[pat_idxs, ret_columns]
        
        return self.pat_data.loc[:, ret_columns]

    @property
    def num_bins(self):
        if self.time_bins is None:
            return None
        return len(self.time_bins) - 1

    @property
    def time_coordinates(self):
        if self.time_bins is None:
            return None
        return self.time_bins[:-1]

    def generate_continuous_label(self, new_column_t='y_t', new_column_e='y_e', normalize=False):
        print('[time-to-event converter] to continuous time with normalize = {}'.format(normalize))
        self.column_label = [new_column_t, new_column_e]

        # e = 0 -> no event/censored, e = 1 -> event/uncensored
        self.pat_data.loc[:, new_column_e] = self.pat_data.loc[:, self.column_e]

        if normalize:
            if self.data_split is not None:
                pat_idxs = get_index_by_values(self.pat_data, self.data_split['train'])
                MAX_TIME = self.pat_data.loc[pat_idxs, self.column_t].max()
                print('[time-to-event converter] infer `MAX_TIME` from training patients.')
            else:
                MAX_TIME = self.max_t
                print('[time-to-event converter] infer `MAX_TIME` from all patients.')
            self.pat_data.loc[:, new_column_t] = self.pat_data.loc[:, self.column_t].apply(lambda x: min(1.0, x / MAX_TIME))
            self.label_format = 'continuous_ratio'
        else:
            self.pat_data.loc[:, new_column_t] = self.pat_data.loc[:, self.column_t]
            self.label_format = 'continuous_time'
        
        return self.pat_data

    def generate_discrete_label(self, num_bins=None, new_column_t='y_t', new_column_e='y_e', use_quantiles=True, summary=True):
        """
        based on the quartiles of survival time values (in months) of uncensored patients.
        Refer to Chen et al. Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images.
        """        
        self.column_label = [new_column_t, new_column_e]

        # e = 0 -> no event/censored, e = 1 -> event/uncensored
        self.pat_data.loc[:, new_column_e] = self.pat_data.loc[:, self.column_e]

        if use_quantiles:
            self.label_format = 'discrete_quantile'
        else:
            self.label_format = 'discrete_uniform'

        # To discrete time labels
        # 1. generate time_bins on training data or full data
        # 1.1 select target patient data
        if self.data_split is not None:
            pat_idxs = get_index_by_values(self.pat_data, self.data_split['train'])
            cur_pat_data = self.pat_data.loc[pat_idxs, :]
            print('[time-to-event converter] infer `time_bins` from training patients.')
        else:
            cur_pat_data = self.pat_data
            print('[time-to-event converter] infer `time_bins` from all patients.')

        # 1. get number of time bins, and obtain time bins
        qbins = calculate_discrete_time_bins(
            cur_pat_data, column_t=self.column_t, column_e=self.column_e, 
            num_bins=num_bins, use_quantiles=use_quantiles, max_time=self.max_t
        )

        # 2. convert survival labels
        discrete_labels, qbins = pd.cut(
            self.pat_data[self.column_t], bins=qbins, retbins=True, 
            labels=False, right=False, include_lowest=True
        )
        self.pat_data.loc[:, new_column_t] = discrete_labels.values.astype(int)

        self.time_bins = qbins
        print('[time-to-event converter] time_bins: {}.'.format(self.time_bins))

        if summary:
            self.summary_subgroups()

        return self.pat_data

    def summary_subgroups(self, verbose=False):
        if 'discrete' in self.label_format:
            num_bins = self.num_bins
            for i in range(num_bins):
                if not verbose and (i != 0 or i != num_bins - 1):
                    continue
                
                sub_pat_data = self.pat_data[self.pat_data[self.column_label[0]] == i]

                min_t = sub_pat_data[self.column_t].min()
                max_t = sub_pat_data[self.column_t].max()
                print(f'[Group = {i}] statistic at patient level')
                print('\t- number of patients = {}'.format(len(sub_pat_data)))
                print('\t- min/avg/median/max time = {} / {:.2f} / {} / {}'.format(min_t, 
                    sub_pat_data[self.column_t].mean(), sub_pat_data[self.column_t].median(), max_t))
                print('\t- ratio of event = {}'.format(sub_pat_data[self.column_e].sum() / len(sub_pat_data)))
        else:
            print("Please convert to discrete labels at first.")

    def collect_info_by_pids(self, pids, column_label=None):
        if column_label is None:
            column_label = self.column_label

        sel_pids, pid2sids, pid2label = list(), dict(), dict()
        for pid in pids:
            sel_idxs = self.full_data[self.full_data['patient_id'] == pid].index
            if len(sel_idxs) > 0:
                sel_pids.append(pid)
                pid2sids[pid] = list(self.full_data.loc[sel_idxs, 'pathology_id'])
                
                pat_idx = self.pat_data[self.pat_data['patient_id'] == pid].index[0]
                pid2label[pid] = list(self.pat_data.loc[pat_idx, column_label])
            else:
                print('[time-to-event converter] Warning: patient {} not found.'.format(pid))

        return sel_pids, pid2sids, pid2label