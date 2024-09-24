"""
This is our entry file to run all experiments
"""
import argparse
import yaml
import numpy as np
import time

from runner import BaseHandler, SAHandler, VLSAHandler
from utils.func import args_grid, print_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--handler', '-d', type=str, choices=['SA', 'VLSA', 'CLF'], default='VLSA', help='Model handler.')
    parser.add_argument('--multi_run', action='store_true', help='If execute multi-experiments in this run.')
    parser.add_argument('--sleep', type=int, default=0, help='If sleep X seconds between two runs, only valid in multi_run mode.')
    args = vars(parser.parse_args())
    return args

def get_config(config_path):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def main(handler, config):
    model = handler(config)
    if config['test']:
        metrics = model.exec_test()
    else:
        metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def convert_to_abbr(key):
    ABBR_MAPS = {
        'vlsa_img_encoder_name': 'mil',
        'vlsa_img_encoder_query': 'que',
        'vlsa_img_encoder_query_pooling': 'qpool',
        'vlsa_img_encoder_query_text_method': 'tex',
        'vlsa_img_encoder_query_text_load_idx': 'qkey',
        'vlsa_img_encoder_gated_query': 'gatq',
        'vlsa_img_encoder_query_text_res_ratio': 'resr',
        'vlsa_img_encoder_pred_head': 'head',
        'vlsa_pmt_learner_coop_method': 'coop',
        'vlsa_pmt_learner_adapter_method': 'adap',
        'data_split_seed': 'fold',
        'num_shot': 'shot',
        'seed_shot': 'fssd',
        'vlsa_img_encoder_pooling': 'pool',
        'dataset_name': 'data',
    }

    if key in ABBR_MAPS.keys():
        print(f"[info] abbreviate {key} as {ABBR_MAPS[key]}.")
        return ABBR_MAPS[key]
    else:
        return key

def ignore_it_in_save_path(key, value):
    IGNORE_LIST = {
        'num_shot': lambda x: x < 0,
        'dataset_name': lambda x: True,
    }

    if key in IGNORE_LIST.keys():
        judge_func = IGNORE_LIST[key]
        return judge_func(value)

    return False

def multi_run_main(handler, config, sleep=0):
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    configs = args_grid(config)
    for cur_cfg in configs:
        print('\n')
        for k in hyperparams:
            abbr_key = convert_to_abbr(k)
            abbr_value = convert_to_abbr(cur_cfg[k])
            
            if ignore_it_in_save_path(k, cur_cfg[k]):
                print(f"[info] `{k}` is ignored and will not be added to `save_path`.")
                continue

            cur_cfg['save_path'] += '-{}_{}'.format(abbr_key, abbr_value)
            if cur_cfg['test']:
                cur_cfg['test_save_path'] += '-{}_{}'.format(abbr_key, abbr_value)

        model = handler(cur_cfg)
        if cur_cfg['test']:
            print(cur_cfg['test_save_path'])
            metrics = model.exec_test()
        else:
            print(cur_cfg['save_path'])
            metrics = model.exec()

        time.sleep(sleep)
        
        print('[INFO] Metrics:', metrics)

if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    print_config(config)
    
    if cfg['handler'] == 'SA':
        handler = SAHandler
    elif cfg['handler'] == 'VLSA':
        handler = VLSAHandler
    elif cfg['handler'] == 'CLF':
        handler = BaseHandler
    else:
        handler = None
        raise RuntimeError(f"Expected `SA`, `VLSA`, or `CLF` but got {cfg['handler']}")

    if cfg['multi_run']:
        multi_run_main(handler, config, sleep=cfg['sleep'])
    else:
        main(handler, config)
