import os
import os.path as osp
import time
from tqdm import tqdm
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import wandb

from dataset.GraphBatchWSI import collate_MIL_graph
from model.utils import load_model, general_init_weight
from loss.utils import load_loss, loss_reg_l1
from eval.utils import load_evaluator
from dataset.utils import prepare_clf_dataset
from optim import create_optimizer

from utils.func import setup_device, seed_everything
from utils.func import parse_str_dims, fetch_kws
from utils.func import print_config, save_config
from utils.func import print_metrics, EarlyStopping
from utils.func import add_prefix_to_filename, rename_keys
from utils.func import seed_generator, seed_worker
from utils.func import create_output_converter, fill_placeholder
from utils.io import read_file_data_splitting, save_prediction_clf
from utils.io import load_init_text
from runner.global_cfg import CFG as GLOBAL_CFG


class BaseHandler(object):
    """
    This class handles the initialization, training, and 
    testing of general WSI representation learning models.
    """
    def __init__(self, cfg):
        setup_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        dataset_name = cfg['dataset_name']
        print(f"[setup] dataset name: {dataset_name}.")
        # Path setup
        if not cfg['test']:
            ### Fill placeholder in cfg
            cfg['save_path'] = fill_placeholder(cfg['save_path'], dataset_name[5:], ind="{0}")
            temp_keys = [
                'path_patch', 'path_coord', 'path_cluster', 'path_graph', 'path_table', 'data_split_path', 
                'vlsa_img_encoder_query_text_load_idx'
            ]
            for temp_key in temp_keys:
                if temp_key in cfg:
                    cfg[temp_key] = fill_placeholder(cfg[temp_key], dataset_name, ind="{0}")

            temp_keys = ['path_patch', 'path_cluster', 'path_graph', 'path_coord']
            for temp_key in temp_keys:
                if temp_key in cfg:
                    cfg[temp_key] = fill_placeholder(cfg[temp_key], GLOBAL_CFG[dataset_name]['disk_location'], ind="{1}")

            cfg['data_split_path'] = fill_placeholder(cfg['data_split_path'], cfg['data_split_seed'], ind="{2}")

            temp_key = 'vlsa_img_encoder_num_query'
            if temp_key in cfg:
                if cfg[temp_key] is None:
                    init_texts = load_init_text(cfg['vlsa_img_encoder_query_text_load_path'], key=cfg['vlsa_img_encoder_query_text_load_idx'])
                    cfg[temp_key] = len(init_texts)
                    print(f"[info] found null `vlsa_img_encoder_num_query`; filled with {cfg[temp_key]} (the length of texts to load).")
                else:
                    temp_num_query = fill_placeholder(cfg[temp_key], GLOBAL_CFG[dataset_name]['num_query'], ind="{3}")
                    cfg[temp_key] = int(temp_num_query)
            
            ### End of fill placeholder in cfg

            if not osp.exists(cfg['save_path']):
                os.makedirs(cfg['save_path'])
            run_name = cfg['save_path'].split('/')[-1]
            self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
            self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
            self.last_metrics_path = osp.join(cfg['save_path'], 'metrics-last.txt')
            self.best_metrics_path = osp.join(cfg['save_path'], 'metrics-best.txt')
            self.config_path = osp.join(cfg['save_path'], 'print_config.txt')
            self.config_yaml = osp.join(cfg['save_path'], 'config.yaml')
            self.writer = wandb.init(
                project=cfg['wandb_prj'], name=run_name, 
                dir=cfg['wandb_dir'], config=cfg, reinit=True
            )
            print(f"[setup] path to save: {cfg['save_path']}")
        else:
            if "{}" in cfg['test_load_path']:
                cfg['test_load_path'] = cfg['test_load_path'].format(cfg['data_split_seed'])
            if not osp.exists(cfg['test_save_path']):
                os.makedirs(cfg['test_save_path'])
            run_name = cfg['test_save_path'].split('/')[-1]
            self.last_ckpt_path = osp.join(cfg['test_load_path'], 'model-last.pth')
            self.best_ckpt_path = osp.join(cfg['test_load_path'], 'model-best.pth')
            self.last_metrics_path = osp.join(cfg['test_save_path'], 'metrics-last.txt')
            self.best_metrics_path = osp.join(cfg['test_save_path'], 'metrics-best.txt')
            self.config_path = osp.join(cfg['test_save_path'], 'print_config.txt')
            self.config_yaml = osp.join(cfg['test_save_path'], 'config.yaml')
            self.writer = wandb.init(
                project=cfg['test_wandb_prj'], name=run_name, 
                dir=cfg['wandb_dir'], config=cfg, reinit=True
            )
            print(f"[setup] in test mode, loading path: {cfg['test_load_path']}")
            print(f"[setup] in test mode, saving path : {cfg['test_save_path']}")

        # Data setup
        self.data_split = self.func_load_data_split(cfg)
        self.data_meta = self.func_load_meta_data(cfg, data_split=self.data_split)
            
        # Others setup
        self.net = self.func_load_model(cfg).cuda()
        self.loss, self.loss_weight = self.func_load_loss(cfg)
        self.add_network_loss(cfg)
        
        self.optimizer = self.func_load_optimizer(self.net, cfg)
        self.steplr = self.func_load_lrs(self.optimizer, cfg)
        
        self.output_converter = create_output_converter(cfg['net_output_converter'])
        self.evaluator, self.metrics_list, self.ret_metrics = self.func_load_evaluator(cfg, meta_data=self.data_meta)
        print(f"[setup] the {self.ret_metrics[0]} is expected to be larger; the {self.ret_metrics[1]} is expected to be smaller.") 
        print("[setup] for a successful run, please ensure the above is correct in your setting.")

        # checking arguments
        self._check_arguments(cfg)

        self.uid = dict()
        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)
        save_config(cfg, self.config_yaml)

    def _check_arguments(self, cfg):
        print("[setup] start checking all arguments...")
        pass
        print("[setup] argument checking passed.")

    @staticmethod
    def func_load_data_split(cfg):
        path_split = cfg['data_split_path']
        dataset_pids = read_file_data_splitting(path_split)
        print('[exec] in data splitting, finished reading patient ID from {}'.format(path_split))
        return dataset_pids

    @staticmethod
    def func_load_meta_data(cfg, data_split=None):
        print("[setup] there is no meta data to generate.")
        return None

    @staticmethod
    def func_load_model(cfg):
        dims = parse_str_dims(cfg['net_dims'])
        arch = cfg['arch']
        arch_cfg = fetch_kws(cfg, prefix=arch.lower())
        model = load_model(cfg['arch'], dims, **arch_cfg)
        if cfg['init_wt']:
            model.apply(general_init_weight)
        return model

    @staticmethod
    def func_load_loss(cfg):
        loss_names = parse_str_dims(cfg['loss_type'], dtype=str)
        kws_loss = {'loss_type': loss_names}
        loss_weight = dict()
        for loss_name in loss_names:
            kws_loss[loss_name] = fetch_kws(cfg, prefix=f'loss_{loss_name.lower()}')
            key_loss_weight = f'loss_{loss_name.lower()}_weight'
            if key_loss_weight not in cfg:
                cur_loss_weight = 1
            else:
                cur_loss_weight = cfg[key_loss_weight]
            loss_weight[loss_name] = cur_loss_weight
            print("[setup] {}: {}.".format(loss_name, kws_loss[loss_name]))
        loss = load_loss(cfg['task'], **kws_loss)
        return loss, loss_weight

    def add_network_loss(self, cfg):
        pass

    @staticmethod
    def func_load_optimizer(model, cfg):
        cfg_optimizer = SimpleNamespace(opt=cfg['opt_name'], weight_decay=cfg['opt_weight_decay'], 
            lr=cfg['opt_lr'], opt_eps=None, opt_betas=None, momentum=None)
        optimizer = create_optimizer(cfg_optimizer, model)
        return optimizer

    @staticmethod
    def func_load_lrs(optimizer, cfg):
        if cfg['lrs'] is None or not cfg['lrs']:
            print("[setup] learning rate scheduler is disabled.")
            return None

        kws_lrs = fetch_kws(cfg, prefix='lrs')
        steplr = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=kws_lrs['factor'], 
            patience=kws_lrs['patience'], 
            verbose=True
        )
        return steplr

    @staticmethod
    def func_load_evaluator(cfg, meta_data=None):
        assert cfg['evaluator'] in ['Binary', 'Multi-class']
        evaluator = load_evaluator(cfg['task'], cfg['evaluator'])
        if cfg['loss_bce']: # binary
            metrics_list = [
                'auc', 'loss', 'acc', 'acc@mid', 'acc_best', 
                'recall', 'precision', 'f1_score', 'ece', 'mce'
            ]
        else: # multi-class
            metrics_list = ['auc', 'loss', 'acc', 'macro_f1_score', 'micro_f1_score']
        ret_metrics = ['auc', 'loss']
        return evaluator, metrics_list, ret_metrics

    @staticmethod
    def func_prepare_dataset(patient_ids, set_name, cfg, meta_data=None):
        dataset = prepare_clf_dataset(patient_ids, cfg)
        return dataset

    @staticmethod
    def save_prediction_results(data_cltor, path_to_save, **kws):

        save_prediction_clf(data_cltor['uid'], data_cltor['y'], data_cltor['y_hat'], path_to_save, **kws)

    def exec(self):
        print('[exec] with task = {}, arch = {}.'.format(self.cfg['task'], self.cfg['arch']))

        if self.cfg['data_mode'] == 'graph':
            collate_func = collate_MIL_graph
        else:
            collate_func = default_collate

        # Prepare datasets 
        pids_train = self.data_split['train']
        train_set  = self.func_prepare_dataset(pids_train, 'train', self.cfg, self.data_meta)
        self.uid.update({'train': train_set.uid})
        train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=True, worker_init_fn=seed_worker, collate_fn=collate_func
        )

        pids_test  = self.data_split['test']        
        test_set   = self.func_prepare_dataset(pids_test, 'test', self.cfg, self.data_meta)
        self.uid.update({'test': test_set.uid})
        test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False, worker_init_fn=seed_worker, collate_fn=collate_func
        )

        # if the split contains a validation splitting 
        if 'validation' in self.data_split:
            pids_val = self.data_split['validation']
            val_set  = self.func_prepare_dataset(pids_val, 'validation', self.cfg, self.data_meta)
            self.uid.update({'validation': val_set.uid})
            val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=False, worker_init_fn=seed_worker, collate_fn=collate_func
            )
        else:
            val_set    = None 
            val_loader = None

        run_name = 'train'
        zero_shot_eval = False
        # Train
        if 'force_to_skip_training' in self.cfg and self.cfg['force_to_skip_training']:
            print("[exec] warning: your training is skipped...")
        elif 'num_shot' in self.cfg and self.cfg['num_shot'] == 0:
            zero_shot_eval = True
            run_name = 'zero-shot'
            print("[exec] warning: at zero-shot mode, your training is skipped...")
        else:
            val_name = 'validation'
            val_loaders = {'validation': val_loader, 'test': test_loader}
            
            if 'eval_training_loader_per_epoch' in self.cfg and self.cfg['eval_training_loader_per_epoch']:
                train_loader_for_eval = DataLoader(
                    train_set, batch_size=self.cfg['batch_size'], 
                    generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                    shuffle=False,  worker_init_fn=seed_worker, collate_fn=collate_func
                )
                val_loaders['eval-train'] = train_loader_for_eval
            
            if self.cfg['es'] is None or not self.cfg['es']:
                run_with_es = False
            else:
                run_with_es = True

            self._run_training(
                self.cfg['epochs'], 
                train_loader, 'train', 
                val_loaders=val_loaders, 
                val_name=val_name, 
                measure_training_set=True, 
                save_ckpt=True, 
                early_stop=run_with_es, 
                run_name=run_name
            )

        # report performance using the best/last ckpt
        train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False,  worker_init_fn=seed_worker, collate_fn=collate_func
        )

        if zero_shot_eval:
            evals_loader = {'test': test_loader}
            print("[exec] start zero-shot evaluation on test.")
            metrics = self._eval_all(evals_loader, ckpt_type='zero-shot', run_name=run_name, if_print=True)
        else:
            evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
            ckpt_to_use = self.cfg['ckpt_for_eval'] # best / last
            print("[exec] start evaluation on train/val/test with the trained model ({}).".format(ckpt_to_use))
            metrics = self._eval_all(evals_loader, ckpt_type=ckpt_to_use, run_name=run_name, if_print=True)
        
        return metrics

    def exec_test(self):
        print('[exec] test under task = {}, arch = {}.'.format(self.cfg['task'], self.cfg['arch']))
        mode_name = 'test_mode'

        if self.cfg['data_mode'] == 'graph':
            collate_func = collate_MIL_graph
        else:
            collate_func = default_collate
        
        # Prepare datasets 
        if self.cfg['test_path'] == 'train':
            pids = self.data_split['train']
        elif self.cfg['test_path'] == 'validation':
            pids = self.data_split['validation']
        elif self.cfg['test_path'] == 'test':
            pids = self.data_split['test']
        else:
            pass
        print('[exec] test patient IDs from {}'.format(self.cfg['test_path']))

        # Prepare datasets 
        test_set = self.func_prepare_dataset(pids, self.cfg['test_path'], self.cfg, self.data_meta)
        self.uid.update({'exec-test': test_set.uid})

        test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False, worker_init_fn=seed_worker, collate_fn=collate_func
        )

        # Evals
        evals_loader = {'exec-test': test_loader}
        ckpt_to_use  = self.cfg['ckpt_for_eval'] # best / last
        metrics = self._eval_all(evals_loader, ckpt_type=ckpt_to_use, if_print=True, test_mode=True, 
            test_mode_name=mode_name, test_in_between_data=False)
        return metrics

    def _run_training(self, epochs, train_loader, name_loader, val_loaders=None, val_name=None, 
        measure_training_set=True, save_ckpt=True, early_stop=False, run_name='train', **kws):
        """Traing model.

        Args:
            epochs (int): Epochs to run.
            train_loader ('DataLoader'): DatasetLoader of training set.
            name_loader (string): name of train_loader, used for infering patient IDs.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, which gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure_training_set (bool): If measure training set at each epoch.
            save_ckpt (bool): If save models.
            early_stop (bool): If early stopping according to validation loss.
            run_name (string): Name of this training, which would be used as the prefixed name of ckpt files.
        """
        # setup early_stopping
        if early_stop and self.cfg['es_patience'] is not None:
            self.early_stop = EarlyStopping(
                warmup=self.cfg['es_warmup'], 
                patience=self.cfg['es_patience'], 
                start_epoch=self.cfg['es_start_epoch'], 
                verbose=self.cfg['es_verbose']
            )
        else:
            self.early_stop = None

        if val_name is not None:
            assert val_name in val_loaders.keys(), f"{val_name} is not found in current `val_loaders`."
            if val_loaders[val_name] is not None:
                print(f"[{run_name}] specified a dataloader ({val_name}) for early stopping or lr_scheduler.")
            else:
                print(f"[{run_name}] no dataset is specified, so early stopping or lr_scheduler is not active.")
        else:
            print(f"[{run_name}] no dataset is specified, so early stopping or lr_scheduler is not active.")
        
        # iterative training
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch + 1
            train_cltor = self._train_each_epoch(epoch+1, train_loader, name_loader)
            cur_name = name_loader

            if measure_training_set:
                for k_cltor, v_cltor in train_cltor.items():
                    self._eval_and_print(v_cltor, name=cur_name+'/'+k_cltor, at_epoch=epoch+1)

            # val/test
            monitor_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    val_cltor = self.test_model(self.net, val_loaders[k], k)
                    for k_cltor, v_cltor in val_cltor.items():
                        met_main, met_loss = self._eval_and_print(v_cltor, name=k+'/'+k_cltor, at_epoch=epoch+1)
                        if k == val_name and k_cltor == 'pred':
                            monitor_metrics  = 0
                            # closely-related to the default behavior asserted at lines 84 & 85
                            monitor_metrics += met_loss if 'loss' in self.cfg['monitor_metrics'] else 0
                            monitor_metrics += -1 * met_main if 'main' in self.cfg['monitor_metrics'] else 0
            
            if self.steplr is not None:
                self.steplr.step(monitor_metrics)

            if self.early_stop is not None:                
                self.early_stop(epoch, monitor_metrics)
                if self.early_stop.save_ckpt():
                    self.save_model(epoch+1, ckpt_type='best', run_name=run_name)
                    print("[train] {} best model saved at epoch {}".format(run_name, epoch+1))
                if self.early_stop.stop():
                    break
        
        if save_ckpt:
            self.save_model(last_epoch, ckpt_type='last', run_name=run_name) # save models and optimizers
            print("[train] {} last model saved at epoch {}".format(run_name, last_epoch))

    def _train_each_epoch(self, epoch, train_loader, name_loader):
        self.net.train()
        bp_every_batch = self.cfg['bp_every_batch']
        all_raw_pred, all_gt, all_idx = [], [], []

        idx_collector, x_collector, y_collector = [], [], []
        i_batch = 0
        num_samples = len(train_loader)
        loop = tqdm(train_loader, desc=name_loader)
        for data_idx, data_x, data_y in loop:
            # data_x = (feats, coords) | data_y = (label_slide, label_patch)
            i_batch += 1
            # 1. read data (mini-batch)
            data_input = data_x[0] # only use the first item
            data_label = data_y[0]

            data_input = data_input.cuda()
            data_label = data_label.cuda()

            x_collector.append(data_input)
            y_collector.append(data_label)
            idx_collector.append(data_idx)

            # in a mini-batch
            if i_batch % bp_every_batch == 0 or i_batch == num_samples:
                # 2. update network
                batch_loss, batch_pred = self._update_network(x_collector, y_collector)
                all_raw_pred.append(batch_pred)
                all_gt.append(torch.cat(y_collector, dim=0).detach().cpu())
                all_idx.append(torch.cat(idx_collector, dim=0).detach().cpu())

                # 3. reset mini-batch
                idx_collector, x_collector, y_collector = [], [], []
                torch.cuda.empty_cache()

                # 4. log and print
                wandb.log({'train/batch_loss': batch_loss})
                loop.set_description(f"Epoch [{epoch}/{self.cfg['epochs']}]")
                loop.set_postfix(loss=batch_loss)

        all_raw_pred = torch.cat(all_raw_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0).squeeze(1) # [B, ]
        all_idx = torch.cat(all_idx, dim=0).squeeze(-1) # [B, ]

        train_cltor = dict()
        # As it will be used for evaluation
        all_pred = self.output_converter(all_raw_pred)
        all_uids = self._get_unique_id('train', all_idx)
        train_cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids}
        return train_cltor

    def calc_objective_loss(self, pred, label):
        label = label.squeeze(-1) # [B, 1] -> [B, ]
        # We assume the loss function can directly handle the raw output of network,
        # so herein we don't explicitly convert the network's raw outputs.
        clf_loss = self.loss(pred, label)
        return clf_loss

    def _update_network(self, xs, ys):
        """
        Update network using one batch data
        """
        n_sample = len(xs)
        y_hat = []

        for i in range(n_sample):
            pred = self.net(xs[i])
            y_hat.append(pred)

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0) # [B, 2]
        pred_loss = self.calc_objective_loss(bag_preds, bag_label)

        # 3.3 backward gradients and update networks
        if isinstance(pred_loss, torch.Tensor) and pred_loss.requires_grad:
            pred_loss.backward()
            self.optimizer.step()
            val_loss = pred_loss.item()
        else:
            print("[batch train] warning: loss is not evaluated; skipped this batch training.")
            val_loss = 0

        val_preds = bag_preds.detach().cpu()
        return val_loss, val_preds

    def _eval_all(self, evals_loader, ckpt_type='best', run_name='train', task='bag_clf', if_print=True,
        test_mode=False, test_mode_name='test_mode', **kwargs):
        """
        test_mode = True only if run self.exec_test(), indicating a test mode.
        """
        if test_mode:
            print('[eval] warning: you are in test mode now.')
            ckpt_run_name = 'train'
            wandb_group_name = test_mode_name
            metrics_path_name = test_mode_name
            csv_prefix_name = test_mode_name
            save_pred_path = self.cfg['test_save_path']
        else:
            ckpt_run_name = run_name
            wandb_group_name = run_name
            metrics_path_name = run_name
            csv_prefix_name = run_name
            save_pred_path = self.cfg['save_path']
        
        if ckpt_type == 'best':
            ckpt_path = add_prefix_to_filename(self.best_ckpt_path, ckpt_run_name)
            wandb_group = 'bestckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.best_metrics_path, metrics_path_name)
            csv_name = '{}_{}_best'.format(task, csv_prefix_name)
        elif ckpt_type == 'last':
            ckpt_path = add_prefix_to_filename(self.last_ckpt_path, ckpt_run_name)
            wandb_group = 'lastckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.last_metrics_path, metrics_path_name)
            csv_name = '{}_{}_last'.format(task, csv_prefix_name)
        elif ckpt_type == 'zero-shot':
            ckpt_path = None
            wandb_group = 'lastckpt/{}'.format(wandb_group_name) # lastckpt/zero-shot
            print_path = add_prefix_to_filename(self.best_metrics_path, metrics_path_name) # zero-shot_path
            csv_name = '{}_{}_last'.format(task, csv_prefix_name) # task_zero-shot_last
        else:
            pass

        metrics = dict()
        for k, loader in evals_loader.items():
            if loader is None:
                continue
            cltor = self.test_model(self.net, loader, k, ckpt_path=ckpt_path)
            metrics[k] = []
            for k_cltor, v_cltor in cltor.items():
                met_main, met_loss = self._eval_and_print(
                    v_cltor, 
                    name='{}/{}/{}'.format(wandb_group, k, k_cltor), 
                    at_epoch=ckpt_type
                )
                metrics[k].append((f"{k_cltor}_{self.ret_metrics[0]}", met_main))
                metrics[k].append((f"{k_cltor}_{self.ret_metrics[1]}", met_loss))

            used_cltor = cltor['pred']
            if self.cfg['save_prediction']:
                full_path_save = osp.join(save_pred_path, '{}_pred_{}.csv'.format(csv_name, k))
                self.save_prediction_results(used_cltor, full_path_save, type_pred=self.cfg['evaluator'])

        if if_print:
            print_metrics(metrics, print_to_path=print_path)

        return metrics

    def _eval_and_print(self, cltor, name='', ret_metrics=None, at_epoch=None):
        if ret_metrics is None:
            ret_metrics = self.ret_metrics
        if at_epoch is None:
            at_epoch = 'NA'
        eval_metrics = self.metrics_list
        eval_results = self.evaluator.compute(cltor, eval_metrics)
        eval_results = rename_keys(eval_results, name, sep='/')

        print("[{}] At epoch {}:".format(name, at_epoch), end=' ')
        print(' '.join(['{}={:.6f},'.format(k, v) for k, v in eval_results.items()]))
        wandb.log(eval_results)

        return [eval_results[name+'/'+k] for k in ret_metrics]

    def _get_unique_id(self, k, idxs, concat=None):
        if k not in self.uid:
            raise KeyError('Key {} not found in `uid`'.format(k))
        uids = self.uid[k]
        idxs = idxs.squeeze().tolist()
        if concat is None:
            return [uids[i] for i in idxs]
        else:
            return [uids[v] + "-" + str(concat[i].item()) for i, v in enumerate(idxs)]

    def test_model(self, model, loader, loader_name, ckpt_path=None):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'], strict=False)
        model.eval()

        all_idx, all_raw_pred, all_pred, all_gt = [], [], [], []
        for data_idx, data_x, data_y in loader:
            # data_x = (feats, coords) | data_y = (label_slide, label_patch)
            X = data_x[0].cuda() 
            data_label = data_y[0] 
            with torch.no_grad():
                raw_pred = model(X)
                # To convert raw predictions for 
                # evaluation and prediction saving
                pred = self.output_converter(raw_pred)
            all_gt.append(data_label)
            all_raw_pred.append(raw_pred.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_idx.append(data_idx)
        
        all_raw_pred = torch.cat(all_raw_pred, dim=0) # [B, num_cls]
        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0).squeeze() # [B, ]
        all_idx = torch.cat(all_idx, dim=0).squeeze() # [B, ]

        cltor = dict()
        all_uids = self._get_unique_id(loader_name, all_idx)
        cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids}

        return cltor

    def _get_state_dict(self, epoch):
        # filter specified modules
        if 'model_saver_module_filter' in self.cfg:
            module_filter = self.cfg['model_saver_module_filter']
            assert isinstance(module_filter, str)
            print(f"[warning] modules with `{module_filter}` will not be saved.")
        else:
            module_filter = None
            print(f"[info] all modules will be saved.")

        model_state_dict = self.net.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        if module_filter is not None:
            _new_model_state_dict = dict()
            for k, v in model_state_dict.items():
                if module_filter in k:
                    continue
                _new_model_state_dict.update({k: v})
            model_state_dict = _new_model_state_dict

            _new_optimizer_state_dict = dict()
            for k, v in optimizer_state_dict.items():
                if module_filter in k:
                    continue
                _new_optimizer_state_dict.update({k: v})
            optimizer_state_dict = _new_optimizer_state_dict

        return {
            'epoch': epoch,
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
        }

    def save_model(self, epoch, ckpt_type='best', run_name='train'):
        net_ckpt_dict = self._get_state_dict(epoch)
        if ckpt_type == 'last':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.last_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.best_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))

    def resume_model(self, ckpt_type='best', run_name='train'):
        if ckpt_type == 'last':
            net_ckpt = torch.load(add_prefix_to_filename(self.last_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            net_ckpt = torch.load(add_prefix_to_filename(self.best_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))
        self.net.load_state_dict(net_ckpt['model'], strict=False)
        self.optimizer.load_state_dict(net_ckpt['optimizer'], strict=False)
        print('[model] resume the network from {}_{} at epoch {}...'.format(ckpt_type, run_name, net_ckpt['epoch']))

