from dataset.PatchWSI import WSIPatchClf
from dataset.PatchWSI import WSIPatchSurv
from dataset.PatchWSI import FewShot_WSIPatchSurv


def prepare_clf_dataset(patient_ids:list, cfg, **kws):
    """
    Interface for preparing slide-level classification dataset

    patient_ids: a list including all patient IDs.
    cfg: a dict where 'path_patch', 'path_table', and 'feat_format' are included.
    """
    path_patch = cfg['path_patch']
    path_table = cfg['path_table']
    feat_format = cfg['feat_format']
    if 'path_label' in kws:
        path_label = kws['path_label']
    else:
        path_label = None
    if 'ratio_sampling' in kws:
        ratio_sampling = kws['ratio_sampling']
    else:
        ratio_sampling = None
    if 'ratio_mask' in kws:
        if cfg['test']: # only used in a test mode
            ratio_mask = kws['ratio_mask']
        else:
            ratio_mask = None
    else:
        ratio_mask = None
    if 'filter_slide' in kws:
        if_remove_slide = kws['filter_slide']
    else:
        if_remove_slide = None
    if 'random_patch_path' in kws:
        random_patch_path = kws['random_patch_path']
    else:
        random_patch_path = False

    dataset = WSIPatchClf(
        patient_ids, path_patch, path_table, path_label=path_label, read_format=feat_format, ratio_sampling=ratio_sampling, 
        ratio_mask=ratio_mask, coord_path=cfg['path_coord'], filter_slide=if_remove_slide, random_patch_path=random_patch_path
    )
    return dataset

def prepare_surv_dataset(patient_ids:list, cfg, **kws):
    """
    Interface for preparing patient-level survival dataset

    patient_ids: a list including all patient IDs.
    cfg: a dict where 'path_patch', 'path_table', and 'feat_format' are included.
    kws: additional kws. The argument `meta_data` must be specified.
    """
    path_patch = cfg['path_patch']
    mode = cfg['data_mode']
    feat_format = cfg['feat_format']
    if 'ratio_sampling' in kws:
        ratio_sampling = kws['ratio_sampling']
    else:
        ratio_sampling = None

    assert 'meta_data' in kws, "The argument `meta_data` must be specified."
    meta_data = kws['meta_data']

    dataset = WSIPatchSurv(
        patient_ids, path_patch, mode, meta_data, read_format=feat_format, ratio_sampling=ratio_sampling,
        cluster_path=cfg['path_cluster'], coord_path=cfg['path_coord'], graph_path=cfg['path_graph']
    )

    if 'num_shot' in kws and kws['num_shot'] > 0:
        seed = kws['seed_shot'] if 'seed_shot' in kws else 42
        dataset = FewShot_WSIPatchSurv(dataset, kws['num_shot'], seed)
    
    return dataset
