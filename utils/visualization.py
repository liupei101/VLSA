"""
Partial codes for visualization are borrowed from https://github.com/mahmoodlab/PANTHER/tree/main/src/visualization.
"""
import cv2
import numpy as np
import torch
from torch import Tensor
from PIL import Image, ImageOps
import argparse
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm, rcParams
from matplotlib import patches as mpatches
import seaborn as sns


########################################################
#              barplot (horizontal plot)
########################################################
def get_mixture_plot(mixtures, ticks_step=0.1, margin=0.1, xlabel='Language-encoded prognostic prior', ylabel='Contribution to risk'):
    if isinstance(mixtures, Tensor):
        mixtures = mixtures.numpy()
    
    colors = [
        '#696969','#556b2f','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1']

    min_c, max_c = min(mixtures.min() - margin, 0), mixtures.max() + margin
    cmap = {f'p{k}':v for k,v in enumerate(colors[:len(mixtures)])}
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(5, 3), dpi=300)

    prop = fm.FontProperties(fname="./tools/Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mixtures = pd.DataFrame(mixtures, index=cmap.keys()).T
    ax = sns.barplot(mixtures, palette=cmap)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_xlabel(xlabel, fontproperties=prop, fontsize=14)
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=14)
    y_ticks = [round(round(min_c, 1) + ticks_step * i, 1) for i in range(1+int((max_c - min_c) / ticks_step))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontproperties = prop, fontsize=14)
    x_ticks = [i for i in range(len(cmap))]
    ax.set_xticks(x_ticks)
    x_ticks = ["p$_{" + str(i+1) + "}$" for i in range(len(cmap))]
    ax.set_xticklabels(x_ticks, fontproperties=prop, fontsize=14)
    ax.set_ylim([min_c, max_c])
    plt.close()
    return ax.get_figure()


########################################################
#               barplot (vertical plot)
########################################################
def get_imp_plot(imp, ticks_step=0.1, margin=0.1, ylabel='Language-encoded prognostic prior', xlabel='Contribution to risk'):
    if isinstance(imp, Tensor):
        imp = imp.numpy()
    
    colors = [
        '#696969','#556b2f','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1']

    min_c, max_c = min(imp.min() - margin, 0), imp.max() + margin
    cmap = {f'p{k}':v for k, v in enumerate(colors[:len(imp)])}
    mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(3.4, 4), dpi=300)

    prop = fm.FontProperties(fname="./tools/Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    idx = [k for k in cmap.keys()]
    imp = pd.DataFrame({'x': idx[::-1], 'y': imp[::-1]})
    ax = sns.barplot(imp, x='y', y='x', palette=cmap, hue='x', legend=False)
    plt.axis('on')
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=14)
    ax.set_xlabel(xlabel, fontproperties=prop, fontsize=14)
    x_ticks = [round(round(min_c, 1) + ticks_step * i, 1) for i in range(1+int((max_c - min_c) / ticks_step))]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontproperties=prop, fontsize=14)
    ax.set_yticks(idx[::-1])
    ax.set_yticklabels(idx[::-1], fontproperties=prop, fontsize=14)
    ax.set_xlim([min_c, max_c])
    plt.close()
    return ax.get_figure()


########################################################
#                  Survival curves
########################################################
def get_survival_plot(pred_prob):
    # pred_prob: it should be the predicted incident function
    if isinstance(pred_prob, Tensor):
        pred_prob = pred_prob.numpy()

    pred_prob = np.squeeze(pred_prob)
    CIF = np.cumsum(pred_prob, axis=0)
    risk = CIF.sum()
    survival = 1 - CIF

    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(4.15, 4), dpi=300)

    prop = fm.FontProperties(fname="./tools/Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    data = pd.DataFrame({'IF': pred_prob, 'Survival': survival})
    ax = sns.lineplot(data, palette={'IF': '#ea6b66', 'Survival': '#117b7b'}, lw=1.8)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_ylabel('Predicted probability', fontproperties=prop, fontsize=12)
    ax.set_xlabel('Time interval', fontproperties=prop, fontsize=12)
    ax.set_xticks([i for i in range(len(data))])
    ax.set_xticklabels([f't{i}' for i in range(len(data))], fontproperties=prop, fontsize=12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontproperties=prop, fontsize=12)
    ax.set_ylim([0, 1.02])
    #title_prop = fm.FontProperties(fname="./tools/Arial.ttf", size=13)
    #ax.legend(title="Risk = %.3f"%risk, loc='center right', prop=prop, frameon=False, fontsize=12, title_fontproperties=title_prop)
    ax.legend(loc='best', prop=prop, frameon=False, fontsize=12)
    plt.close()
    return ax.get_figure()

def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = [
        '#696969','#556b2f','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1'
    ]
    
    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict


########################################################
#     Patch heatmap (prototypical clusters) on WSIs
########################################################
def visualize_categorical_heatmap(
        wsi,
        coords, 
        labels, 
        label2color_dict,
        vis_level=None,
        patch_size=(256, 256),
        canvas_color=(255, 255, 255),
        alpha=0.4,
        add_border=False,
        verbose=True,
    ):

    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]

    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size

    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in tqdm(range(len(coords))):
        coord = coords[idx]
        color = label2color_dict[labels[idx][0]]
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)

        if add_border:
            blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0])))

        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img


########################################################
#            Heatmap for ranking embeddings
########################################################
def analyze_ordinality(text_features, show_heatmap_tick=False, bar_tick_start=None, bar_tick_steps=4):
    # obtain the similarity score
    with torch.no_grad():
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = text_features @ text_features.t()
        logits_maxnorm = logits / logits.max()
    logits_maxnorm = logits_maxnorm.detach().cpu()

    # 1. visualization
    prop = fm.FontProperties(fname="./tools/Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    fig = plt.figure(figsize=(5, 4), dpi=300)
    if not show_heatmap_tick:
        ax = sns.heatmap(logits_maxnorm.cpu().numpy(), cmap="Reds", xticklabels=False, yticklabels=False)
    else:
        ax = sns.heatmap(logits_maxnorm.cpu().numpy(), cmap="Reds")
        ax.xaxis.tick_top()

    if bar_tick_start is None:
        bar_tick_start = int(logits_maxnorm.min().item() * 100) + 1 
    else:
        if bar_tick_start < 1:
            bar_tick_start = int(bar_tick_start * 100)
        else:
            bar_tick_start = int(bar_tick_start)
    ticks = [round(i / 100, 2) for i in range(bar_tick_start, 101, bar_tick_steps)]
    ax.collections[-1].colorbar.set_ticks(ticks)
    ax.collections[-1].colorbar.set_ticklabels(["%.2f"%t for t in ticks], fontproperties=prop, fontsize=8)

    if show_heatmap_tick:
        ticks = [i + 0.5 for i in range(logits_maxnorm.shape[0])]
        ax.set_yticks(ticks)
        ax.set_yticklabels([i for i in range(logits_maxnorm.shape[0])], fontproperties=prop, fontsize=9)
        ax.set_xticks(ticks)
        ax.set_xticklabels([i for i in range(logits_maxnorm.shape[0])], fontproperties=prop, fontsize=9)
    plt.close()
    
    text = ""
    # 2. ordinality statistics
    num_text_features = len(text_features)
    num_comparisons = num_text_features - 1
    ordinality_span_ls = [2 ** i for i in range(int(np.floor(np.log2(num_comparisons))) + 1)]
    ordinality_span_ls.append(num_comparisons)

    compare_sign_mat = torch.triu(logits_maxnorm[:-1, :-1] > logits_maxnorm[:-1, 1:])
    for span in ordinality_span_ls:
        mask = torch.ones_like(compare_sign_mat, dtype=torch.bool)
        mask_2 = ~torch.ones((num_comparisons - span, num_comparisons - span), dtype=torch.bool).triu()
        if num_comparisons - span > 0:
            mask[:(num_comparisons - span), -(num_comparisons - span):] &= mask_2
        mask = mask.triu()
        acc = ((compare_sign_mat * mask).sum() / mask.sum()).item()
        error = 1 - acc
        text += f"[Ordinality] current span = {span}: Acc = {acc}, Error = {error}.\n"

    return text, ax.get_figure()


########################################################
#            Attention Heatmap on WSIs
########################################################
def generate_pred_mask(downsample_sizes, mask_size, mask_level, pred, coord, pred_level, pred_patch_size=256, threshold=None):
    assert len(mask_size) == 2
    assert len(pred) == len(coord)
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(-1)
    n_channel = pred.shape[-1]
    # filter patches with lower scores
    if threshold is not None:
        assert isinstance(threshold, float)
        pred[pred < threshold] = 0.0
    
    mask = np.zeros((mask_size[1], mask_size[0], n_channel)) # pass (height, width) to generate matrix as a figure of size (width, height)
    downsample_mask = downsample_sizes[mask_level]
    downsample_pred = downsample_sizes[pred_level]
    mask_patch_size = pred_patch_size // (downsample_mask // downsample_pred)
    mask_coord = np.round(coord / downsample_mask)
    for i in range(len(mask_coord)):
        # fill the mask matrix
        c = mask_coord[i].astype(np.int32)
        mask[c[1]:(c[1]+mask_patch_size), c[0]:(c[0]+mask_patch_size)] = pred[i]
    return mask

def generate_heatmap(original_image, heat_map_2d_array, kernel_size=11, opacity=0.3, cmap=cv2.COLORMAP_TURBO, save_name=None, norm=True):
    # get height and width of the image
    height, width = original_image.shape[0], original_image.shape[1]

    # resize heat map to make sure it is equal to the size of image
    heat_map = cv2.GaussianBlur(heat_map_2d_array, (kernel_size, kernel_size), 0)
    heat_map = cv2.resize(heat_map, (width, height))

    # normalize heat map values from 0 to 255
    if norm:
        max_value = np.max(heat_map)
        min_value = np.min(heat_map)
        heat_map = (heat_map - min_value) / (max_value - min_value)
    heat_map = np.array(heat_map * 255, dtype=np.uint8)

    # apply color map to convert heat map into RGB values
    heat_map = cv2.applyColorMap(heat_map, cmap)

    # merge heat map and background image
    outImage = cv2.addWeighted(heat_map, opacity, original_image, 1 - opacity, 0)

    if save_name is not None:
        cv2.imwrite(save_name, outImage)

    finalImage = cv2.cvtColor(outImage, cv2.COLOR_BGR2RGB)

    return finalImage
