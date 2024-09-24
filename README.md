# VLSA: Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology

*On updating*

 [[arXiv Preprint]](https://arxiv.org/abs/2409.09369) | [[VLSA Walkthrough]](https://github.com/liupei101/VLSA?tab=readme-ov-file#vlsa-walkthrough) | [[Awesome VLM Papers]](https://github.com/liupei101/VLSA?tab=readme-ov-file#-awesome-vlm-papers) | [[WSI Preprocessing]](https://github.com/liupei101/VLSA?tab=readme-ov-file#wsi-preprocessing) | [[Citation]](https://github.com/liupei101/VLSA?tab=readme-ov-file#-citation)

**Abstract**: Histopathology Whole-Slide Images (WSIs) provide an important tool to assess cancer prognosis in computational pathology (CPATH). While existing survival analysis (SA) approaches have made exciting progress, they are generally limited to adopting highly-expressive architectures and only coarse-grained patient-level labels to learn prognostic visual representations from gigapixel WSIs. Such learning paradigm suffers from important performance bottlenecks, when facing present scarce training data and standard multi-instance learning (MIL) framework in CPATH. To break through it, this paper, for the first time, proposes a new Vision-Language-based SA (**VLSA**) paradigm. Concretely, (1) VLSA is driven by pathology VL foundation models. It no longer relies on high-capability networks and shows the advantage of *data efficiency*. (2) In vision-end, VLSA encodes prognostic language prior and then employs it as *auxiliary signals* to guide the aggregating of prognostic visual features at instance level, thereby compensating for the weak supervision in MIL. Moreover, given the characteristics of SA, we propose i) *ordinal survival prompt learning* to transform continuous survival labels into textual prompts; and ii) *ordinal incidence function* as prediction target to make SA compatible with VL-based prediction. VLSA's predictions can be interpreted intuitively by our Shapley values-based method. The extensive experiments on five datasets confirm the effectiveness of our scheme. Our VLSA could pave a new way for SA in CPATH by offering weakly-supervised MIL an effective means to learn valuable prognostic clues from gigapixel WSIs.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="100%" height="auto" src="./docs/fig-vlsa-overview.png"></a>
</div>

---

📚 Recent updates:
- 24/09/24: codes & papers are live
- 24/09/10: release VLSA

## VLSA Walkthrough

*in progress*

## 🔥 Awesome VLM Papers

**Vision-Language Foundation Models for Computational Pathology**:

| Model          | Architecture | Paper             | Code            | Data   |
| :------------- | :---------------- | :---------------- | :-------------- | :----- |
| PLIP (NatMed'23) | [CLIP](https://github.com/openai/CLIP) | [A visual language foundation model for pathology image analysis using medical twitter](https://www.nature.com/articles/s41591-023-02504-3) | [Github](https://github.com/PathologyFoundation/plip) | 208,414 pathology images paired with natural language descriptions from twitter |
| Quilt-Net (NeurIPS'23) | [CLIP](https://github.com/openai/CLIP) | [Quilt-1M: One million image-text pairs for histopathology](https://papers.neurips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html) | [Github](https://github.com/wisdomikezogwo/quilt1m)            | 802,148 image and text pairs from YouTube  |
| CONCH (NatMed'24) | [CoCa](https://arxiv.org/pdf/2205.01917) | [A Vision-Language Foundation Model for Computational Pathology](https://www.nature.com/articles/s41591-024-02856-4) | [Github](https://github.com/mahmoodlab/CONCH) | over 1.17 million image-caption pairs  |
| PathAlign (arXiv'24) | [BLIP-2](https://arxiv.org/abs/2301.12597) | [PathAlign: A vision-language model for whole slide images in histopathology](https://arxiv.org/abs/2406.19578) | -  |  over 350,000 WSIs and diagnostic text pairs |

**WSI Classification or Survival Analysis with Vision-Language Models**:

| Model          | Subfield    | Paper             | Code            | Base   |
| :------------- | :---------- | :---------------- | :-------------- | :----- |
| TOP (NeurIPS'23)      | WSI Classification    | [The rise of ai language pathologists: Exploring two-level prompt learning for few-shot weakly-supervised whole slide image classification](https://papers.nips.cc/paper_files/paper/2023/hash/d599b81036fd1a3b3949b7d444f31082-Abstract-Conference.html) | [Github](https://github.com/miccaiif/TOP)            | Few-shot WSI classification   |
| FiVE (CVPR'24)     | WSI Classification    | [Generalizable whole slide image classification with fine-grained visual-semantic interaction](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Generalizable_Whole_Slide_Image_Classification_with_Fine-Grained_Visual-Semantic_Interaction_CVPR_2024_paper.pdf) | [Github](https://github.com/ls1rius/WSI_FiVE)            | VLM pretraining for WSI classification  |
| ViLa-MIL (CVPR'24)         | WSI Classification   | [Vila-mil: Dual-scale vision language multiple instance learning for whole slide image classification](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ViLa-MIL_Dual-scale_Vision-Language_Multiple_Instance_Learning_for_Whole_Slide_Image_CVPR_2024_paper.pdf) | [Github](https://github.com/Jiangbo-Shi/ViLa-MIL)                 | Dual-scale features for WSI classification |
| CPLIP (CVPR'24)     | WSI Classification    | [CPLIP: Zero-Shot Learning for Histopathology with Comprehensive Vision-Language Alignment](https://openaccess.thecvf.com/content/CVPR2024/papers/Javed_CPLIP_Zero-Shot_Learning_for_Histopathology_with_Comprehensive_Vision-Language_Alignment_CVPR_2024_paper.pdf) | [Github](https://github.com/iyyakuttiiyappan/CPLIP)            | Zero-shot WSI classification  |
| VLSA (arXiv'24)         | WSI Survival Analysis | [Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology](https://arxiv.org/abs/2409.09369) | [Github](https://github.com/liupei101/VLSA)  | VLM-driven vision-language survival analysis |

**NOTE**: please open *a new PR* if you want to add your work into this table.

## WSI Preprocessing

Following [CONCH](https://github.com/mahmoodlab/CONCH), we first divide each WSI into patches of 448 * 448 pixels at 20x magnification. Then we adopt the image encoder of CONCH to extract patch features.

Our complete procedure in WSI preprocessing follows [Pipeline-Processing-TCGA-Slides-for-MIL](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL). You could move to it for a detailed tutorial.

## 📝 Citation

If you find this work helps your research, please consider citing our paper:
```txt
@misc{liu2024interpretablevisionlanguagesurvivalanalysis,
    title={Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology}, 
    author={Pei Liu and Luping Ji and Jiaxiang Gou and Bo Fu and Mao Ye},
    year={2024},
    eprint={2409.09369},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.09369}, 
}
```