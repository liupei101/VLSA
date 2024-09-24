"""
VLSA's implementation compatitable with 
- OpenAI CLIP (github.com/openai/CLIP)
- HuggingFace CLIP (github.com/huggingface/transformers)
- mahmoodlab/CONCH (github.com/mahmoodlab/CONCH)
"""
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.prompt_learners import load_prompt_learner
from model.prompt_learners import load_prompt_adapter
from model.prompt_encoder import get_prompt_encoder
from model.utils_vl import load_vl_model_to_cpu
from model.utils_vl import Tokenizer
from model.deepmil import logit_pooling


class VLSA(nn.Module):
    def __init__(
        self,
        text_encoder_cfg,
        image_encoder_cfg,
        prompt_learner_cfg,
        pretrained_prompt_learner_cfg=None,
        info_prefix='VLSA-UNI',
        **kwargs,
    ) -> None:
        super().__init__()

        self.kwargs = kwargs
        print(f"[{info_prefix}] Found additional kwargs: {self.kwargs}.")
        assert 'vlsa_api' in kwargs, "Please specify `vlsa_api` in arguments."
        assert 'path_clip_model' in kwargs, "Please specify `path_clip_model` in arguments."

        self.text_tokenizer = Tokenizer(
            root=kwargs['path_clip_model'], 
            name=text_encoder_cfg['name'],
            api=kwargs['vlsa_api']
        )

        vl_model = load_vl_model_to_cpu(
            text_encoder_cfg,
            image_encoder_cfg,
            root=kwargs['path_clip_model'],
            api=kwargs['vlsa_api']
        )
        
        # Language-end
        self.pmt_learner_name = prompt_learner_cfg['name']
        self.prompt_encoder = get_prompt_encoder(vl_model, api=kwargs['vlsa_api'])
        if self.pmt_learner_name == 'CoOp':
            self.prompt_learner, pretrained_text_features = self._build_prompt_learner(
                prompt_learner_cfg, pretrained_prompt_learner_cfg
            )
            if pretrained_text_features is not None:
                pretrained_text_features = pretrained_text_features.detach().clone()
                self.register_buffer("pretrained_text_features", pretrained_text_features, persistent=False)
                print("[VLSA] warning: skip CoOp-based prompt learner and use pretrained text features.")
        
        elif self.pmt_learner_name == 'Adapter':
            self.prompt_adapter = self._build_prompt_adapter(prompt_learner_cfg, pretrained_prompt_learner_cfg)
        
        else:
            raise ValueError(f"{self.pmt_learner_name} is not a valid name of prompt learner.")

        # Vision-end
        if hasattr(vl_model, 'vision_model'):
            assert kwargs['vlsa_api'] == 'HF'
            self.mil_encoder = vl_model.vision_model
        elif hasattr(vl_model, 'visual'):
            assert kwargs['vlsa_api'] in ['CLIP', 'CONCH']
            self.mil_encoder = vl_model.visual
        else:
            raise ValueError(f"[{info_prefix}] `vision_model` or `visual` is not found in {vl_model}.")

        # reset query network for VLFAN (MIL encoder)
        if image_encoder_cfg['name'] == 'VLFAN':
            if image_encoder_cfg['query'] == 'Text': 
                query_text_cfg = dict()
                for k in image_encoder_cfg:
                    if k.startswith("query_text"):
                        _k = k.split("query_text_")[-1]
                        query_text_cfg[_k] = image_encoder_cfg[k]
                load_neg_pmts = image_encoder_cfg['gated_query'] if 'gated_query' in image_encoder_cfg else False
                query_text_cfg.update(dict(
                    tokenizer = self.text_tokenizer,
                    num_prompts = image_encoder_cfg['num_query'],
                    load_negative_prompts = load_neg_pmts
                ))
                visual_query_network = load_prompt_adapter(self.prompt_encoder, query_text_cfg)
                self.mil_encoder.reset_query(visual_query_network)
            else:
                visual_query_network = None
        
        self.text_encoder_cfg = text_encoder_cfg
        self.image_encoder_cfg = image_encoder_cfg
        self.prompt_learner_cfg = prompt_learner_cfg

        self.logit_scale = vl_model.logit_scale

    def _build_prompt_learner(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer, 
            text_config = self.prompt_encoder.text_config,
            token_embedding = self.prompt_encoder.token_embedding
        ))
        prompt_learner = load_prompt_learner(_prompt_learner_cfg['method'], _prompt_learner_cfg)

        # if use pretrained text prompts
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            assert pretrained_prompt_learner_cfg is not None, "Please specify `config` for `pretrained_prompt_learner`."
            prompt_learner.load_pretrained_parameters(pretrained_prompt_learner_cfg['ckpt'])
            
            # if there is no trainable parameter, pre-compute the fixed text features
            if _prompt_learner_cfg['frozen_context_embeds'] and _prompt_learner_cfg['frozen_rank_embeds']:
                with torch.no_grad():
                    pretrained_text_features = self.compute_text_features_with_coop(prompt_learner)

        return prompt_learner, pretrained_text_features

    def _build_prompt_adapter(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _pretrained_prompt_learner_cfg = pretrained_prompt_learner_cfg.copy()

        # if use CoOp-pretrained text prompts for Adapter
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            _pretrained_prompt_learner_cfg['pretrained'] = True
            _, pretrained_text_features = self._build_prompt_learner(
                _pretrained_prompt_learner_cfg, {'ckpt': _pretrained_prompt_learner_cfg['ckpt']}
            )
            assert pretrained_text_features is not None, "Found empty `pretrained_text_features`."
            pretrained_text_features = pretrained_text_features.detach().clone()

        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer,
            num_prompts = _prompt_learner_cfg['num_ranks'],
            pretrained_prompt_features = pretrained_text_features,
        ))
        prompt_adapter = load_prompt_adapter(self.prompt_encoder, _prompt_learner_cfg)

        return prompt_adapter

    def compute_text_features_with_coop(self, prompt_learner):
        sentence_embeds = prompt_learner()
        pseudo_sentence_tokens = prompt_learner.pseudo_sentence_tokens
        text_features = self.prompt_encoder(
            prompts_embedding=sentence_embeds, 
            prompts_pseudo_tokens=pseudo_sentence_tokens
        )
        return text_features

    def forward_text_only(self):
        # use pretrained_text_features if exists
        if hasattr(self, 'pretrained_text_features'):
            return self.pretrained_text_features.clone()

        if self.pmt_learner_name == 'CoOp':
            text_features = self.compute_text_features_with_coop(self.prompt_learner)

        elif self.pmt_learner_name == 'Adapter':
            text_features = self.prompt_adapter()

        else:
            text_features = None
            pass

        return text_features

    def encode_instances(self, X):
        return self.mil_encoder(X)

    def get_logit_scale(self):
        return self.logit_scale.exp()

    def forward(self, X):
        """
        X: a bag with instance feature vectors, with shape of [1, N, feat_dim].
        """
        text_features = self.forward_text_only()
        text_features = F.normalize(text_features, dim=-1) # [num_ranks, emb_dim]

        image_features = self.encode_instances(X)
        image_features = F.normalize(image_features, dim=-1) # [1, emb_dim] or [N, emb_dim]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t() # [1, num_ranks] or [N, num_ranks]

        # at zero-shot mode, mil_encoder is Identity and logits come from all instances
        if logits.shape[0] > 1:
            _, logits = logit_pooling(logits, self.image_encoder_cfg['pooling'])

        return logits, image_features, text_features
