from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from utils.io import load_init_prompt, load_init_text
from model.layers import Adapter


class PromptAdapter(nn.Module):
    """
    CLIPAdapter implementation.

    Please refer to Gao et al., CLIP-Adapter: Better Vision-Language Models with Feature Adapters, IJCV, 2023.
                 and Yu et al., Task Residual for Tuning Vision-Language Models, CVPR, 2023.
    """
    def __init__(
        self,
        prompt_encoder,
        tokenizer = None, 
        method: str = 'default',
        load_path: Optional[str] = None,
        load_idx: Union[int,str] = 0,
        load_negative_prompts: bool = False,
        load_negative_idx: str = 'prompt_normal_tissue',
        num_prompts: int = 4,
        init_prompt_path: Optional[str] = None,
        init_prompt_context_idx: int = 0,
        init_prompt_rank_idx: int = 0,
        pretrained_prompt_features = None,
        dim_reduction: int = 4,
        keep_ratio: float = 0.8,
        res_ratio: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()

        assert method in ['default', 'FC', 'Adapter', 'TaskRes']
        self.method = method

        if kwargs:
            print(f"[PromptAdapter] Found irrelevant kwargs: {kwargs}")

        if pretrained_prompt_features is None:
            if init_prompt_path is not None:
                _, init_rank_prompts = load_init_prompt(init_prompt_path, init_prompt_context_idx, init_prompt_rank_idx, replace=True)
                assert len(init_rank_prompts) == num_prompts, f"Expected {num_prompts} initial prompts, but got {len(init_rank_prompts)}."
                print(f"[PromptAdapter] loaded initial prompts from {init_prompt_path}: {init_rank_prompts}.")
                init_texts = init_rank_prompts

            elif load_path is not None:
                init_texts = load_init_text(load_path, key=str(load_idx))
                assert len(init_texts) == num_prompts, f"Expected {num_prompts} initial texts, but got {len(init_texts)}."
                print(f"[PromptAdapter] loaded initial texts from {load_path}: {init_texts}.")

            else:
                raise RuntimeError("Please specify `init_prompt_path` or `load_path` to load initial prompts or texts.")

            # [num_prompts, ctx_length]
            token_ids = tokenizer(init_texts, return_raw_tokens=False, return_num_tokens=False)
            with torch.no_grad():
                prompt_features = prompt_encoder(prompts_text=token_ids) # [num_prompts, dim_embedding]

        else:
            assert len(pretrained_prompt_features) == num_prompts, f"Expected {num_prompts} initial texts, but got {len(pretrained_prompt_features)}."
            prompt_features = pretrained_prompt_features
            print(f"[PromptAdapter] use given pretrained prompt features.")

        self.register_buffer("prompt_features", prompt_features, persistent=False)

        if load_negative_prompts:
            assert load_path is not None, "Found null `load_path`."
            neg_texts = load_init_text(load_path, key=load_negative_idx)
            print(f"[PromptAdapter] loaded {len(neg_texts)} negative texts from {load_path}: {neg_texts}.")

            neg_token_ids = tokenizer(neg_texts, return_raw_tokens=False, return_num_tokens=False)
            with torch.no_grad():
                neg_prompt_features = prompt_encoder(prompts_text=neg_token_ids)
                neg_prompt_features = neg_prompt_features.mean(0, keepdims=True) # [1, dim_embedding]

            self.register_buffer("neg_prompt_features", neg_prompt_features, persistent=False)

        dim_embedding = prompt_features.shape[-1]
        dtype_embedding = prompt_features.dtype
        
        # Adapter for text-end
        if self.method == 'Adapter':
            self.adapter = Adapter(dim_embedding, dim_reduction).to(dtype_embedding)
            assert keep_ratio >= 0 and keep_ratio <= 1.0
            self.keep_ratio = keep_ratio
        
        elif self.method == 'TaskRes':
            self.residual_features = nn.Parameter(torch.randn(num_prompts, dim_embedding))
            if load_negative_prompts:
                self.neg_residual_features = nn.Parameter(torch.randn(1, dim_embedding))  
                print("[PromptAdapter] added one residual feature for negative prompt learning.")
            else:
                self.neg_residual_features = None
            self.res_ratio = res_ratio

        elif self.method == 'FC':
            self.fc = nn.Sequential(
                nn.Linear(dim_embedding, dim_embedding, bias=False),
                nn.Dropout(0.25),
            )

        print(f"[PromptAdapter] initialized a PromptAdapter with method ({self.method}).")

    def get_raw_prompt_features(self):
        raw_features = self.prompt_features.clone()
        if hasattr(self, 'neg_prompt_features'):
            neg_prompt_features = self.neg_prompt_features.clone()
            raw_features = torch.cat([raw_features, neg_prompt_features], dim=0) # [P + 1, d]

        return raw_features

    def forward(self):
        prompt_features = self.prompt_features.clone()

        if self.method == 'Adapter':
            adapted_features = self.adapter(prompt_features)
            text_features = (1 - self.keep_ratio) * adapted_features + self.keep_ratio * prompt_features

        elif self.method == 'TaskRes':
            text_features = self.res_ratio * self.residual_features + prompt_features # [P, d]
            if hasattr(self, 'neg_prompt_features'):
                neg_prompt_features = self.neg_prompt_features.clone()
                if self.neg_residual_features is not None:
                    neg_text_features = self.res_ratio * self.neg_residual_features + neg_prompt_features # [1, d]
                else:
                    neg_text_features = neg_prompt_features
                # append the negative prompts
                text_features = torch.cat([text_features, neg_text_features], dim=0) # [P + 1, d]

        elif self.method == 'FC':
            if hasattr(self, 'neg_prompt_features'):
                neg_prompt_features = self.neg_prompt_features.clone()
                # append the negative prompts
                _prompt_features = torch.cat([prompt_features, neg_prompt_features], dim=0) # [P + 1, d]
            else:
                _prompt_features = prompt_features
            
            text_features = self.fc(_prompt_features)

        else:
            text_features = prompt_features
        
        return text_features
