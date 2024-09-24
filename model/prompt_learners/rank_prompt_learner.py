"""
Codes are adapted from OrdinalCLIP (https://github.com/xk-huang/OrdinalCLIP/tree/main/ordinalclip/models/prompt_leaners).
"""
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from utils.io import load_init_prompt
from .plain_prompt_learner import PlainPromptLearner


class RankPromptLearner(PlainPromptLearner):
    interpolation_functions = {
        "linear": lambda weights, num_ranks: 1.0 - weights / (num_ranks - 1),
        "inv_prop": lambda weights, _, eps=1e-5: 1.0 / (weights + eps),
        "normal": lambda weights, _: torch.exp(-weights * weights),
    }

    def __init__(
        self,
        text_config,
        tokenizer, 
        token_embedding,
        num_base_ranks: int,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        init_prompt_path: Optional[str] = None,
        init_prompt_context_idx: int = 0,
        init_prompt_rank_idx: int = 0,
        rank_specific_context: bool = False,
        interpolation_type: str = "linear",
        **kwargs,
    ) -> None:
        super(PlainPromptLearner, self).__init__()

        self.cfg_max_num_tokens  = text_config['max_num_tokens']
        self.cfg_embedding_dim   = text_config['embedding_dim']
        self.cfg_embedding_dtype = text_config['embedding_dtype']

        if kwargs:
            print(f"[RankPromptLearner] Found irrelevant kwargs: {kwargs}.")

        init_context, init_rank_names = load_init_prompt(init_prompt_path, init_prompt_context_idx, init_prompt_rank_idx)
        print(f"[RankPromptLearner] loaded initial context ({init_context}) and rank names ({init_rank_names}) from {init_prompt_path}.")

        # context embeds
        context_embeds, _num_context_tokens = self.create_context_embeds(
            tokenizer, token_embedding, num_ranks, num_context_tokens, 
            init_context, rank_specific_context
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(
            context_embeds
        )  # (num_context_tokens, embedding_dim) or (num_ranks, num_context_tokens, embedding_dim)

        # rank embeds
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_base_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            tokenizer, token_embedding, num_base_ranks, num_tokens_per_rank, 
            init_rank_names, num_context_tokens
        )
        num_tokens_per_rank = [max(_num_tokens_per_rank)] * num_ranks
        print(f"[RankPromptLearner] `num_tokens_per_rank (base -> final)`: {_num_tokens_per_rank} -> {num_tokens_per_rank}.")
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embedding_dim)
        assert (
            len(rank_embeds) == num_base_ranks
        ), f"Expected len(rank_embeds) {len(rank_embeds)} == num_base_ranks {num_base_ranks}."

        # psudo sentence tokens
        pseudo_sentence_tokens = self.create_pseudo_sentence_tokens(
            num_tokens_per_rank, num_context_tokens, num_ranks
        )  # (num_ranks, max_num_tokens)
        self.register_buffer("pseudo_sentence_tokens", pseudo_sentence_tokens, persistent=False)

        self.num_context_tokens = num_context_tokens
        self.num_tokens_per_rank = num_tokens_per_rank
        if rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Got an invalid rank_tokens_position: {rank_tokens_position}.")
        self.rank_tokens_position = rank_tokens_position
        self.num_ranks = num_ranks
        self.num_base_ranks = num_base_ranks

        sentence_embeds = self.create_sentence_embeds_template(
            tokenizer, token_embedding, num_ranks, pseudo_sentence_tokens
        )
        self.register_buffer("sentence_embeds", sentence_embeds, persistent=False)

        interpolation_weights = self.create_interpolation_weights(
            num_base_ranks, num_ranks, interpolation_type
        )
        self.register_buffer("interpolation_weights", interpolation_weights, persistent=False)

        print(f"[RankPromptLearner] Finished initializing a Rank Prompt Learner.")

    def create_interpolation_weights(self, num_base_ranks, num_ranks, interpolation_type):
        if interpolation_type not in self.interpolation_functions:
            raise ValueError(f"Got an invalide interpolation_type: {interpolation_type}.")
        interpolation_func = self.interpolation_functions[interpolation_type]

        interpolation_weights = torch.arange(num_ranks)[..., None].repeat(1, num_base_ranks).to(self.cfg_embedding_dtype)
        if num_base_ranks == 1:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, 3)[1:2].to(self.cfg_embedding_dtype)
        else:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, num_base_ranks).to(self.cfg_embedding_dtype)
        interpolation_weights = torch.abs(interpolation_weights - base_interpolation_weights[None])
        interpolation_weights = interpolation_func(interpolation_weights, num_ranks)
        interpolation_weights = interpolation_weights / interpolation_weights.sum(dim=-1, keepdim=True)
        
        return interpolation_weights

    def forward(self):
        # context_embeds: (num_ranks, num_context_tokens, embedding_dim)
        context_embeds = self.context_embeds

        # rank_embeds: (num_ranks, max_num_tokens_per_rank, embedding_dim)
        if context_embeds.dim() == 2:
            context_embeds = context_embeds[None].expand(self.num_ranks, *context_embeds.shape)
        rank_embeds = torch.sum(self.interpolation_weights[..., None, None] * self.rank_embeds[None, ...], dim=1)

        # sentence_embeds: (num_ranks, self.cfg_max_num_tokens, embedding_dim)
        sentence_embeds = self.sentence_embeds.clone()
        if self.rank_tokens_position == "tail":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [context_embeds[i], rank_embeds[i, :_num_tokens_per_rank]], dim=0
                )
        elif self.rank_tokens_position == "front":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [rank_embeds[i, :_num_tokens_per_rank], context_embeds[i]], dim=0
                )
        elif self.rank_tokens_position == "middle":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                _context_embeds = context_embeds[i]
                half_range = self.num_context_tokens // 2
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [
                        _context_embeds[:half_range],
                        rank_embeds[i, :_num_tokens_per_rank],
                        _context_embeds[half_range:],
                    ],
                    dim=0,
                )

        return sentence_embeds
