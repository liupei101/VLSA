"""
Codes are adapted from OrdinalCLIP (https://github.com/xk-huang/OrdinalCLIP/tree/main/ordinalclip/models/prompt_leaners).
"""
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from utils.io import load_init_prompt


class PlainPromptLearner(nn.Module):
    rank_tokens_position_candidates = {"tail", "middle", "front"}

    def __init__(
        self,
        text_config,
        tokenizer, 
        token_embedding,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        init_prompt_path: Optional[str] = None,
        init_prompt_context_idx: int = 0,
        init_prompt_rank_idx: int = 0,
        rank_specific_context: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cfg_max_num_tokens  = text_config['max_num_tokens']
        self.cfg_embedding_dim   = text_config['embedding_dim']
        self.cfg_embedding_dtype = text_config['embedding_dtype']

        if kwargs:
            print(f"[PlainPromptLearner] Found irrelevant kwargs: {kwargs}")

        init_context, init_rank_names = load_init_prompt(init_prompt_path, init_prompt_context_idx, init_prompt_rank_idx)
        print(f"[PlainPromptLearner] loaded initial context ({init_context}) and rank names ({init_rank_names}) from {init_prompt_path}.")

        # context embeds
        context_embeds, _num_context_tokens = self.create_context_embeds(
            tokenizer, token_embedding, num_ranks, num_context_tokens, 
            init_context, rank_specific_context
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(
            context_embeds
        )  # (num_context_tokens, embeds_dim) or (num_ranks, num_context_tokens, embeds_dim)

        # rank embeds
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            tokenizer, token_embedding, num_ranks, num_tokens_per_rank, 
            init_rank_names, num_context_tokens
        )
        num_tokens_per_rank = _num_tokens_per_rank
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert len(rank_embeds) == num_ranks, f"Expected len(rank_embeds) {len(rank_embeds)} == num_ranks {num_ranks}."

        # psudo sentence tokens
        pseudo_sentence_tokens = self.create_pseudo_sentence_tokens(
            num_tokens_per_rank, num_context_tokens, num_ranks
        )  # (num_ranks, cfg_max_num_tokens)
        self.register_buffer("pseudo_sentence_tokens", pseudo_sentence_tokens, persistent=False)

        self.num_context_tokens = num_context_tokens
        self.num_tokens_per_rank = num_tokens_per_rank
        if rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"[PlainPromptLearner] Got an invalid rank_tokens_position: {rank_tokens_position}")
        self.rank_tokens_positon = rank_tokens_position
        self.num_ranks = num_ranks

        sentence_embeds = self.create_sentence_embeds_template(
            tokenizer, token_embedding, num_ranks, pseudo_sentence_tokens
        )
        self.register_buffer("sentence_embeds", sentence_embeds, persistent=False)

    def forward(self):
        # context_embeds: (num_ranks, num_context_tokens, embeds_dim)
        # rank_embeds: (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        context_embeds = self.context_embeds
        if context_embeds.dim() == 2:
            context_embeds = context_embeds[None].expand(self.num_ranks, *context_embeds.shape)

        # sentence_embeds: (num_ranks, self.cfg_max_num_tokens, embeddings_dim)
        sentence_embeds = self.sentence_embeds.clone()
        if self.rank_tokens_positon == "tail":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [context_embeds[i], self.rank_embeds[i, :_num_tokens_per_rank]], dim=0
                )
        elif self.rank_tokens_positon == "front":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [self.rank_embeds[i, :_num_tokens_per_rank], context_embeds[i]], dim=0
                )
        elif self.rank_tokens_positon == "middle":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                _context_embeds = context_embeds[i]
                half_range = self.num_context_tokens // 2
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [
                        _context_embeds[:half_range],
                        self.rank_embeds[i, :_num_tokens_per_rank],
                        _context_embeds[half_range:],
                    ],
                    dim=0,
                )

        return sentence_embeds

    def load_pretrained_parameters(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        context_embeds = ckpt['model']['prompt_learner.context_embeds']
        assert context_embeds.shape == self.context_embeds.shape
        self.context_embeds = nn.Parameter(context_embeds)

        rank_embeds = ckpt['model']['prompt_learner.rank_embeds']
        assert rank_embeds.shape == self.rank_embeds.shape
        self.rank_embeds = nn.Parameter(rank_embeds)

        print(f"[Prompt Learner] overrided `context_embeds` and `rank_embeds` with pretrained ckpt ({ckpt_path}).")

    def create_sentence_embeds_template(self, tokenizer, token_embedding, num_ranks, pseudo_sentence_tokens):
        with torch.no_grad():
            # it should return a tensor with [4, ] (vector tensor)
            token_ids, num_tokens = tokenizer("X.", return_raw_tokens=False, return_num_tokens=True)
            assert num_tokens == 2, "Expected 2 text tokens for the text `X.`."
            assert token_ids[0] == tokenizer.bos_token_id and token_ids[3] == tokenizer.eos_token_id
            pad_embed = token_embedding(torch.LongTensor([tokenizer.pad_token_id]))[0]
            sot_embed = token_embedding(token_ids[[0]])[0]
            eot_embed = token_embedding(token_ids[[3]])[0]
            full_stop_embed = token_embedding(token_ids[[2]])[0]

        # use pad_token_embedding as placeholder, following the default tokenizer behavior.
        sentence_embeds = pad_embed[None, None].repeat(
            num_ranks, self.cfg_max_num_tokens, 1
        )  # not the same pad_embed!

        argmax_index = pseudo_sentence_tokens.argmax(dim=-1)
        rank_index = torch.arange(num_ranks)

        sentence_embeds[rank_index, 0] = sot_embed
        sentence_embeds[rank_index, argmax_index] = eot_embed
        sentence_embeds[rank_index, argmax_index - 1] = full_stop_embed

        return sentence_embeds

    def create_pseudo_sentence_tokens(self, num_tokens_per_rank, num_context_tokens, num_ranks):
        pseudo_sentence_tokens = torch.zeros(num_ranks, self.cfg_max_num_tokens, dtype=torch.long)

        if isinstance(num_tokens_per_rank, List):
            assert num_ranks == len(num_tokens_per_rank)
            for i, _num_tokens_per_rank in enumerate(num_tokens_per_rank):
                # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
                sentence_length = 1 + num_context_tokens + _num_tokens_per_rank + 1 + 1
                pseudo_sentence_tokens[i, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long) + 1
        else:
            # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
            sentence_length = 1 + num_context_tokens + num_tokens_per_rank + 1 + 1
            pseudo_sentence_tokens[:, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long) + 1
        
        return pseudo_sentence_tokens

    def create_rank_embeds(
        self, tokenizer, token_embedding, num_ranks, num_tokens_per_rank, 
        init_rank_names, num_context_tokens
    ):
        if init_rank_names is not None:
            num_can = len(init_rank_names)

            if num_can > num_ranks:
                rank_names_to_select = np.linspace(0, num_can - 1, num_ranks).astype(np.int32)
                new_rank_names = [init_rank_names[idx] for idx in rank_names_to_select]
                print(f"[PlainPromptLearner] selected {len(new_rank_names)} rank names from {num_can} candidates.")
                rank_names = new_rank_names
            elif len(init_rank_names) < num_ranks:
                num_sec = len(init_rank_names)
                len_sec = num_ranks // num_sec
                new_rank_names = [init_rank_names[min(i // len_sec, num_sec - 1)] for i in range(num_ranks)]
                rank_names = new_rank_names
            else:
                rank_names = init_rank_names
            print(f"[PlainPromptLearner] final rank names: {rank_names}.")

            # return a tensor of [num_ranks, num_max_tokens], without <sot> and <eot>
            rank_tokens, _num_tokens_per_rank = tokenizer(
                rank_names, return_raw_tokens=True, return_num_tokens=True
            )
            _num_tokens_per_rank = _num_tokens_per_rank.tolist()
            
            print(f"[PlainPromptLearner] num_tokens_per_rank: {num_tokens_per_rank} -> {_num_tokens_per_rank}.")
            num_tokens_per_rank = _num_tokens_per_rank

            max_num_tokens_per_rank = max(num_tokens_per_rank)
            # 3 is <eot>, <sot>, and <full_stop>
            if max_num_tokens_per_rank > self.cfg_max_num_tokens - num_context_tokens - 3:
                raise ValueError(f"The rank name is too long: {rank_names[np.argmax(num_tokens_per_rank)]}.")

            with torch.no_grad():
                # [num_ranks, max_num_token, cfg_embedding_dim]
                rank_embeds = token_embedding(rank_tokens)
            assert rank_embeds.shape[1] == max_num_tokens_per_rank

        else:
            print(f"[PlainPromptLearner] num rank: {num_ranks}.")
            print(f"[PlainPromptLearner] num_tokens_per_rank: {num_tokens_per_rank}.")
            if isinstance(num_tokens_per_rank, List):
                max_num_tokens_per_rank = np.max(num_tokens_per_rank)
            else:
                max_num_tokens_per_rank = num_tokens_per_rank
            if self.cfg_max_num_tokens < num_context_tokens + max_num_tokens_per_rank + 3:
                raise ValueError(f"The value of `max_num_tokens_per_rank` ({max_num_tokens_per_rank}) is too large.")
            rank_embeds = torch.empty(
                (num_ranks, max_num_tokens_per_rank, self.cfg_embedding_dim), 
                dtype=self.cfg_embedding_dtype
            )
            nn.init.normal_(rank_embeds, std=0.02)

        return (rank_embeds, num_tokens_per_rank)

    def create_context_embeds(
        self,
        tokenizer,
        token_embedding,
        num_ranks: int,
        num_context_tokens: int,
        init_context: Optional[str],
        rank_specific_context: bool
    ):
        # context embeddings
        print("[PlainPromptLearner] init context token...")
        if init_context is not None:
            init_context = init_context.replace("_", " ")
            print(f"[PlainPromptLearner] init context: {init_context}")

            # return a tensor of [num_token, ], without <sot> and <eot>
            prompt_tokens, _num_context_tokens = tokenizer(
                init_context, return_raw_tokens=True, return_num_tokens=True
            )
            print(f"[PlainPromptLearner] num_context_tokens: {num_context_tokens} -> {_num_context_tokens}")
            num_context_tokens = _num_context_tokens

            with torch.no_grad():
                # [num_token, cfg_embedding_dim]
                context_embeds = token_embedding(prompt_tokens)
            assert num_context_tokens == len(context_embeds)
            
            print(f"[PlainPromptLearner] rank_specific_context: {rank_specific_context}")
            if rank_specific_context is True:
                context_embeds = context_embeds[None].repeat(num_ranks, 1, 1)
        else:
            print(f"[PlainPromptLearner] num context tokens: {num_context_tokens}")
            print(f"[PlainPromptLearner] rank_specific_context: {rank_specific_context}")

            if rank_specific_context is True:
                context_embeds = torch.empty(
                    (num_ranks, num_context_tokens, self.cfg_embedding_dim), 
                    dtype=self.cfg_embedding_dtype
                )
            else:
                context_embeds = torch.empty(
                    (num_context_tokens, self.cfg_embedding_dim), 
                    dtype=self.cfg_embedding_dtype
                )
            nn.init.normal_(context_embeds, std=0.02)

        return context_embeds, num_context_tokens
