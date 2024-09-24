"""Reimplementation of the text encoder part of different VL models.

Available VL models contains:
- CLIP (OpenAI official API at github.com/openai/CLIP)
- HuggingFace CLIP (HuggingFace API from the python package `transformers`)
- CONCH (CONCH custom API at github.com/mahmoodlab/CONCH)

These reimplementations strictly follow their corresponding official implementation.
The key difference is that our implementation directly take **prompt embeddings** as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

import model.clip as clip
import model.conch as conch


def get_prompt_encoder(vl_model, api):
    if api == 'CLIP':
        prompt_encoder = CLIPPromptEncoder(vl_model)
    elif api == 'HF':
        prompt_encoder = HFCLIPPromptEncoder(vl_model)
    elif api == 'CONCH':
        prompt_encoder = CONCHPromptEncoder(vl_model)
    else:
        raise ValueError(f"Got an invalid api ({api}).")

    return prompt_encoder


class CLIPPromptEncoder(nn.Module):
    """
    CLIP text encoder (OpenAI official API at github.com/openai/CLIP)
    """
    def __init__(self, clip_model: clip.model.CLIP):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        # Additional attributes used for downstream calls
        self.token_embedding = clip_model.token_embedding
        self.text_config = {
            'max_num_tokens': 77,
            'embedding_dim': self.token_embedding.embedding_dim,
            'embedding_dtype': self.token_embedding.weight.dtype
        }

    def generate_pseudo_tokens(self, text):
        # eot_token is the highest number in each sequence tokenized by official OpenAI CLIP
        idx_eot_token = text.argmax(dim=-1)
        pseudo_tokens = torch.zeros_like(text)
        for i in range(text.shape[0]):
            sentence_length = idx_eot_token[i] + 1
            pseudo_tokens[i, :sentence_length] = torch.arange(0, sentence_length).to(text) + 1

        return pseudo_tokens

    def forward(self, prompts_text=None, prompts_embedding=None, prompts_pseudo_tokens=None):
        """
        input:
            prompts_text: tokenized prompt texts, with shape [n_batch, length_ctx].
            prompts_embedding: the embedding of ranking prompts, with shape [n_batch, length_ctx, dim_embedding].
                Note that this input differs from the text_encoder input of official CLIP. 
            prompts_pseudo_tokens: the pseudo-tokens of ranking prompts, with shape [n_batch, length_ctx], 
                where 0 / >0 indicates that current place is a pad_token / a real word token.
        """
        if prompts_text is not None:
            assert prompts_text.shape[1] == self.text_config['max_num_tokens'], "Found invalid input of `prompts_text`."
            if prompts_pseudo_tokens is None:
                prompts_pseudo_tokens = self.generate_pseudo_tokens(prompts_text) # [batch_size, n_ctx]
            prompts_embedding = self.token_embedding(prompts_text)  # [batch_size, n_ctx, d_model]
        else:
            assert prompts_embedding is not None, "Found null `prompts_text`, please specify `prompts_embedding`."
            assert prompts_pseudo_tokens is not None, "Found null `prompts_text`, please specify `prompts_pseudo_tokens`."
            
        x = prompts_embedding.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        x = x[torch.arange(x.shape[0]), prompts_pseudo_tokens.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


class HFCLIPPromptEncoder(nn.Module):
    """
    CLIP text encoder on HuggingFace (HuggingFace API from the python package `transformers`)

    Adapted from HuggingFace API:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L658
    """
    def __init__(self, hf_clip_model: CLIPModel):
        super().__init__()
        # load text model and text projection
        hf_clip_text_model = hf_clip_model.text_model

        # Config of clip text model
        self.hf_text_config = hf_clip_text_model.config

        # Embedding
        self.embeddings = hf_clip_text_model.embeddings

        # Transformer
        self.encoder = hf_clip_text_model.encoder
        self.final_layer_norm = hf_clip_text_model.final_layer_norm
        # For `pooled_output` computation
        self.eos_token_id = self.hf_text_config.eos_token_id

        # Text projection
        self.text_projection = hf_clip_model.text_projection

        # Additional attributes used for downstream calls
        self.token_embedding = hf_clip_text_model.embeddings.token_embedding
        self.text_config = {
            'max_num_tokens': 77,
            'embedding_dim': self.token_embedding.embedding_dim,
            'embedding_dtype': self.token_embedding.weight.dtype
        }

    def generate_pseudo_tokens(self, text):
        # directly use the `eos_token_id` configured in the model
        idx_eot_token = (text.to(dtype=torch.int) == self.eos_token_id).int().argmax(dim=-1)
        pseudo_tokens = torch.zeros_like(text)
        for i in range(text.shape[0]):
            sentence_length = idx_eot_token[i] + 1
            pseudo_tokens[i, :sentence_length] = torch.arange(0, sentence_length).to(text) + 1

        return pseudo_tokens

    def forward(self, prompts_text=None, prompts_embedding=None, prompts_pseudo_tokens=None):
        """
        input:
            prompts_text: tokenized prompt texts, with shape [n_batch, length_ctx].
            prompts_embedding: the embedding of ranking prompts, with shape [n_batch, length_ctx, dim_embedding].
                Note that this input differs from the text_encoder input of HuggingFace CLIP. 
            prompts_pseudo_tokens: the pseudo-tokens of ranking prompts, with shape [n_batch, length_ctx],
                where 0 / >0 indicates that current place is a pad_token / a real word token.
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = self.hf_text_config.output_attentions
        output_hidden_states = (self.hf_text_config.output_hidden_states)
        return_dict = self.hf_text_config.use_return_dict

        if prompts_text is not None:
            # `prompts_text` may have less than 77 tokens in HF CLIP
            input_shape = prompts_text.size()[:2] # [n_batch, length_ctx]
            if prompts_pseudo_tokens is None:
                prompts_pseudo_tokens = self.generate_pseudo_tokens(prompts_text) # [batch_size, n_ctx]
            # `input_ids` as the input to obtain token embeddings
            hidden_states = self.embeddings(input_ids=prompts_text)  # [n_batch, length_ctx, dim_embedding]
        else:
            assert prompts_embedding is not None, "Found null `prompts_text`, please specify `prompts_embedding`."
            assert prompts_pseudo_tokens is not None, "Found null `prompts_text`, please specify `prompts_pseudo_tokens`."
            input_shape = prompts_embedding.size()[:2] # [n_batch, length_ctx]
            # prompt embeddings (rather than `input_ids`) as the input to obtain token embeddings
            hidden_states = self.embeddings(inputs_embeds=prompts_embedding) # [n_batch, length_ctx, dim_embedding]

        # Get attention mask
        attention_mask = (prompts_pseudo_tokens > 0).to(torch.long) 

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            attention_mask = attention_mask.to(device=hidden_states.device)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value, indicated by prompts_pseudo_tokens
            prompts_pseudo_tokens.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        text_features = self.text_projection(pooled_output)

        return text_features


class CONCHPromptEncoder(nn.Module):
    """
    CONCH (CONCH custom API at github.com/mahmoodlab/CONCH)
    """
    def __init__(self, coca_model: conch.coca_model.CoCa):
        super().__init__()
        # load text model and text projection
        coca_text_model = coca_model.text

        # Config of clip text model
        self.pad_id = coca_text_model.pad_id
        assert self.pad_id == 0, "Assume pad_id = 0 in CONCH to encode prompts as expected."
        self.heads = coca_text_model.heads

        # Embedding
        self.positional_embedding = coca_text_model.positional_embedding

        self.attn_mask = coca_text_model.attn_mask
        # Transformer
        self.transformer = coca_text_model.transformer
        self.ln_final = coca_text_model.ln_final

        self.cls_emb = coca_text_model.cls_emb

        # Text projection
        self.text_projection = coca_text_model.text_projection

        # Additional attributes used for downstream calls
        self.token_embedding = coca_text_model.token_embedding
        self.text_config = {
            'max_num_tokens': 127,
            'embedding_dim': self.token_embedding.embedding_dim,
            'embedding_dtype': self.token_embedding.weight.dtype
        }

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def generate_pseudo_tokens(self, text):
        # eot_token is ahead of the first pad_token (refer to CONCH's tokenizer)
        idx_eot_token = (text.to(dtype=torch.int) == self.pad_id).int().argmax(dim=-1) - 1
        pseudo_tokens = torch.zeros_like(text)
        for i in range(text.shape[0]):
            sentence_length = idx_eot_token[i] + 1
            pseudo_tokens[i, :sentence_length] = torch.arange(0, sentence_length).to(text) + 1

        return pseudo_tokens

    def forward(self, prompts_text=None, prompts_embedding=None, prompts_pseudo_tokens=None):
        """
        Adapted from CONCH API:
            https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/transformer.py#L418

        input:
            prompts_text: tokenized prompt texts, with shape [n_batch, length_ctx].
            prompts_embedding: the embedding of ranking prompts, with shape [n_batch, length_ctx, dim_embedding].
                Note that this input differs from the text_encoder input of HuggingFace CLIP. 
            prompts_pseudo_tokens: the pseudo-tokens of ranking prompts, with shape [n_batch, length_ctx],
                where 0 / >0 indicates that current place is a pad_token / a real word token.
        """
        cast_dtype = self.transformer.get_cast_dtype()
        if prompts_text is not None:
            device = prompts_text.device
            assert prompts_text.shape[1] == self.text_config['max_num_tokens'] + 1, "Found invalid input of `prompts_text`."
            prompts_text = prompts_text[:, :-1] if self.cls_emb is not None else prompts_text # make space for CLS token
            seq_len = prompts_text.shape[1] # max_length - 1 = 128 - 1
            if prompts_pseudo_tokens is None:
                prompts_pseudo_tokens = self.generate_pseudo_tokens(prompts_text) # [batch_size, n_ctx]
            x = self.token_embedding(prompts_text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        else:
            assert prompts_embedding is not None, "Found null `prompts_text`, please specify `prompts_embedding`."
            assert prompts_pseudo_tokens is not None, "Found null `prompts_text`, please specify `prompts_pseudo_tokens`."
            device = prompts_embedding.device
            seq_len = prompts_embedding.shape[1] # max_length - 1 = 128 - 1
            x = prompts_embedding.to(cast_dtype)  # [batch_size, n_ctx, d_model]

        assert seq_len == self.text_config['max_num_tokens']

        prompts_pseudo_tokens = prompts_pseudo_tokens.to(device)
        attn_mask = self.attn_mask.to(device)
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(prompts_pseudo_tokens, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), prompts_pseudo_tokens.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        return pooled
