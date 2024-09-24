##################################################
# Utility functions for Vision-Language models
# These functions support 
# - OpenAI CLIP
# - HuggingFace transformers
# - mahmoodlab CONCH
##################################################
from typing import Union, List
import os.path as osp
import torch
from transformers import CLIPModel
from transformers import AutoTokenizer

import model.deepmil as mil_encoders
import model.clip as clip
import model.conch as conch


class Tokenizer(object):
    def __init__(self, root=None, name=None, api='CLIP'):
        super().__init__()
        self.api = api
        self.pad_token_id = 0
        self.bos_token_id = 49406
        self.eos_token_id = 49407

        if api == 'CLIP':
            print("[Tokenizer] CLIP Tokenizer runs with built-in functions.")
            self.tokenizer = None
        elif api == 'HF':
            path = osp.join(root, name)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            print(f"[Tokenizer] Loaded HF Tokenizer from {path}.")
        elif api == 'CONCH':
            self.tokenizer = conch.get_tokenizer()
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            print("[Tokenizer] CONCH Tokenizer runs with built-in functions.")
        else:
            raise ValueError(f"Got an invalid api ({api}).")

    def __call__(
        self, 
        text: Union[str, List[str]],
        return_raw_tokens=True, 
        return_num_tokens=True
    ):
        if isinstance(text, str):
            _text = [text]
        else:
            _text = text

        if self.api == 'CLIP':
            token_ids = clip.tokenize(_text)

        elif self.api == 'HF':
            res = self.tokenizer(_text, padding=True, return_tensors='pt')
            token_ids = res['input_ids']

        elif self.api == 'CONCH':
            token_ids = conch.tokenize(self.tokenizer, _text)

        # <sot> and <eot> are not included in total token numbers
        token_cnt = (token_ids == self.eos_token_id).int().argmax(dim=-1) - 1

        if return_raw_tokens:
            _max_token_cnt = token_cnt.max()
            token_ids = token_ids[:, 1:(_max_token_cnt+1)]

        if isinstance(text, str):
            token_ids = token_ids[0]
            token_cnt = token_cnt[0]

        if return_num_tokens:
            return token_ids, token_cnt

        return token_ids

def load_vl_model_to_cpu(
    text_encoder_cfg,
    image_encoder_cfg,
    root,
    api,
    info_prefix='VL model loading'
):
    print(f"[{info_prefix}] Building VL model with {api} API...")

    # text backbone
    text_backbone_name = text_encoder_cfg['name']
    print(f"[{info_prefix}] Text backbone : {text_backbone_name}.")

    if api == 'CLIP':
        url = clip._MODELS[text_backbone_name]
        full_path = osp.join(root, "openai_clip")
        model_path = clip._download(url, root=full_path)
        print(f"[{info_prefix}] saved OpenAI CLIP-{text_backbone_name} to {full_path}.")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())

    elif api == 'HF':
        # text_backbone_name == 'openai/xxxx' / 'vinid/plip'
        local_model_path = osp.join(root, text_backbone_name)
        model = CLIPModel.from_pretrained(local_model_path)

    elif api == 'CONCH':
        # text_backbone_name == 'mahmoodlab/conch'
        local_model_path = osp.join(root, text_backbone_name, "pytorch_model.bin")
        model = conch.create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=local_model_path,
            return_transform=False,
        )

    else:
        raise ValueError(f"Got an invalid api ({api}).")

    # vision backbone (a MIL encoder)
    image_encoder_name = image_encoder_cfg['name']
    mil_backbone_name = image_encoder_name
    print(f"[{info_prefix}] MIL backbone: {mil_backbone_name}")

    MIL_MODEL = getattr(mil_encoders, mil_backbone_name, None)
    if MIL_MODEL is None:
        raise ValueError(f"[{info_prefix}] Got an invalid MIL encoder name: {mil_backbone_name}.")        
    
    mil_model = MIL_MODEL(**image_encoder_cfg)
    print(f"[{info_prefix}] Try {mil_backbone_name} with arguments: {image_encoder_cfg}.")

    if api == 'CLIP':
        model.visual = mil_model
    elif api == 'HF':
        model.vision_model = mil_model
    elif api == 'CONCH':
        model.visual = mil_model
    else:
        raise ValueError(f"[{info_prefix}] Got an invalid api ({api}).")

    return model
