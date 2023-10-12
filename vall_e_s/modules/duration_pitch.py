import copy
import torch
import torch.nn as nn
from torch import nn, einsum, Tensor
from utils.utils import default, exists
from functools import partial
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Tuple, Union, Optional, List

class DurationPitchPredictorTrunk(nn.Module):
    def __init__(
        self,
        dim = 512,
        depth = 10,
        kernel_size = 3,
        dim_context = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        use_resnet_block = True,
        num_convs_per_resnet_block = 2,
        num_convolutions_per_block = 3,
        use_flash_attn = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        conv_klass = ConvBlock if not use_resnet_block else partial(ResnetBlock, num_convs = num_convs_per_resnet_block)

        for _ in range(depth):
            layer = nn.ModuleList([
                nn.Sequential(*[
                    conv_klass(dim, dim, kernel_size) for _ in range(num_convolutions_per_block)
                ]),
                RMSNorm(dim),
                Attention(
                    dim,
                    dim_context = dim_context,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = dropout,
                    use_flash = use_flash_attn,
                    cross_attn_include_queries = True
                )
            ])

            self.layers.append(layer)

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...'),
            nn.ReLU()
        )
    def forward(
        self,
        x,
        encoded_prompts,
        prompt_mask = None,
    ):
        for conv, norm, attn in self.layers:
            x = conv(x)
            x = attn(norm(x), encoded_prompts, mask = prompt_mask) + x

        return self.to_pred(x)

class DurationPitchPredictor(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_phoneme_tokens = None,
        tokenizer: Optional[Tokenizer] = None,
        dim_encoded_prompts = None,
        num_convolutions_per_block = 3,
        use_resnet_block = True,
        num_convs_per_resnet_block = 2,
        depth = 10,
        kernel_size = 3,
        heads = 8,
        dim_head = 64,
        dim_hidden = 512,
        dropout = 0.2,
        use_flash_attn = False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        num_phoneme_tokens = default(num_phoneme_tokens, tokenizer.vocab_size if exists(tokenizer) else None)

        dim_encoded_prompts = default(dim_encoded_prompts, dim)

        self.phoneme_token_emb = nn.Embedding(num_phoneme_tokens, dim) if exists(num_phoneme_tokens) else nn.Identity()

        self.to_pitch_pred = DurationPitchPredictorTrunk(
            dim = dim_hidden,
            depth = depth,
            kernel_size = kernel_size,
            dim_context = dim_encoded_prompts,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            use_resnet_block = use_resnet_block,
            num_convs_per_resnet_block = num_convs_per_resnet_block,
            num_convolutions_per_block = num_convolutions_per_block,
            use_flash_attn = use_flash_attn,
        )

        self.to_duration_pred = copy.deepcopy(self.to_pitch_pred)

    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        encoded_prompts,
        prompt_mask = None
    ):
        if is_bearable(x, List[str]):
            assert exists(self.tokenizer)
            x = self.tokenizer.texts_to_tensor_ids(x)

        x = self.phoneme_token_emb(x)

        duration_pred, pitch_pred = map(lambda fn: fn(x, encoded_prompts = encoded_prompts, prompt_mask = prompt_mask), (self.to_duration_pred, self.to_pitch_pred))


        return duration_pred, pitch_pred