"""
TimeSformer in PyTorch, with CLIP ViT initialisation

A PyTorch implement of TimeSformer as described in
'Is Space-Time Attention All You Need for Video Understanding?' - https://arxiv.org/pdf/2102.05095.pdf

Acknowledgments:
- This code builds on Ross Wightman's vision_transformer code in pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

- It is also inspired by lucidrains timesformer implementation:
https://github.com/lucidrains/TimeSformer-pytorch

- Hacked together by Max Bain
https://github.com/m-bain/video-transformers/blob/main/video-transformers/timesformer.py

Further hacking to adapt to CLIP's VisualTransformer by James Thewlis

CLIP from https://github.com/openai/CLIP
Learning Transferable Visual Models From Natural Language Supervision https://arxiv.org/abs/2103.00020
"""

from collections import OrderedDict

import clip
import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn.init import constant_, xavier_uniform_

__all__ = ["make_timesformer_clip_vit"]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def attn(q, k, v):
    sim = einsum("b i d, b j d -> b i j", q, k)
    attn = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", attn, v)
    return out


def multi_head_attention_space(
    x,
    dim,
    nheads,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    nframes,
):
    head_dim = dim // nheads
    assert head_dim * nheads == dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    q, k, v = (x @ in_proj_weight.t() + in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q_heads = rearrange(q, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)
    k_heads = rearrange(k, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)
    v_heads = rearrange(v, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)

    # Extract cls tokens
    q_cls, q_rest = q_heads[:, 0:1, :], q_heads[:, 1:, :]
    k_cls, k_rest = k_heads[:, 0:1, :], k_heads[:, 1:, :]
    v_cls, v_rest = v_heads[:, 0:1, :], v_heads[:, 1:, :]

    cls_out = attn(q_cls, k_heads, v_heads)

    # Rearrange frames into the batch dimension
    q_space = rearrange(
        q_rest,
        "(b_nheads) (frames patches) dim -> (b_nheads frames) patches dim",
        frames=nframes,
    )
    k_space = rearrange(
        k_rest,
        "(b_nheads) (frames patches) dim -> (b_nheads frames) patches dim",
        frames=nframes,
    )
    v_space = rearrange(
        v_rest,
        "(b_nheads) (frames patches) dim -> (b_nheads frames) patches dim",
        frames=nframes,
    )

    # Replicate k_cls, v_cls across time
    r = q_space.shape[0] // k_cls.shape[0]
    k_cls = repeat(k_cls, "b_nheads 1 dim -> (b_nheads r) 1 dim", r=r)
    v_cls = repeat(v_cls, "b_nheads 1 dim -> (b_nheads r) 1 dim", r=r)

    k_space_cls = torch.cat((k_cls, k_space), dim=1)
    v_space_cls = torch.cat((v_cls, v_space), dim=1)

    attn_output = attn(q_space, k_space_cls, v_space_cls)

    # Rearrange frame dimension back
    attn_output = rearrange(
        attn_output,
        "(b_nheads frames) patches dim -> (b_nheads) (frames patches) dim",
        frames=nframes,
    )

    # add back the cls token
    attn_output = torch.cat((cls_out, attn_output), dim=1)

    # merge back the heads
    attn_output = rearrange(
        attn_output, "(b nheads) grid dim -> b grid (nheads dim)", nheads=nheads
    )

    out = attn_output @ out_proj_weight.t() + out_proj_bias

    return out


def multi_head_attention_time(
    x,
    dim,
    nheads,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    nframes,
):
    head_dim = dim // nheads
    assert head_dim * nheads == dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    q, k, v = (x @ in_proj_weight.t() + in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q_heads = rearrange(q, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)
    k_heads = rearrange(k, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)
    v_heads = rearrange(v, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)

    b_nheads = q_heads.shape[0]

    # Extract cls tokens
    q_cls, q_rest = q_heads[:, 0:1, :], q_heads[:, 1:, :]
    k_cls, k_rest = k_heads[:, 0:1, :], k_heads[:, 1:, :]
    v_cls, v_rest = v_heads[:, 0:1, :], v_heads[:, 1:, :]

    cls_out = attn(q_cls, k_heads, v_heads)

    # Rearrange spatial patches into the batch dimension
    q_time = rearrange(
        q_rest,
        "(b_nheads) (frames patches) dim -> (b_nheads patches) frames dim",
        frames=nframes,
    )
    k_time = rearrange(
        k_rest,
        "(b_nheads) (frames patches) dim -> (b_nheads patches) frames dim",
        frames=nframes,
    )
    v_time = rearrange(
        v_rest,
        "(b_nheads) (frames patches) dim -> (b_nheads patches) frames dim",
        frames=nframes,
    )

    # Replicate k_cls, v_cls across space
    r = q_time.shape[0] // k_cls.shape[0]
    k_cls = repeat(k_cls, "b_nheads 1 dim -> (b_nheads r) 1 dim", r=r)
    v_cls = repeat(v_cls, "b_nheads 1 dim -> (b_nheads r) 1 dim", r=r)

    k_time_cls = torch.cat((k_cls, k_time), dim=1)
    v_time_cls = torch.cat((v_cls, v_time), dim=1)

    attn_output = attn(q_time, k_time_cls, v_time_cls)

    # Rearrange patches dimension back
    attn_output = rearrange(
        attn_output,
        "(b_nheads patches) frames dim -> (b_nheads) (frames patches) dim",
        frames=nframes,
        b_nheads=b_nheads,
    )

    # add back the cls token
    attn_output = torch.cat((cls_out, attn_output), dim=1)

    # merge back the heads
    attn_output = rearrange(
        attn_output, "(b nheads) grid dim -> b grid (nheads dim)", nheads=nheads
    )

    out = attn_output @ out_proj_weight.t() + out_proj_bias

    return out


class SpaceAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, nframes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.nframes = nframes

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def forward(self, x):
        return multi_head_attention_space(
            x,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj.weight,
            self.out_proj.bias,
            self.nframes,
        )

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)


class TimeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, nframes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.nframes = nframes

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def forward(self, x):
        return multi_head_attention_time(
            x,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj.weight,
            self.out_proj.bias,
            self.nframes,
        )

    def _reset_parameters(self):
        # Initialise time weights as 0
        constant_(self.in_proj_weight, 0.0)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.weight, 1.0)
        constant_(self.out_proj.bias, 0.0)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        nframes: int = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.nframes = nframes

        self.attn = SpaceAttention(d_model, n_head, nframes)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.timeattn = TimeAttention(d_model, n_head, nframes)
        self.ln_time = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        assert self.attn_mask is None
        return self.attn(x)

    def forward(self, x: torch.Tensor):
        time_x = x + self.timeattn(self.ln_time(x))
        space_x = time_x + self.attention(self.ln_1(time_x))

        x = space_x

        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        nframes: int = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask, nframes=nframes)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        nframes: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.nframes = nframes
        self.width = width

        self.patches_per_frame = (input_resolution // patch_size) ** 2
        self.num_patches = nframes * self.patches_per_frame

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )

        self.temporal_embed = nn.Parameter(torch.zeros(nframes, width))

        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, nframes=nframes)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        B, F, C, H, W = x.shape

        x = x.view(-1, C, H, W)
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(B, -1, self.width)  # shape = [B, nframes * grid ** 2, width]

        # Prepend the CLS token
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, nframes * grid ** 2 + 1, width]

        # We must tile the positional embedding over time
        # First pull out the CLS element of the position embedding which
        # needn't be tiled
        cls_pos_embed = self.positional_embedding[0:1, :]
        # Now repeat the other elements over time
        tile_pos_embed = self.positional_embedding[1:, :].repeat(F, 1)

        # In the TimeSFormer we also have a temporal embedding
        # Which must be tiled over space
        tile_temporal_embed = self.temporal_embed.repeat_interleave(
            self.patches_per_frame, dim=0
        )

        # Now combine them all
        total_pos_embed = torch.cat(
            [cls_pos_embed, tile_pos_embed + tile_temporal_embed], dim=0
        )

        x = x + total_pos_embed.to(x.dtype)
        x = self.ln_pre(x)

        # shape [batch, grid', width]  (where grid' = nframes * grid ** 2 + 1)

        # In the CLIP code there is a permute here since pytorch
        # MultiHeadAttention wants [token,batch,dim] shape,
        # but now we have an attention impl with batch in first dim
        x = self.transformer(x)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


def make_timesformer_clip_vit(nframes):
    clip_input_resolution = 224
    clip_patch_size = 32
    clip_width = 768
    clip_layers = 12
    clip_heads = 12
    clip_output_dim = 512

    timesfm = VisualTransformer(
        input_resolution=clip_input_resolution,
        patch_size=clip_patch_size,
        width=clip_width,
        layers=clip_layers,
        heads=clip_heads,
        output_dim=clip_output_dim,
        nframes=nframes,
    )

    model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    clip_vit_state = model.visual.state_dict()

    missing, unexpected = timesfm.load_state_dict(clip_vit_state, strict=False)
    assert len(unexpected) == 0
    # The only missing keys should be the new time-related ones
    assert all(["time" in x or "temporal" in x for x in missing])

    return timesfm


if __name__ == "__main__":
    import pylab

    nframes = 2
    timesfm = make_timesformer_clip_vit(nframes)
    timesfm.eval()

    model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

    torch.manual_seed(123)

    image = torch.randn(2, 1, 3, 224, 224)
    # from PIL import Image
    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).unsqueeze(0)

    # Repeat an image from nframes
    input = torch.cat([image] * nframes, dim=1)

    vit_out = model.visual(input[:, 0, :, :, :])
    print()
    timesformer_out = timesfm(input)

    print(vit_out.min(), vit_out.max())
    print(timesformer_out.min(), timesformer_out.max())

    pylab.plot(vit_out.detach().reshape(-1).numpy())
    pylab.plot(timesformer_out.detach().reshape(-1).numpy())
    pylab.show(block=True)
