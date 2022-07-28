"""
TimeSformer in PyTorch, with CLIP ViT initialisation

This version more closely follows the official TimeSformer
https://github.com/facebookresearch/TimeSformer/blob/main/lib/models/vit.py

CLIP from https://github.com/openai/CLIP 
Learning Transferable Visual Models From Natural Language Supervision https://arxiv.org/abs/2103.00020
"""

from collections import OrderedDict

import clip
import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn.init import constant_, trunc_normal_, xavier_uniform_


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


def multi_head_attention(
    x, dim, nheads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
):

    head_dim = dim // nheads
    assert head_dim * nheads == dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    q, k, v = (x @ in_proj_weight.t() + in_proj_bias).chunk(3, dim=-1)

    q = q * scaling

    q_heads = rearrange(q, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)
    k_heads = rearrange(k, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)
    v_heads = rearrange(v, "b grid (nheads dim) -> (b nheads) grid dim", nheads=nheads)

    attn_output = attn(q_heads, k_heads, v_heads)

    # merge back the heads
    attn_output = rearrange(
        attn_output, "(b nheads) grid dim -> b grid (nheads dim)", nheads=nheads
    )

    out = attn_output @ out_proj_weight.t() + out_proj_bias

    return out


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        trunc_normal_(self.in_proj_weight, std=0.02)
        trunc_normal_(self.out_proj.weight, std=0.02)

        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        return multi_head_attention(
            x,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj.weight,
            self.out_proj.bias,
        )


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

        self.attn = Attention(d_model, n_head)

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

        self.timeattn = Attention(d_model, n_head)
        self.ln_time = LayerNorm(d_model)
        self.temporal_fc = nn.Linear(d_model, d_model)

    def attention(self, x: torch.Tensor):
        assert self.attn_mask is None
        return self.attn(x)

    def forward(self, x: torch.Tensor, B, F, gridW):
        num_spatial_tokens = (x.size(1) - 1) // F
        gridH = num_spatial_tokens // gridW

        # print('tfm x ', x.min(), x.max())

        ## Temporal
        xt = x[:, 1:, :]
        xt = rearrange(xt, "b (h w t) m -> (b h w) t m", b=B, h=gridH, w=gridW, t=F)
        res_temporal = self.timeattn(self.ln_time(xt))
        res_temporal = rearrange(
            res_temporal, "(b h w) t m -> b (h w t) m", b=B, h=gridH, w=gridW, t=F
        )
        res_temporal = self.temporal_fc(res_temporal)
        xt = x[:, 1:, :] + res_temporal

        ## Spatial
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, F, 1)
        cls_token = rearrange(cls_token, "b t m -> (b t) m", b=B, t=F).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, "b (h w t) m -> (b t) (h w) m", b=B, h=gridH, w=gridW, t=F)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.attention(self.ln_1(xs))
        # print('tfm att', res_spatial.min(), res_spatial.max())

        ### Taking care of CLS token
        cls_token = res_spatial[:, 0, :]
        cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=F)
        cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
        res_spatial = res_spatial[:, 1:, :]
        res_spatial = rearrange(
            res_spatial, "(b t) (h w) m -> b (h w t) m", b=B, h=gridH, w=gridW, t=F
        )
        res = res_spatial
        x = xt

        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
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

    def forward(self, x: torch.Tensor, B, F, grid):
        for blk in self.resblocks:
            x = blk(x, B, F, grid)
        return x


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

        for m in self.transformer.resblocks:
            m_str = str(m)
            if "Block" in m_str:
                nn.init.constant_(m.temporal_fc.weight, 0)
                nn.init.constant_(m.temporal_fc.bias, 0)

    def forward(self, x: torch.Tensor):
        B, F, C, H, W = x.shape

        x = x.view(-1, C, H, W)
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        gridW = x.size(-1)

        x = x.flatten(2).transpose(2, 1)

        cls_token = self.class_embedding[None, None, :].to(x.dtype)
        cls_tokens = cls_token.expand(x.size(0), -1, -1)
        x_ = x
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.positional_embedding[None, :, :].to(x.dtype)

        ## Time Embeddings
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]
        x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=F)
        x = x + self.temporal_embed[None, :, :].to(x.dtype)
        # x = self.time_drop(x)
        x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=F)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = self.transformer(x, B, F, gridW)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


def make_timesformer_clip_vit_alt(nframes, model="ViT-B/32"):
    if model == "ViT-B/32":
        clip_input_resolution = 224
        clip_patch_size = 32
        clip_width = 768
        clip_layers = 12
        clip_heads = 12
        clip_output_dim = 512
    elif model == "ViT-B/16":
        clip_input_resolution = 224
        clip_patch_size = 16
        clip_width = 768
        clip_layers = 12
        clip_heads = 12
        clip_output_dim = 512
    elif model == "ViT-L/14":
        clip_input_resolution = 224
        clip_patch_size = 14
        clip_width = 1024
        clip_layers = 24
        clip_heads = 16
        clip_output_dim = 768

    timesfm = VisualTransformer(
        input_resolution=clip_input_resolution,
        patch_size=clip_patch_size,
        width=clip_width,
        layers=clip_layers,
        heads=clip_heads,
        output_dim=clip_output_dim,
        nframes=nframes,
    )

    model, preprocess = clip.load(model, device="cpu", jit=False)
    clip_vit_state = model.visual.state_dict()

    missing, unexpected = timesfm.load_state_dict(clip_vit_state, strict=False)
    assert len(unexpected) == 0
    # The only missing keys should be the new time-related ones
    assert all(["time" in x or "temporal" in x for x in missing])

    return timesfm


if __name__ == "__main__":
    import matplotlib
    import pylab
    from PIL import Image

    nframes = 8
    timesfm = make_timesformer_clip_vit_alt(nframes)
    timesfm.eval()

    model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

    torch.manual_seed(123)

    image = torch.randn(2, 1, 3, 224, 224)
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
