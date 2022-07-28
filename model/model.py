import warnings
from shutil import Error

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torch.nn.init import constant_

from .timesformer_clip_alt import make_timesformer_clip_vit_alt


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def normalize_eps(x, eps=1e-9):
    return normalize(x + eps)


def squash(s):
    s = s + 1e-9
    mag_sq = torch.sum(s**2, dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s


def sub_mean(state, s):
    if state.training and "finaltf" not in state.branch_to_freeze:
        # Fake batch norm to store running stats
        _ = state.mean_center_bn(s.detach())

        # Subtract the batch mean
        s = s - s.mean(0)
    else:
        s = s - state.mean_center_bn.running_mean
    return s


def bn(state, s):
    # Make sure running mean is used when
    # final transformer is frozen
    if "finaltf" in state.branch_to_freeze:
        state.mean_center_bn.eval()

    s = state.mean_center_bn(s)
    return s


NEEDS_STATE = ["sub_mean", "bn"]
RESIDUAL_ACTIVATIONS = {
    "normalize": normalize_eps,
    "squash": squash,
    "squash10": lambda x: 10 * squash(x),
    "squash1p2": lambda x: 1.2 * squash(x),
    "squash1p5": lambda x: 1.5 * squash(x),
    "squash1p8": lambda x: 1.8 * squash(x),
    "tanh": lambda x: torch.tanh(x).clone(),
    "none": lambda x: x,
    "sub_mean": sub_mean,
    "bn": bn,
    None: lambda x: x,
}


class MLP(nn.Module):
    def __init__(self, num_classes=512, num_features=512, p=0.2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class JointEmbedding(nn.Module):
    def __init__(self, input_dims_a, input_dims_b, embedding_dims, normalize=True):
        super().__init__()
        self.branch_a = self._make_branch(input_dims_a, embedding_dims)
        self.branch_b = self._make_branch(input_dims_b, embedding_dims)
        self.normalize = normalize

    @staticmethod
    def _make_branch(input_dims, num_features):
        return nn.Sequential(
            nn.Linear(input_dims, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )

    def forward(self, x_a, x_b):
        feats_a = self.branch_a(x_a)
        feats_b = self.branch_b(x_b)
        if self.normalize:
            feats_a = F.normalize(feats_a, p=2, dim=1)
            feats_b = F.normalize(feats_b, p=2, dim=1)
        return feats_a, feats_b


class CLIP(JointEmbedding):
    def __init__(self, input_dims_a, input_dims_b, embedding_dims):
        super().__init__(input_dims_a, input_dims_b, embedding_dims, normalize=True)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_a, x_b):
        feats_a, feats_b = super().forward(x_a, x_b)
        sim = torch.einsum("id, jd -> ij", feats_a, feats_b) * self.temperature
        return feats_a, feats_b, sim


class PretrainedCLIPBase(nn.Module):
    def _common_init(self):
        if getattr(self, "residual_activation", None) in ["sub_mean", "bn"]:
            # Add a bn layer
            self.mean_center_bn = nn.BatchNorm1d(
                self.feature_dim, affine=False, momentum=0.2
            )

    def _adapt_feature(self, feature_main, features_aux):
        # feature_main [b, 512]
        # features_aux [ncomms, b, 512]

        residual_activation = RESIDUAL_ACTIVATIONS[self.residual_activation]

        assert len(feature_main.shape) == 2
        b = feature_main.shape[0]

        concat_feats = torch.stack([feature_main, *features_aux], axis=0)
        concat_feats = normalize(concat_feats)

        assert concat_feats.shape[1] == b

        comm_tfm = self.final_transformer(concat_feats)
        if self.init_from_avg:
            comm_res = normalize(
                torch.mean(torch.stack([normalize(s) for s in comm_tfm], axis=0), dim=0)
            )
        else:
            comm_res = self.final_linear(comm_tfm[0])

        dbg = torch.rand([]) < 0.05
        if dbg:
            eg_pre = comm_res[0].detach().clone()
            bmean = comm_res.mean(0).detach()

        if self.residual_activation in NEEDS_STATE:
            comm_res = residual_activation(self, comm_res)
        else:
            comm_res = residual_activation(comm_res)

        if dbg:
            eg_post = comm_res[0].detach()
            print(
                "Debug residual (activation=",
                self.residual_activation,
                "): pre-activation norm",
                eg_pre.norm(dim=-1).item(),
                "max",
                eg_pre.max().item(),
                ", post-activation norm",
                eg_post.norm(dim=-1).item(),
                "max",
                eg_post.max().item(),
                "bmean",
                bmean[:5],
            )
            if hasattr(self, "writer"):
                self.writer.add_scalar(
                    "pre_activation_norm", eg_pre.norm(dim=-1).item()
                )
                self.writer.add_scalar("pre_activation_max", eg_pre.max().item())
                self.writer.add_scalar(
                    "post_activation_norm", eg_post.norm(dim=-1).item()
                )
                self.writer.add_scalar("post_activation_max", eg_post.max().item())

        if self.training and self.random_skip_adapter:
            comm_mask = torch.rand(comm_res.shape[:-1]) > 0.5
            comm_res[comm_mask] = 0.0

        adapted = normalize(normalize(feature_main) + comm_res)
        # adapted = comm_res
        return adapted

    def _load_comment_features(self, comments):
        empty_string_mask = comments[..., 1] == 49407
        b, ncomms, ntoks = comments.shape
        feats_comm = self.model.encode_text(comments.reshape(b * ncomms, ntoks))
        feats_comm = feats_comm.reshape(b, ncomms, self.feature_dim).float()
        feats_comm[empty_string_mask] = self.mask_embedding
        feats_comm = feats_comm.permute(1, 0, 2)
        return feats_comm

    def _encode_with_comments(self, feats_vis, feats_title, comments):
        # 49407 is the end-of-text token
        # so if it is in position 1 the string is empty

        if (
            hasattr(self, "init_audio_model")
            and self.init_audio_model
            and isinstance(comments, list)
        ):
            comments, feats_audio = comments
            feats_comm = self._load_comment_features(comments)
            feats_audio = feats_audio.permute(1, 0, 2)
            feats_audio = [self.audio_model.mlp(feat) for feat in feats_audio]
            feats_audio = torch.stack(feats_audio, axis=0)
            feats_comm = torch.cat((feats_comm, feats_audio), axis=0)
        else:
            feats_comm = self._load_comment_features(comments)

        bs = feats_title.shape[0]
        if self.training:
            if self.random_comment_masking:
                comm_masks = [
                    torch.randint(low=0, high=2, size=(bs, 1), device=comm.device)
                    for comm in feats_comm
                ]
            else:
                comm_masks = torch.ones(len(feats_comm)).to(feats_comm.device)
            feats_comm = [
                comm * mask + self.mask_embedding * (1 - mask)
                for comm, mask in zip(feats_comm, comm_masks)
            ]
            branch_to_adapt = self.branch_to_adapt
        else:
            branch_to_adapt = self.branch_to_adapt_val

        if branch_to_adapt == "text":
            feats_vis_out = feats_vis
            feats_text_out = self._adapt_feature(feats_title, feats_comm)
        elif branch_to_adapt == "image":
            feats_vis_out = self._adapt_feature(feats_vis, feats_comm)
            feats_text_out = feats_title
        elif branch_to_adapt == "skip":
            feats_vis_out = feats_vis
            feats_text_out = feats_title
        else:
            raise Exception("Unknown branch_to_adapt")

        feats_vis_out_norm = normalize(feats_vis_out)
        feats_text_out_norm = normalize(feats_text_out)

        return feats_vis_out_norm, feats_text_out_norm

    def _freeze(self, branch_to_freeze):
        self.branch_to_freeze = branch_to_freeze

        if branch_to_freeze is False:
            return

        if branch_to_freeze == "none":
            return

        did_freeze = False
        if "visual" in branch_to_freeze:
            did_freeze = True
            for param in self.model.visual.parameters():
                param.requires_grad = False
        if "text" in branch_to_freeze:
            did_freeze = True
            for param in self.model.transformer.parameters():
                param.requires_grad = False
        if "all" in branch_to_freeze:
            did_freeze = True
            for param in self.model.named_parameters():
                param[1].requires_grad = False
        if "finaltf" in branch_to_freeze:
            did_freeze = True
            if hasattr(self, "final_transformer"):
                for param in self.final_transformer.parameters():
                    param.requires_grad = False
                for param in self.final_linear.parameters():
                    param.requires_grad = False
                self.mask_embedding.requires_grad = False
            else:
                warnings.warn(
                    "Tried to freeze finaltf but model"
                    "has no final transformer, ignoring!"
                )

        if not did_freeze:
            raise Exception("Unknown branch_to_freeze")


class PretrainedCLIP(PretrainedCLIPBase):
    def __init__(
        self,
        model_type="ViT-B/32",
        freeze=False,
        residual_activation=None,
        comment_fusion=None,
    ):
        super().__init__()
        self.model, _ = clip.load(model_type, device="cpu", jit=False)
        self.model = self.model.float()

        self.feature_dim = self.model.ln_final.normalized_shape[0]
        self.residual_activation = residual_activation
        self.comment_fusion = comment_fusion
        self._common_init()
        self._freeze(freeze)

    def forward(self, vis, title, comments=None):
        shp = vis.shape
        if len(shp) == 2 and shp[1] == self.feature_dim:
            # Precomputed feature
            feats_vis = vis
        elif len(shp) == 4:
            feats_vis = self.model.encode_image(vis).float()
        elif len(shp) == 5:
            # If there is a temporal dimension take mean over time
            feats_vis = self.model.encode_image(
                vis.reshape(shp[0] * shp[1], shp[2], shp[3], shp[4])
            ).float()
            feats_vis = feats_vis.reshape(shp[0], shp[1], -1).mean(1)

        feats_title = self.model.encode_text(title)

        if (
            comments is None
            or self.comment_fusion is None
            or self.comment_fusion == "None"
        ):
            feats_text = feats_title
        else:
            b, ncomms, ntoks = comments.shape
            feats_comm = (
                self.model.encode_text(comments.reshape(b * ncomms, ntoks))
                .reshape(b, ncomms, self.feature_dim)
                .float()
            )

            if self.comment_fusion == "averaging":
                feats_text = torch.mean(
                    torch.cat(
                        [feats_title.unsqueeze(0), feats_comm.permute(1, 0, 2)], 0
                    ),
                    axis=0,
                )
            else:
                raise Error("Comment fusion method not specified.")

        feats_text = normalize(feats_text)
        feats_vis = normalize(feats_vis)

        sim = self.model.logit_scale.exp().to(vis.device) * feats_vis @ feats_text.t()

        return feats_vis, feats_text, sim


class PretrainedCLIP_finaltf(PretrainedCLIPBase):
    def __init__(
        self,
        model_type="ViT-B/32",
        freeze=False,
        branch_to_adapt="text",
        branch_to_adapt_val="text",
        residual_activation=None,
        n_layers=2,
        n_heads=8,
        init_from_avg=True,
        random_comment_masking=False,
        random_skip_adapter=True,
        init_audio_model=False,
        audio_model_ckpt=None,
        clip_audio_ckpt=None,
    ):
        super().__init__()
        self.model, _ = clip.load(model_type, device="cpu", jit=False)
        self.model = self.model.float()
        self.feature_dim = self.model.ln_final.normalized_shape[0]

        self.final_transformer = clip.model.Transformer(
            width=self.feature_dim, layers=int(n_layers), heads=int(n_heads)
        )
        self.final_linear = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.mask_embedding = nn.Parameter(torch.randn(1, self.feature_dim))
        # self.cls_embedding = nn.Parameter(torch.randn(1, self.feature_dim))
        self.branch_to_adapt = branch_to_adapt
        self.branch_to_adapt_val = branch_to_adapt_val
        self.residual_activation = residual_activation
        self.init_from_avg = init_from_avg
        self.random_comment_masking = random_comment_masking
        self.random_skip_adapter = random_skip_adapter
        self.init_audio_model = init_audio_model
        if self.init_audio_model:
            try:
                from GDT.model import AudioBaseNetwork, Identity
            except Exception as e:
                raise ValueError(
                    "for audio experiments, GDT repository needs to be cloned from https://github.com/facebookresearch/GDT."
                )

            self.audio_model = AudioBaseNetwork("resnet9", pretrained=True, duration=1)
            if audio_model_ckpt is not None:
                ckpt = torch.load(audio_model_ckpt, map_location="cpu")
                ckpt = {
                    k.split("audio_network.")[1]: v
                    for k, v in ckpt["model"].items()
                    if "audio" in k
                }
                ckpt["base.fc.weight"] = self.audio_model.base.fc.weight
                ckpt["base.fc.bias"] = self.audio_model.base.fc.bias
                self.audio_model.load_state_dict(ckpt)
            if clip_audio_ckpt is not None:
                pretrained_clip = torch.load(clip_audio_ckpt, map_location="cpu")
                clip_ckpt = {
                    k[len("model.") :]: v
                    for k, v in pretrained_clip["state_dict"].items()
                    if "model" in k
                }
                self.model.load_state_dict(clip_ckpt)

            self.audio_model.fc = Identity()
            self.audio_model.mlp = MLP()

        if self.init_from_avg:
            for i in range(int(n_layers)):
                dict(self.final_transformer.resblocks[i].mlp.c_proj.named_parameters())[
                    "weight"
                ].data.zero_()
                dict(self.final_transformer.resblocks[i].mlp.c_proj.named_parameters())[
                    "bias"
                ].data.zero_()
                dict(self.final_transformer.resblocks[i].attn.named_parameters())[
                    "out_proj.weight"
                ].data.zero_()

        constant_(self.final_linear.weight, 0.0)
        # constant_(self.final_linear.bias, 0.)

        self._common_init()
        self._freeze(freeze)

    def forward(self, vis, title, comments):
        shp = vis.shape
        if len(shp) == 2 and shp[1] == self.feature_dim:
            # Precomputed feature
            feats_vis = vis
        if len(shp) == 4:
            feats_vis = self.model.encode_image(vis).float()
        elif len(shp) == 5:
            # If there is a temporal dimension take mean over time
            feats_vis = self.model.encode_image(
                vis.reshape(shp[0] * shp[1], shp[2], shp[3], shp[4])
            ).float()
            feats_vis = feats_vis.reshape(shp[0], shp[1], -1).mean(1)

        feats_title = self.model.encode_text(title)

        feats_vis, feats_text = self._encode_with_comments(
            feats_vis, feats_title, comments
        )

        sim = self.model.logit_scale.exp().to(vis.device) * feats_vis @ feats_text.t()

        return feats_vis, feats_text, sim


class PretrainedCLIP_TimeSformer(PretrainedCLIPBase):
    def __init__(self, model_type="ViT-B/32", freeze=False, residual_activation=None):
        super().__init__()
        self.model, _ = clip.load(model_type, device="cpu", jit=False)
        self.model = self.model.float()
        self.model.visual = make_timesformer_clip_vit_alt(nframes=8, model=model_type)
        self.residual_activation = residual_activation

        self._common_init()
        self._freeze(freeze)

    def forward(self, im, text, comments=None):
        # [b, t, c, h, w] im.shape

        feats_im = self.model.visual(im)

        feats_text = self.model.encode_text(text)

        feats_im = normalize(feats_im)
        feats_text = normalize(feats_text)

        sim = self.model.logit_scale.exp().to(im.device) * feats_im @ feats_text.t()

        return feats_im, feats_text, sim


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    from .timesformer_clip_alt import Attention

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class PretrainedCLIP_TimeSformer_finaltf(PretrainedCLIPBase):
    def __init__(
        self,
        model_type="ViT-B/32",
        freeze=False,
        branch_to_adapt="text",
        branch_to_adapt_val="text",
        residual_activation=None,
        visual_device=None,
        n_layers=2,
        n_heads=8,
        init_from_avg=True,
        random_comment_masking=False,
        random_skip_adapter=True,
    ):
        super().__init__()
        self.model, _ = clip.load(model_type, device="cpu", jit=False)
        self.model = self.model.float()
        self.model.visual = make_timesformer_clip_vit_alt(nframes=8, model=model_type)

        self.feature_dim = self.model.ln_final.normalized_shape[0]
        self.final_transformer = clip.model.Transformer(
            width=self.feature_dim, layers=int(n_layers), heads=int(n_heads)
        )
        self.final_linear = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.mask_embedding = nn.Parameter(torch.randn(1, self.feature_dim))
        self.branch_to_adapt = branch_to_adapt
        self.branch_to_adapt_val = branch_to_adapt_val
        self.residual_activation = residual_activation
        self.init_from_avg = init_from_avg
        self.random_comment_masking = random_comment_masking
        self.random_skip_adapter = random_skip_adapter

        if self.init_from_avg:
            for i in range(int(n_layers)):
                dict(self.final_transformer.resblocks[i].mlp.c_proj.named_parameters())[
                    "weight"
                ].data.zero_()
                dict(self.final_transformer.resblocks[i].mlp.c_proj.named_parameters())[
                    "bias"
                ].data.zero_()
                dict(self.final_transformer.resblocks[i].attn.named_parameters())[
                    "out_proj.weight"
                ].data.zero_()

        constant_(self.final_linear.weight, 0.0)
        # constant_(self.final_linear.bias, 0.)

        self._common_init()
        self._freeze(freeze)

        self.multigpu = visual_device is not None
        # Process visual branch on a separate gpu
        if self.multigpu:
            self.visual_device = torch.device(visual_device)
            self.text_device = None

    def forward(self, vis, title, comments):
        # [b, t, c, h, w] vis shape
        # [b, ntoks] title shape
        # [b, ncomms, ntoks] comments shape

        if self.multigpu and self.text_device is None:
            # Transfer visual branch to other gpu
            # if it hasn't been done yet
            self.text_device = title.device
            self.model.visual.to(self.visual_device)

        vis = vis.type(self.model.dtype)
        if self.multigpu:
            feats_vis = self.model.visual(vis.to(self.visual_device)).to(
                self.text_device
            )
        else:
            feats_vis = self.model.visual(vis)

        feats_title = self.model.encode_text(title)

        feats_vis, feats_text = self._encode_with_comments(
            feats_vis.float(), feats_title.float(), comments
        )

        sim = self.model.logit_scale.exp().to(vis.device) * feats_vis @ feats_text.t()

        return feats_vis, feats_text, sim


class R2Plus1D_34_IG65M_32frames(nn.Module):
    """Wrapper around R(2+1)D 34 model pretrained
    on IG65M with 32 frame clips

    TODO:
        * output features, max or avg spatio/temporal pooling
        see https://github.com/moabitcoin/ig65m-pytorch/blob/master/ig65m/cli/extract.py#L24

        * try to replicate collab experts feature extraction
    """

    def __init__(self, pool_spatial="mean", pool_temporal="mean"):
        super().__init__()

        # Will use r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth
        self.model = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_32_ig65m",
            num_classes=359,
            pretrained=True,
        )

        self.pool_spatial = Reduce("n c t h w -> n c t", reduction=pool_spatial)
        self.pool_temporal = Reduce("n c t -> n c", reduction=pool_temporal)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.pool_spatial(x)
        x = self.pool_temporal(x)

        return x
