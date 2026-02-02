#  Copyright Jian Wang @ MPI-INF (c) 2023.

from copy import deepcopy
from functools import partial
from einops import repeat


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from mmpose.models import build_posenet
from mmpose.models.backbones.base_backbone import BaseBackbone
from ..builder import BACKBONES
from ..builder import NECKS
from .vit import ViT, PatchEmbed as OrigPatchEmbed
from .undistort_transformer.undistort_patch import UndistortPatch
from pdb import set_trace
import loralib as lora




def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_lora = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.use_lora = use_lora

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

       
        if self.use_lora is False:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        else:
            self.qkv = lora.MergedLinear(
                dim,                         # in_features
                (self.num_heads * (dim // self.num_heads)) * 3,   # out_features = 3 * all_head_dim
                r=8,                         # 랭크(보통 4~8)
                enable_lora=[True, False, True],  # Q, K, V 중 Q/V에만 LoRA
                merge_weights=False          # 학습 중엔 분리, eval() 때 자동 병합 가능
            )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None, use_lora = False
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, use_lora = use_lora
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedMLP(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_num_horizontal, patch_num_vertical, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = patch_num_horizontal * patch_num_vertical
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        in_dims = in_chans * patch_size[0] * patch_size[1]

        self.proj = nn.Linear(in_dims, embed_dim)
        self.input_layernorm = nn.LayerNorm(in_dims)
        self.output_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        B, patch_num, C, H, W = x.shape
        assert patch_num == self.num_patches
        x = x.view(B * patch_num, C * H * W)
        x = self.input_layernorm(x)
        x = self.proj(x)
        x = self.output_layernorm(x)
        x = x.view(B, patch_num, self.embed_dim)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_num_horizontal, patch_num_vertical, patch_size=32, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = patch_num_horizontal * patch_num_vertical
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        in_dims = in_chans * patch_size[0] * patch_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio),
                              padding=0)

    def forward(self, x, **kwargs):
        B, patch_num, C, H, W = x.shape
        # set_trace()
        assert patch_num == self.num_patches
        x = x.view(B * patch_num, C, H, W)
        x = self.proj(x)
        x = x.view(B, patch_num, self.embed_dim)

        return x



class PatchEmbedWithCNN(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_num_horizontal, patch_num_vertical, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = patch_num_horizontal * patch_num_vertical
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        in_dims = in_chans * patch_size[0] * patch_size[1]

        # use resnet to extract feature
        self.proj = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.proj.fc = nn.Linear(2048, self.embed_dim)

    def forward(self, x, **kwargs):
        B, patch_num, C, H, W = x.shape
        assert patch_num == self.num_patches
        x = x.view(B * patch_num, C, H, W)
        x = self.proj(x)
        x = x.view(B, patch_num, self.embed_dim)
        return x


@BACKBONES.register_module()
class UndistortViT(ViT):
    def __init__(self, fisheye2sphere_config=None, **vit_kwargs):
        # 1) 원본 ViT 초기화: 여기서 patch_embed, pos_embed, blocks 다 세팅되고
        #    pretrained 체크포인트를 받을 준비가 된다.
        super().__init__(**vit_kwargs)

        # 2) Store and build undistortion patch sampler
        from mmpose.models import build_posenet

        new_patch_num_horizontal = 12
        new_patch_num_vertical = 16
        self.patch_embed = PatchEmbed(new_patch_num_horizontal,
                                          new_patch_num_vertical,
                                          patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
                                          ratio=self.ratio)
        self.fisheye2sphere_config = fisheye2sphere_config
        self.fisheye2sphere = build_posenet(fisheye2sphere_config)

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        # reuse original ViT init, so pretrained weights (pos_embed, patch_embed.proj) are loaded
        super().init_weights(pretrained)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        # set_trace()
        B = x.shape[0]
        # 1) fisheye → undistortion patch sampling: (B, N_patches, C, H_patch, W_patch)
        x = self.fisheye2sphere(x)
        # N, C, Ph, Pw = x.shape[1:]
        # 2) PatcPwh embedding using original ViT.patch_embed (pretrained conv)
        # x = x.view(B * N, C, Ph, Pw)
        x = self.patch_embed(x)
        
        # set_trace()
        # x_seq: (B*N, 1, embed_dim) → reshape to (B, N, embed_dim)
        # x = x.squeeze(1).view(B, N, self.embed_dim)

        # 3) Resize and add positional embeddings
        if self.pos_embed is not None:
            # original grid size from ViT config
            # ori_h, ori_w = self.patch_embed.origin_patch_shape
            # # new grid dims from undistort sampler config
            # new_h = self.fisheye2sphere_config['patch_num_vertical']
            # new_w = self.fisheye2sphere_config['patch_num_horizontal']
            # # interpolate pos_embed to new grid
            # new_pos = get_abs_pos(self.pos_embed, new_h, new_w, ori_h, ori_w)
            # x = x + new_pos[:, 1:]
            # set_trace()
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        # set_trace()

        # 4) Prepend task tokens and apply Transformer blocks
        task_tokens = repeat(self.task_tokens, '() n d -> b n d', b=B)
        # set_trace()
        x = torch.cat((task_tokens, x), dim=1)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)
        # set_trace()
        task_tokens = x[:, :self.task_tokens_num]  # [N,J,C]
        # task_tokens = torch.cat(task_tokens_, dim=-1)
        xp = x[:, self.task_tokens_num:]  # [N,Hp*Wp,C]
        # set_trace()

        xp = xp.permute(0, 2, 1).reshape(B, -1, 16, 12).contiguous()

        return xp, task_tokens

        # # 5) Split task tokens vs patch tokens
        # task_out = x[:, : self.task_tokens_num]
        # patch_out = x[:, self.task_tokens_num :]
        #         # 6) Reconstruct spatial feature map with original grid size
        # ori_h, ori_w = self.patch_embed.origin_patch_shape
        # # reshape to (B, embed_dim, ori_h, ori_w) to match downstream head expectation
        # xp = patch_out.permute(0, 2, 1).reshape(B, self.embed_dim, ori_h, ori_w)
        # xp = xp.contiguous()
        # return xp, task_out

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

