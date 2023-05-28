import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange

class Sequence_ASRI(nn.Module):
    def __init__(self, in_dims, embed_dims, num_heads, grid_size):
        super(Sequence_ASRI, self).__init__()
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5
        self.in_dims = in_dims
        self.embed_dims = embed_dims
        self.grid_size = grid_size

        self.proj_qkv = nn.Linear(in_dims, 4 * embed_dims, bias=True)
        self.proj_attn = nn.Linear(embed_dims, embed_dims, bias=True)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims, bias=True),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims, bias=True))

        self.shuffle = nn.Sequential(
            nn.LayerNorm(embed_dims),
            Rearrange('b (c h w)-> b c h w', h=1, w=1),
            nn.Conv2d(embed_dims, in_dims * 4 * 4, kernel_size=1, stride=1, padding=0),
            Rearrange('b (c p1 p2) h w-> b c (h p1) (w p2)', p1=4, p2=4),
            nn.UpsamplingBilinear2d(size=(16, 16)),
            nn.Conv2d(embed_dims, in_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dims),
            nn.ReLU(inplace=True)
        )

        self.arrange = Rearrange('b (s c head)-> s head b c', head=num_heads, s=4)
        self.rearrange = Rearrange('head b c -> b (c head)', head=num_heads)

    def forward(self, x):  # x, b x in_dims x h x w
        # shapes = x.shape
        fea_gap = torch.mean(x, dim=(2, 3), keepdim=False)

        proj_qkv = self.arrange(self.proj_qkv(fea_gap))
        q, k, v, fea_skip = proj_qkv[0], proj_qkv[1], proj_qkv[2], proj_qkv[3]

        attn = (q @ k.transpose(-2, -1)) * self.scale # 8*B*B
        attn = attn.softmax(dim=-1)
        attn = attn @ v
        attn = self.rearrange(attn)
        attn = self.proj_attn(attn)
        attn = attn + self.rearrange(fea_skip)

        attn = self.ffn(attn) + attn
        fea_up = self.shuffle(attn)

        fea_merge = x + fea_up
        return fea_merge


class Sequence_SSCM(nn.Module):
    def __init__(self, in_dims, embed_dims, num_heads, grid_size):
        super(Sequence_SSCM, self).__init__()
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5

        self.proj_q = nn.Linear(in_dims, embed_dims, bias=True)
        self.proj_k = nn.Linear(in_dims, embed_dims, bias=True)
        self.proj_v = nn.Conv2d(in_dims, embed_dims, kernel_size=1, stride=1, padding=0)

        self.proj_attn = nn.Sequential(
            nn.Conv2d(embed_dims, in_dims, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_dims),
            # nn.GELU(),
            nn.ReLU(inplace=True))

        self.arrange_qk = Rearrange('b (c head) -> head b c', head=num_heads)
        self.arrange_v = Rearrange('b (c head) h w -> head b (c h w)', head=num_heads)
        self.rearrange = Rearrange('head b (c h w) -> b (c head) h w', head=num_heads, h=grid_size, w=grid_size)

    def forward(self, x1, x2):  # fs, bxcxhxw, fq, bxcxhxw
        shapes = x1.shape
        fea_gap1 = torch.mean(x1, dim=(2, 3), keepdim=False)
        fea_gap2 = torch.mean(x2, dim=(2, 3), keepdim=False)

        proj_q = self.arrange_qk(self.proj_q(fea_gap1))
        proj_k = self.arrange_qk(self.proj_k(fea_gap2))
        proj_v = self.arrange_v(self.proj_v(x2))

        attn = (proj_q @ proj_k.transpose(-2, -1)) * self.scale# 8 B1 B2
        attn = attn.softmax(dim=-1)# 8 B1 B2
        attn = attn @ proj_v

        attn = self.rearrange(attn)
        attn = self.proj_attn(attn)
        return attn


class Pixel_SSCM(nn.Module):
    def __init__(self, in_dims, embed_dims, num_heads, grid_size):
        super(Pixel_SSCM, self).__init__()
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5

        self.proj_q = nn.Conv2d(in_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_dims, embed_dims, kernel_size=1, stride=1, padding=0)

        self.proj_attn = nn.Conv2d(embed_dims, in_dims, kernel_size=1, stride=1, padding=0)#nn.Sequential(

        self.proj_out = nn.Sequential(
            Rearrange('b c h w ->b (h w) c', h=grid_size, w=grid_size),
            nn.LayerNorm(in_dims),
            nn.Linear(in_dims, in_dims, bias=True),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            Rearrange('b (h w) c ->b c h w ', h=grid_size, w=grid_size))

        self.arrange = Rearrange('b (c head) h w -> head b (h w) c', head=num_heads)
        self.arrange_v = Rearrange('b (c head) h w -> head (b h w) c', head=num_heads)
        self.arrange2 = Rearrange('head b q h w ->  b head h (q w)', head=num_heads, h=256, w=256)
        self.rearrange = Rearrange('b head (h w) c -> b (c head) h w', head=num_heads, h=grid_size, w=grid_size)

    def forward(self, x1, x2):  # fs, bxcxhxw, fq, bxcxhxw
        shapes = x1.shape

        proj_q = self.arrange(self.proj_q(x1))
        proj_k = self.arrange(self.proj_k(x2))
        proj_v = self.arrange_v(self.proj_v(x2))

        attn = torch.einsum('pbnc,pqmc->pbqnm', proj_q, proj_k)#8 B1 B2 256 256
        attn = attn.softmax(dim=-1)
        attn = self.arrange2(attn)
        attn = attn @ proj_v
        attn = self.rearrange(attn)
        attn = self.proj_attn(attn)

        x = attn + x1
        x = self.proj_out(x)
        return x


class HRLM(nn.Module):
    def __init__(self, in_dims, embed_dims, num_heads, grid_size):  # in_dims:输入特征图维度，grid_size：特征图大小
        super(HRLM, self).__init__()
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5
        self.in_dims = in_dims
        self.embed_dims = embed_dims
        self.grid_size = grid_size

        self.sa = Sequence_ASRI(in_dims=in_dims, embed_dims=embed_dims, num_heads=num_heads, grid_size=grid_size)
        self.ca = Sequence_SSCM(in_dims=in_dims, embed_dims=embed_dims, num_heads=num_heads, grid_size=grid_size)
        self.sca = Pixel_SSCM(in_dims=in_dims, embed_dims=embed_dims, num_heads=num_heads, grid_size=grid_size)

    def forward(self, fea_support, fea_query):  # support 特征：fs, b x in_dims x h x w;  query 特征：fq, b x in_dims x h x w
        # self-attention
        sa_support = self.sa(fea_support)
        sa_query = self.sa(fea_query)

        # co-attention
        ca_support = self.ca(sa_support, sa_query)
        ca_query = self.ca(sa_query, sa_support)

        sca_support = self.sca(sa_support, ca_query)
        sca_query = self.sca(sa_query, ca_support)
        return sca_support, sca_query


