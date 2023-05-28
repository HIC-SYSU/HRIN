import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


class Label_Relation(nn.Module):
    def __init__(self, in_dims, grid_size, num_class):
        super(Label_Relation, self).__init__()
        self.in_dims = in_dims
        self.proj_support = nn.Conv2d(in_dims, in_dims, kernel_size=5, stride=1, padding=2)
        self.proj_query = nn.Conv2d(in_dims, in_dims, kernel_size=5, stride=1, padding=2)
        self.pred_support_ = nn.Linear(256 * 256, 1)
        self.pred_support = nn.Conv2d(in_dims, num_class+1, kernel_size=5, stride=1, padding=2)
        self.pred_query = nn.Conv2d(in_dims, num_class+1, kernel_size=5, stride=1, padding=2)
        self.arrange = Rearrange('b c h w-> b (h w) c ')
        self.rearrange = Rearrange('b (h w) c-> b c h w', h=grid_size, w=grid_size)

    def forward(self, fs, fq, label):  #fs, b x in_dims x 256 x 256;  fq, b1 x in_dims x 256 x 256;
        fs = self.proj_support(fs)
        fq = self.proj_query(fq)

        background = torch.ones_like(label) - label
        # foreground + background
        fea_background = fs * background
        fea_foreground = fs * label

        fea_background = fea_background.reshape(-1, 32, 256 * 256)
        fea_label1 = fea_foreground.reshape(-1, 32, 256 * 256)

        fea_background = self.pred_support_(fea_background)
        fea_label1 = self.pred_support_(fea_label1)

        agree_background = torch.mean(fea_background, dim=0, keepdim=False).reshape(1, 1, -1)
        agree_label1 = torch.mean(fea_label1, dim=0, keepdim=False).reshape(1, 1, -1)

        # concat
        fea_mf = torch.cat([agree_background, agree_label1], dim=1)

        fea_fq = self.arrange(fq)
        fea_mf1 = fea_mf.transpose(-1, -2)
        weight = fea_fq @ fea_mf1 * (self.in_dims * 1.0 ** -0.5)
        weight = weight.softmax(dim=-1)
        logit_query = weight @ fea_mf
        logit_query = self.rearrange(logit_query)
        logit_query = logit_query + fq

        # predict
        pred_support = self.pred_support(fs)
        pred_query = self.pred_query(logit_query)
        return pred_support, pred_query