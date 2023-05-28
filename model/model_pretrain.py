import torch
import torch.nn as nn
import torch.nn.functional as F

from .component import *
from .HRLM import *
from .prototype_pretrain import *


class Segmentation(nn.Module):
    def __init__(self, num_class):
        super(Segmentation, self).__init__()

        self.encoding = Encoding(feature_root=32, depth=5)
        self.relation = HRLM(in_dims=512, embed_dims=512, num_heads=8, grid_size=16)  # in_dims:输入特征图维度，grid_size：特征图大小
        self.upsampling = Upsampling(feature_root=32, depth=5)
        self.metric = Label_Relation(in_dims=32, grid_size=256, num_class=num_class)  # in_dims:输入特征图维度，grid_size：特征图大小

    def forward(self, imgs, label, l_support, l_query):  # fs, bxcxhxw, fq, bxcxhxw
        img_support = imgs[:l_support]
        img_query = imgs[-l_query:]
        fea_supports = self.encoding(img_support)
        fea_querys = self.encoding(img_query)

        fea_support = fea_supports[-1]
        fea_query = fea_querys[-1]
        fea_support, fea_query = self.relation(fea_support, fea_query)

        fea_support = self.upsampling(fea_support, fea_supports)
        fea_query = self.upsampling(fea_query, fea_querys)
        pred_support, pred_query = self.metric(fea_support, fea_query, label)
        pred_support_ = F.sigmoid(pred_support)
        pred_query_ = F.sigmoid(pred_query)
        return pred_support_, pred_query_