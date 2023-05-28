import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, labels):   #logits 25*1*256*256   labels 5*1*256*256
        # labels = labels
        smooth = 0.00001
        inse = torch.sum(logits * labels)
        p = torch.sum(logits)
        g = torch.sum(labels)
        loss_value = 1 - (2.0 * inse + smooth) / (p + g + smooth)
        return loss_value



def save_h5(h5f,data,target):
    shape_list=list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0]=None
        dataset = h5f.create_dataset(target, data=data,maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old=dataset.shape[0]
    len_new=len_old+data.shape[0]
    shape_list[0]=len_new
    dataset.resize(tuple(shape_list))
    dataset[len_old:len_new] = data