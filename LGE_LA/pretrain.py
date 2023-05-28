# coding=utf-8
from __future__ import print_function
from model.model_pretrain import Segmentation as SEG_NET

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import h5py


from torch.utils.data import DataLoader
from comon.evaluation import DiceLoss
from Data.dataset.LGE_LA import Abdominaldataset

loss_fun = DiceLoss()



def main(path, path_model, gpu, lr_train, epochs, num_class):  # Initialize everything and train
    num_l2_train = 30
    data_loader = DataLoader(Abdominaldataset(data_root=path,num_batch=30,num_split=30), batch_size=1,shuffle=False,drop_last=False)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    model = SEG_NET(num_class=num_class)
    model = (nn.DataParallel(model)).cuda()
    # model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_train)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_l2_train, gamma=0.99, last_epoch=-1)


    for epoch in range(epochs):
        print('---', str(epoch), '---')
        # train
        Los_sup, Los_query, Acc_sup, Acc_query = 0, 0, 0, 0
        for i,(images,labels) in enumerate(data_loader):
            sup_img = torch.as_tensor(np.float32(images[0]).squeeze())
            query_img = torch.as_tensor(np.float32(images[1]).squeeze())
            sup_la = torch.as_tensor(np.float32(labels[0]).squeeze())
            query_la = torch.as_tensor(np.float32(labels[1]).squeeze())
            l_sup = sup_img.shape[0]
            l_query = query_img.shape[0]

            input_imgs = torch.unsqueeze(torch.cat([sup_img, query_img], dim=0), dim=1).cuda()
            sup_la = sup_la.cuda().permute(1,0,2,3)  # B1*5*256*256 第二维度是label
            query_la = query_la.cuda().permute(1,0,2,3)

            optimizer.zero_grad()
            pred_support, pred_query = model(input_imgs, sup_la, l_sup, l_query)  # B2*1*256*256

            loss_sup_bg = loss_fun(pred_support[:, 0, :, :], sup_la[:, 0, :, :])
            loss_sup_label1 = loss_fun(pred_support[:, 1, :, :], sup_la[:, 1, :, :])
            loss_sup_label2 = loss_fun(pred_support[:, 2, :, :], sup_la[:, 2, :, :])
            loss_sup_label3 = loss_fun(pred_support[:, 3, :, :], sup_la[:, 3, :, :])
            loss_sup_label4 = loss_fun(pred_support[:, 4, :, :], sup_la[:, 4, :, :])

            loss_query_bg = loss_fun(pred_query[:, 0, :, :], query_la[:, 0, :, :])
            loss_query_label1 = loss_fun(pred_query[:, 1, :, :], query_la[:, 1, :, :])
            loss_query_label2 = loss_fun(pred_query[:, 2, :, :], query_la[:, 2, :, :])
            loss_query_label3 = loss_fun(pred_query[:, 3, :, :], query_la[:, 3, :, :])
            loss_query_label4 = loss_fun(pred_query[:, 4, :, :], query_la[:, 4, :, :])

            loss_sup = (loss_sup_bg + loss_sup_label1 + loss_sup_label2 + loss_sup_label3 + loss_sup_label4) / 5
            loss_query = (loss_query_bg + loss_query_label1 + loss_query_label2 + loss_query_label3 + loss_query_label4) / 5
            loss = 0.42*loss_sup + loss_query

            loss.backward()
            optimizer.step()
            schedulerD.step()

            sup_atrium_sigmoid = torch.tensor((pred_support > 0.5)).float()
            sup_atrium_dice = 2.0 * torch.sum(sup_atrium_sigmoid * sup_la) / (torch.sum(sup_atrium_sigmoid) + torch.sum(sup_la))
            query_atrium_sigmoid = torch.tensor((pred_query > 0.5)).float()
            query_atrium_dice = 2.0 * torch.sum(query_atrium_sigmoid * query_la) / (torch.sum(query_atrium_sigmoid) + torch.sum(query_la))

            los_sup = loss_sup.cpu().detach().numpy()
            los_query = loss_query.cpu().detach().numpy()
            sup_atrium_dice = sup_atrium_dice.cpu().detach().numpy()
            query_atrium_dice = query_atrium_dice.cpu().detach().numpy()
            Los_sup += los_sup
            Los_query += los_query
            Acc_sup += sup_atrium_dice
            Acc_query += query_atrium_dice
        Loss_sup = Los_sup / num_l2_train
        Loss_query = Los_query / num_l2_train
        Accuracy_sup = Acc_sup / num_l2_train
        Accuracy_query = Acc_query / num_l2_train
        os.chdir(path_model)
        torch.save(model.state_dict(), 'model_pretrain.pkl')

        print('train: sup:', Loss_sup, Accuracy_sup, 'query:', Loss_query, Accuracy_query)

    return




if __name__ == '__main__':
    path_data = '/data/lixuejiao/data/'
    path_model = '/data/lixuejiao/data/HRIN/LGE_LA/pretrain/Abdominal/'
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    main(path_data, path_model, gpu='2', lr_train=0.0001, epochs=2000, num_class=4)