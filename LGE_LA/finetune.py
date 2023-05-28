# coding=utf-8
from __future__ import print_function
import h5py
import torch
import numpy as np
from tqdm import tqdm
import os
from model.model_finetune import Segmentation as Net2
from evaluation import DiceLoss, save_h5
import time
from torch.utils.data import DataLoader
from Data.dataset.LGE_LA import LADataSetFineTune


loss_fun = DiceLoss()

def main(path_data, path_model, lr_train, epochs, center, rate, num_class):  # Initialize everything and train
    file_name = 'Brompton.pkl' if center == 'C1' else "CARMA.pkl"
    num_split = 124 if center == 'C1' else 92
    train_dataset = LADataSetFineTune(data_root=path_data, file_name=file_name,rate=rate, num_split=num_split, phase='train')
    train_data_loader = DataLoader(train_dataset, batch_size=1, drop_last=False)

    path = '/.../Abdominal/model_pretrain.pkl'
    model_fine = Net2(num_class=num_class)
    save_model = torch.load(path)
    model_dict = model_fine.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model_fine.load_state_dict(model_dict)
    # model = (nn.DataParallel(model_test)).cuda()
    model_fine = model_fine.cuda()
    optimizer = torch.optim.Adam(model_fine.parameters(), lr=lr_train)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_dataset.num_batch, gamma=0.99, last_epoch=-1)

    for epoch in range(epochs):
        print('---', str(epoch), '---')
        time1 = time.time()
        Los_sup, Los_query, Acc_sup, Acc_query = 0, 0, 0, 0
        for n,(images,labels) in enumerate(train_data_loader):
            sup_img = torch.as_tensor(np.float32(images[0]).squeeze())
            query_img = torch.as_tensor(np.float32(images[1]).squeeze())
            sup_la = torch.as_tensor(np.float32(labels[0]).squeeze())
            query_la = torch.as_tensor(np.float32(labels[1]).squeeze())
            l_sup = sup_img.shape[0]
            l_query = query_img.shape[0]

            input_imgs = torch.unsqueeze(torch.cat([sup_img, query_img], dim=0), dim=1).cuda()
            sup_la = torch.unsqueeze(sup_la, dim=1).cuda()
            sup_bg = torch.ones_like(sup_la) - sup_la
            query_la = torch.unsqueeze(query_la, dim=1).cuda()
            query_bg = torch.ones_like(query_la) - query_la

            optimizer.zero_grad()
            pred_support, pred_query = model_fine(input_imgs, sup_la, l_sup, l_query)
            loss_sup_bg = loss_fun(torch.unsqueeze(pred_support[:, 0, :, :], dim=1), sup_bg)
            loss_sup_label1 = loss_fun(torch.unsqueeze(pred_support[:, 1, :, :], dim=1), sup_la)
            loss_query_bg = loss_fun(torch.unsqueeze(pred_query[:, 0, :, :], dim=1), query_bg)
            loss_query_label1 = loss_fun(torch.unsqueeze(pred_query[:, 1, :, :], dim=1), query_la)

            if epoch < 10:
                loss_sup = (loss_sup_bg + loss_sup_label1) / 2
                loss_query = (loss_query_bg + loss_query_label1) / 2
                loss =  0.5*loss_sup + 0.5*loss_query
            else:
                loss = 0.5*loss_sup_label1 + 0.5*loss_query_label1


            if rate == 0.6:
                if epoch < 10:
                    loss_sup = (loss_sup_bg + loss_sup_label1) / 2
                    loss_query = (loss_query_bg + loss_query_label1) / 2
                    loss =  loss_sup + loss_query
                else:
                    loss = loss_sup_label1 + loss_query_label1


            loss.backward()
            optimizer.step()
            schedulerD.step()

            sup_atrium_sigmoid = torch.tensor((pred_support[:, 1, :, :] > 0.5)).float()
            sup_atrium_dice = 2.0 * torch.sum(sup_atrium_sigmoid * torch.squeeze(sup_la)) / (torch.sum(sup_atrium_sigmoid) + torch.sum(torch.squeeze(sup_la)))
            query_atrium_sigmoid = torch.tensor((pred_query[:, 1, :, :] > 0.5)).float()
            query_atrium_dice = 2.0 * torch.sum(query_atrium_sigmoid * torch.squeeze(query_la)) / (torch.sum(query_atrium_sigmoid) + torch.sum(torch.squeeze(query_la)))

            los_sup = loss_sup.cpu().detach().numpy()
            los_query = loss_query.cpu().detach().numpy()
            sup_atrium_dice = sup_atrium_dice.cpu().detach().numpy()
            query_atrium_dice = query_atrium_dice.cpu().detach().numpy()
            Los_sup += los_sup
            Los_query += los_query
            Acc_sup += sup_atrium_dice
            Acc_query += query_atrium_dice
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        Loss_sup = Los_sup / train_dataset.num_batch
        Loss_query = Los_query / train_dataset.num_batch
        Accuracy_sup = Acc_sup / train_dataset.num_batch
        Accuracy_query = Acc_query / train_dataset.num_batch
        os.chdir(path_model)
        # data_fine = [epoch, Loss_sup, Accuracy_sup, Loss_query, Accuracy_query]
        # file = path_model + 'finetune.h5'
        # with h5py.File(file, 'a') as h5f:
        #     save_h5(h5f, data=np.array(data_fine), target='data')
        torch.save(model_fine.state_dict(), 'model_fine.pkl')
        print('train: sup:', Loss_sup, Accuracy_sup, 'query:', Loss_query, Accuracy_query)


        # test
        with torch.no_grad():
            os.chdir(path_model)
            model_test = Net2(num_class=num_class)
            # model = (nn.DataParallel(model)).cuda()
            model_test = model_test.cuda()
            state_dict_load = torch.load('model_fine.pkl')
            model_test.load_state_dict(state_dict_load)
            # model.eval()

            test_dataset = LADataSetFineTune(data_root=path_data, file_name=file_name, rate=rate, num_split=num_split, phase='test')
            test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

            test_Los_sup, test_Los_query, test_Acc_sup, test_Acc_query = 0, 0, 0, 0
            for m,(images,labels) in enumerate(test_data_loader):
                sup_img = torch.as_tensor(np.float32(images[0]).squeeze())
                query_img = torch.as_tensor(np.float32(images[1]).squeeze())
                sup_la = torch.as_tensor(np.float32(labels[0]).squeeze())
                query_la = torch.as_tensor(np.float32(labels[1]).squeeze())
                l_sup = sup_img.shape[0]
                l_query = query_img.shape[0]
                input_imgs = torch.unsqueeze(torch.cat([sup_img, query_img], dim=0), dim=1).cuda()
                sup_la = torch.unsqueeze(sup_la, dim=1).cuda()
                query_la = torch.unsqueeze(query_la, dim=1).cuda()

                pred_support, pred_query = model_test(input_imgs, sup_la, l_sup, l_query)  # B2*1*256*256
                loss_sup = loss_fun(torch.unsqueeze(pred_support[:, 1, :, :], dim=1), sup_la)
                loss_query = loss_fun(torch.unsqueeze(pred_query[:, 1, :, :], dim=1), query_la)
                sup_atrium_sigmoid = torch.tensor((pred_support[:, 1, :, :] > 0.5)).float()
                sup_atrium_dice = 2.0 * torch.sum(sup_atrium_sigmoid * torch.squeeze(sup_la)) / (torch.sum(sup_atrium_sigmoid) + torch.sum(torch.squeeze(sup_la)))
                query_atrium_sigmoid = torch.tensor((pred_query[:, 1, :, :] > 0.5)).float()
                query_atrium_dice = 2.0 * torch.sum(query_atrium_sigmoid * torch.squeeze(query_la)) / (torch.sum(query_atrium_sigmoid) + torch.sum(torch.squeeze(query_la)))

                los_sup = loss_sup.cpu().detach().numpy()
                los_query = loss_query.cpu().detach().numpy()
                sup_atrium_dice = sup_atrium_dice.cpu().detach().numpy()
                query_atrium_dice = query_atrium_dice.cpu().detach().numpy()
                test_Los_sup += los_sup
                test_Los_query += los_query
                test_Acc_sup += sup_atrium_dice
                test_Acc_query += query_atrium_dice
            t_Loss_sup = test_Los_sup / test_dataset.num_batch
            t_Loss_query = test_Los_query / test_dataset.num_batch
            t_Accuracy_sup = test_Acc_sup / test_dataset.num_batch
            t_Accuracy_query = test_Acc_query / test_dataset.num_batch
            time3 = time.time()
            print('epoch {}, test total time {:.2f}'.format(epoch, time3 - time2))
            # data_test = [epoch, t_Loss_sup, t_Accuracy_sup, t_Loss_query, t_Accuracy_query]
            # file = path_model + 'test.h5'
            # with h5py.File(file, 'a') as h5f:
            #     save_h5(h5f, data=np.array(data_test), target='data')
            print('test: sup:', t_Loss_sup, t_Accuracy_sup, 'query:', t_Loss_query, t_Accuracy_query)
    return



if __name__ == '__main__':
    path_data = '/.../'
    path_model = '/.../HRIN/LGE_LA/finetune/C2/5%/'
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main(path_data, path_model, lr_train=0.0001, epochs=100, center='C2', rate=0.05, num_class=1)
