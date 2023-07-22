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
from Data.dataset.othertasks import TaskDatasetFineTune


loss_fun = DiceLoss()

def main(path_data, path_model, file_name, task, lr_train, epochs, num_class):  # Initialize everything and train
    # file_name = 'Brompton.pkl' if center is 'C1' else "CARMA.pkl"
    num_split = 10
    train_dataset = TaskDatasetFineTune(data_root=path_data, file_name=file_name,phase='train')
    train_data_loader = DataLoader(train_dataset, batch_size=1, drop_last=False)

    path = '/.../HRIN/othertasks/pretrain/'+ str(task) +'/model_pretrain.pkl'
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
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_split, gamma=0.99, last_epoch=-1)

    global w1
    global w2
    w1 = Variable(torch.ones(1), requires_grad=True).cuda()
    w2 = Variable(torch.ones(1), requires_grad=True).cuda()
    
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
            pred_support, pred_query = model_fine(input_imgs, sup_la, l_sup, l_query)  # B2*1*256*256
            loss_sup_bg = loss_fun(torch.unsqueeze(pred_support[:, 0, :, :], dim=1), sup_bg)
            loss_sup_label1 = loss_fun(torch.unsqueeze(pred_support[:, 1, :, :], dim=1), sup_la)
            loss_query_bg = loss_fun(torch.unsqueeze(pred_query[:, 0, :, :], dim=1), query_bg)
            loss_query_label1 = loss_fun(torch.unsqueeze(pred_query[:, 1, :, :], dim=1), query_la)

            if epoch < 10:
                loss_sup = (loss_sup_bg + loss_sup_label1) / 2
                loss_query = (loss_query_bg + loss_query_label1) / 2
                loss =  0.5*loss_sup + 0.5*loss_query
            else:
                loss = w1*loss_sup_label1 + w2*loss_query_label1

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
        Loss_sup = Los_sup / num_split
        Loss_query = Los_query / num_split
        Accuracy_sup = Acc_sup / num_split
        Accuracy_query = Acc_query / num_split
        # os.chdir(path_model)
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

            test_dataset = TaskDatasetFineTune(data_root=path, file_name=file_name, phase='test')
            test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

            Los_sup, Los_query, Acc_sup, Acc_query = 0, 0, 0, 0
            for m,(images,labels) in enumerate(test_data_loader):
                sup_img = torch.as_tensor(np.float32(images[0]))
                query_img = torch.as_tensor(np.float32(images[1]))
                sup_la = torch.as_tensor(np.float32(labels[0]))
                query_la = torch.as_tensor(np.float32(labels[1]))
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
                Los_sup += los_sup
                Los_query += los_query
                Acc_sup += sup_atrium_dice
                Acc_query += query_atrium_dice
            t_Loss_sup = Los_sup / num_split
            t_Loss_query = Los_query / num_split
            t_Accuracy_sup = Acc_sup / num_split
            t_Accuracy_query = Acc_query / num_split
            time3 = time.time()
            print('epoch {}, test total time {:.2f}'.format(epoch, time3 - time2))
            # data_test = [epoch, t_Loss_sup, t_Accuracy_sup, t_Loss_query, t_Accuracy_query]
            # file = path_model + 'test.h5'
            # with h5py.File(file, 'a') as h5f:
            #     save_h5(h5f, data=np.array(data_test), target='data')
            print('test: sup:', t_Loss_sup, t_Accuracy_sup, 'query:', t_Loss_query, t_Accuracy_query)
    return



if __name__ == '__main__':
    task = 'CT_LA'

    if task == 'CT_LA':
        file_name = 'CT_LA_128.pkl'
    elif task == 'MR_LV':
        file_name = 'MR_LV_128.pkl'
    elif task == 'MR_RV':
        file_name = 'MR_RV_128.pkl'
    elif task == 'MR_RA':
        file_name = 'MR_RA_128.pkl'

    path_data = '/.../'
    path_model = '/.../HRIN/othertasks/finetune/'+str(task)+'/'
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main(path_data, path_model, file_name, task=str(task), lr_train=0.0001, epochs=100, num_class=1)
