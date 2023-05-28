import torch
import cv2
import os
import nibabel as nib
import pickle
from PIL import Image
import numpy as np
from skimage import measure,color
import skimage
import skimage.morphology as sm
import copy
import torchvision.transforms as transforms



def transform_sample(imgs):
    imgs_list = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        out = transforms.Resize(size=[128, 128])(Image.fromarray(img))
        # out = transforms.functional.to_tensor(out)
        # out = self.normalize(out)
        imgs_list.append(out)
    out = np.dstack(imgs_list)
    out = np.transpose(out)
    return out


def trans_resize(input):
    imgs_list = []
    shapes = input.shape
    for i in range(shapes[0]):
        # print('before--', input[i].shape)
        if shapes[1] < shapes[2]:  # 512
            m = shapes[2] - shapes[1]
            n = int(m / 2)
            img = np.pad(input[i, :, :], ((n, n), (0, 0)), 'constant', constant_values=(0, 0))  ##补0
            # print('after---', img.shape)
        elif shapes[1] > shapes[2]:  # 288
            m = shapes[1] - shapes[2]
            n = int(m / 2)
            img = np.pad(input[i, :, :], ((0, 0), (n, n)), 'constant', constant_values=(0, 0))  ##补0
            # print('after---', img.shape)
        elif shapes[2] ==135:  # 288
            m = shapes[1] - shapes[2]
            n = int(m / 2)
            img = np.pad(input[i, :, :], ((0, 0), (n-1, n)), 'constant', constant_values=(0, 0))  ##补0
        ##resize
        out = transforms.Resize(size=[128, 128])(Image.fromarray(img))
        imgs_list.append(out)
    out = np.dstack(imgs_list)
    out = np.transpose(out)
    return out

def getBinaryTensor(img, boundary):
    one = np.ones_like(img)
    zero = np.zeros_like(img)
    return np.where(img == boundary, one, zero)   # 满足条件返回one, 不满足条件返回zero

def la_extract(patient, boundary):
    LA = []
    for slice in patient:
        # x = Image.open(png)  # 打开图片
        label = np.asarray(slice)
        a1 = getBinaryTensor(label, boundary=boundary)
        LA.append(a1)
    return np.array(LA)

def label_extract(patient):
    LV, RA, RV = [], [] ,[]
    for slice in patient:
        # x = Image.open(png)  # 打开图片
        label = np.asarray(slice)
        a1 = getBinaryTensor(label, boundary=500)
        LV.append(a1)
        a2 = getBinaryTensor(label, boundary=550)
        RA.append(a2)
        a3 = getBinaryTensor(label, boundary=600)
        RV.append(a3)
    return [np.array(LV), np.array(RA), np.array(RV)]


def label_extract_CT(patient):
    bg, LV, RA, RV = [], [] ,[],[]
    for slice in patient:
        # x = Image.open(png)  # 打开图片
        label = np.asarray(slice)
        # 提出第1个label，pixel=63
        a1 = getBinaryTensor(label, boundary=500)
        LV.append(a1)
        # 提出第2个label，pixel=126
        a2 = getBinaryTensor(label, boundary=550)
        RA.append(a2)
        # 提出第3个label，pixel=189
        a3 = getBinaryTensor(label, boundary=600)
        RV.append(a3)
        a00 = a1 + a2 + a3
        a0 = np.ones_like(a1) - a00
        bg.append(a0)
        m = np.array(bg)
    return [np.array(bg), np.array(LV), np.array(RA), np.array(RV)]


def label_extract_ult(patient):
    bg, Wall, RV =  [] ,[], []
    for slice in patient:
        # x = Image.open(png)  # 打开图片
        label = np.asarray(slice)
        # if label.shape[0]>256:
        #     x = int(label.shape[0] / 2)
        #     y = int(label.shape[1] / 2)
        #     label = label[(x-128):(x+128), (y-128):(y+128)]

        # 提出第1个label，pixel=63
        a1 = getBinaryTensor(label, boundary=0)
        bg.append(a1)
        # 提出第2个label，pixel=126
        a2 = getBinaryTensor(label, boundary=2)

        Wall.append(a2)
        # 提出第3个label，pixel=189
        a3 = getBinaryTensor(label, boundary=1)
        RV.append(a3)
    return [np.array(bg), np.array(Wall), np.array(RV)]


def z_roi(data): #
    temp=data
    shapes = temp.shape
    low=0
    height=shapes[0]
    for i in range(shapes[0]):
        if np.max(temp[i,:,:])>0:
            low=i
            break
    for i in range(shapes[0]-1,low, -1):
        if np.max(temp[i, :, :]) > 0:
            height = i
            break
    return low, height

def z_roi_2(labels): #
    temp_1 = labels[1]
    temp_2 = labels[2]
    temp_3 = labels[3]
    shapes = temp_1.shape
    # low_1=0
    # height=shapes[0]

    for i in range(shapes[0]):
        if np.max(temp_1[i,:,:])>0:
            low_1=i
            break
    for i in range(shapes[0]-1,low_1, -1):
        if np.max(temp_1[i, :, :]) > 0:
            height_1 = i
            break

    for i in range(shapes[0]):
        if np.max(temp_2[i,:,:])>0:
            low_2=i
            break
    for i in range(shapes[0]-1,low_2, -1):
        if np.max(temp_2[i, :, :]) > 0:
            height_2 = i
            break


    for i in range(shapes[0]):
        if np.max(temp_3[i,:,:])>0:
            low_3=i
            break
    for i in range(shapes[0]-1,low_3, -1):
        if np.max(temp_3[i, :, :]) > 0:
            height_3 = i
            break

    low = min(low_1, low_2, low_3)
    height = max(height_1, height_2, height_3)

    return low, height

def z_roi_3(labels): #
    temp_1 = labels[1]
    temp_2 = labels[2]
    temp_3 = labels[3]
    shapes = temp_1.shape
    # low_1=0
    # height=shapes[0]

    for i in range(shapes[0]):
        if np.max(temp_1[i,:,:])>0:
            low_1=i
            break
    for i in range(shapes[0]-1,low_1, -1):
        if np.max(temp_1[i, :, :]) > 0:
            height_1 = i
            break

    for i in range(shapes[0]):
        if np.max(temp_2[i,:,:])>0:
            low_2=i
            break
    for i in range(shapes[0]-1,low_2, -1):
        if np.max(temp_2[i, :, :]) > 0:
            height_2 = i
            break


    for i in range(shapes[0]):
        if np.max(temp_3[i,:,:])>0:
            low_3=i
            break
    for i in range(shapes[0]-1,low_3, -1):
        if np.max(temp_3[i, :, :]) > 0:
            height_3 = i
            break

    low = min(low_1, low_2, low_3)
    height = max(height_1, height_2, height_3)

    return low, height

def xy_roi(data):
    temp=data
    shapes=temp.shape

    xl=0
    xh=shapes[1]

    yl=0
    yh=shapes[2]


    for i in range(shapes[1]):
        if np.max(temp[:, i, :]) > 0:
            xl = i
            break
    for i in range(shapes[1]-1, xl, -1):
        if np.max(temp[:, i, :]) > 0:
            xh = i
            break

    for i in range(shapes[2]):
        if np.max(temp[:, :, i]) > 0:
            yl = i
            break
    for i in range(shapes[2]-1, yl, -1):
        if np.max(temp[:, :, i]) > 0:
            yh = i
            break

    mean_x= (xl+xh)//2
    mean_y= (yl+yh)//2

    xl=mean_x-128
    xh=mean_x+128

    yl=mean_y-128
    yh=mean_y+128

    if mean_x-128<0:
        xl=0
        xh=256
    if mean_x+128>512:
        xl=256
        xh=512

    if mean_y-128<0:
        yl=0
        yh=256
    if mean_y+128>512:
        yl=256
        yh=512
    return xl,xh,yl,yh


def xy_roi_2(data):
    temp=data
    shapes=temp.shape

    xl=0
    xh=shapes[1]

    yl=0
    yh=shapes[2]


    for i in range(shapes[1]):
        if np.max(temp[:, i, :]) > 0:
            xl = i
            break
    for i in range(shapes[1]-1, xl, -1):
        if np.max(temp[:, i, :]) > 0:
            xh = i
            break

    for i in range(shapes[2]):
        if np.max(temp[:, :, i]) > 0:
            yl = i
            break
    for i in range(shapes[2]-1, yl, -1):
        if np.max(temp[:, :, i]) > 0:
            yh = i
            break

    mean_x= (xl+xh)//2
    mean_y= (yl+yh)//2

    xl=mean_x-128
    xh=mean_x+128

    yl=mean_y-128
    yh=mean_y+128

    if mean_x-128<0:
        xl=0
        xh=256
    if mean_x+128>shapes[2]:
        xl=shapes[2] - 256
        xh=shapes[2]

    if mean_y-128<0:
        yl=0
        yh=256
    if mean_y+128>shapes[2]:
        yl=shapes[2] - 256
        yh=shapes[2]
    return xl,xh,yl,yh


def xy_roi_3(data):
    shapes=data[0].shape
    temp_1 = data[0]
    temp_2 = data[1]
    temp_3 = data[2]

    xl=0
    xh=shapes[1]  # H

    yl=0
    yh=shapes[2]  # W


    for i in range(shapes[1]):
        if np.max(temp_1[:, i, :]) > 0 or np.max(temp_2[:, i, :]) > 0 or np.max(temp_3[:, i, :]) > 0:
            xl = i
            break
    for i in range(shapes[1]-1, xl, -1):
        if np.max(temp_1[:, i, :]) > 0 or np.max(temp_2[:, i, :]) > 0 or np.max(temp_3[:, i, :]) > 0:
            xh = i
            break

    for i in range(shapes[2]):
        if np.max(temp_1[:, :, i]) > 0 or np.max(temp_2[:, :, i]) > 0 or np.max(temp_3[:, :, i]) > 0:
            yl = i
            break
    for i in range(shapes[2]-1, yl, -1):
        if np.max(temp_1[:, :, i]) > 0 or np.max(temp_2[:, :, i]) > 0 or np.max(temp_3[:, :, i]) > 0:
            yh = i
            break

    mean_x= (xl+xh)//2
    mean_y= (yl+yh)//2

    xl=mean_x-144
    xh=mean_x+144

    yl=mean_y-64
    yh=mean_y+64
    return xl,xh,yl,yh

def ROI_2(Imgs,Epis): #LA
    temp_img=[]
    temp_epi=[]
    for i in range(len(Imgs)):
        temp=Epis[i]
        shapes=temp.shape

        low,height = z_roi(temp)
        # xl, xh, yl, yh =xy_roi(temp)
        temp_img.append(Imgs[i][low:height,:, :])
        temp_epi.append(Epis[i][low:height,:, :])
        print('img:', (Imgs[i][low:height,:, :]).shape)
    return temp_img,temp_epi

def ROI_3(Imgs,Epis): #LA
    temp_img=[]
    temp_epi=[]
    for i in range(len(Imgs)):
        temp=Epis[i]
        shapes=temp.shape

        low,height = z_roi(temp)
        img = Imgs[i][low:height, :, :]
        epi = Epis[i][low:height, :, :]
        img = trans_resize(img)
        label_0 = trans_resize(epi)
        temp_img.append(img)
        temp_epi.append(label_0)
        print('img:', img.shape)
    return temp_img,temp_epi

def ROI_4(Imgs,labels):  #MRIs  LV RA RV
    temp_img=[]
    temp_bg = []
    temp_epi=[]
    temp_endo=[]
    temp_wall=[]
    for i in range(len(Imgs)):
        temp=labels[i][0]
        shapes=temp.shape
        print('img-0:', Imgs[i].shape)
        low,height = z_roi_3(labels[i])

        img = Imgs[i][low:height, :, :]
        label_0 = labels[i][0][low:height, :, :]
        label_1 = labels[i][1][low:height, :, :]
        label_2 = labels[i][2][low:height, :, :]
        label_3 = labels[i][3][low:height, :, :]
        img = trans_resize(img)
        label_0 = trans_resize(label_0)
        label_1 = trans_resize(label_1)
        label_2 = trans_resize(label_2)
        label_3 = trans_resize(label_3)
        temp_img.append(img)
        temp_bg.append(label_0)
        temp_epi.append(label_1)
        temp_endo.append(label_2)
        temp_wall.append(label_3)
        print('img-1:', img.shape)
    return temp_img,[temp_bg, temp_epi,temp_endo,temp_wall]

def ROI_5(Imgs,labels):  #CTs:LV RA RV
    temp_img=[]
    temp_bg = []
    temp_epi=[]
    temp_endo=[]
    temp_wall=[]
    for i in range(len(Imgs)):
        # temp=labels[i][0]
        # shapes=temp.shape
        low,height = z_roi_2(labels[i])
        # xl, xh, yl, yh =xy_roi(temp)
        print('img-0:', (Imgs[i]).shape)
        temp_img.append(Imgs[i][low:height,:, :])
        temp_bg.append(labels[i][0][low:height, :, :])
        temp_epi.append(labels[i][1][low:height,:, :])
        temp_endo.append(labels[i][2][low:height,:, :])
        temp_wall.append(labels[i][3][low:height,:, :])
        print('img-1:', (Imgs[i][low:height,:, :]).shape)
    return temp_img,[temp_bg, temp_epi,temp_endo,temp_wall]

## CT的la数据
def CT():
    path_data = '/.../impor_CT/'
    path_save = '/.../'

    # 处理思路：nii转为数组，提取出la，原图和la存为pkl
    folder_list = os.listdir(path_data)
    list_CT_image, list_CT_label = [], []
    for folder in folder_list:
        file_list = os.listdir(path_data + folder)
        list.sort(file_list)
        for file in file_list:
            if 'image.nii' in file:
                nii_img = nib.load(path_data + folder + '/' + file)
                array_img = (nii_img.get_fdata()).transpose(2,0,1)
                tims = copy.copy(array_img)
                tims = (tims + np.abs(tims)) / 2.0
                tims = (tims - np.min(np.min(np.min(tims)))) * 1.00 / (np.max(np.max(np.max(tims))) - np.min(np.min(np.min(tims))))
                array_img = tims
                array_img = transform_sample(array_img)
                # print('array_img:', array_img.shape)
                list_CT_image.append(array_img)
            elif 'label.nii' in file:
                nii_label = nib.load(path_data + folder + '/' + file)
                array_la = (nii_label.get_fdata()).transpose(2,0,1)
                # array_la = np.resize(array_la, (256, 256))
                label_la = la_extract(array_la, boundary=420)
                label_la = transform_sample(label_la)
                list_CT_label.append(label_la)
    print('ct:', len(list_CT_image), len(list_CT_label))

    list_CT_image, list_CT_label = ROI_2(list_CT_image, list_CT_label)
    os.chdir(path_save)
    with open("CT_128.pkl", 'wb') as fo:
        pickle.dump([list_CT_image, list_CT_label], fo)
    return
## CT其他器官
def CTs():
    path_data = '/.../impor_CT/'
    path_save = '/.../'

    # 处理思路：nii转为数组，提取出la，原图和la存为pkl
    folder_list = os.listdir(path_data)
    list_CT_image, list_CT_labels = [], []
    for folder in folder_list:
        file_list = os.listdir(path_data + folder)
        list.sort(file_list)
        for file in file_list:
            if 'image.nii' in file:
                nii_img = nib.load(path_data + folder + '/' + file)
                array_img = (nii_img.get_fdata()).transpose(2,0,1)
                tims = copy.copy(array_img)
                tims = (tims + np.abs(tims)) / 2.0
                tims = (tims - np.min(np.min(np.min(tims)))) * 1.00 / (np.max(np.max(np.max(tims))) - np.min(np.min(np.min(tims))))
                array_img = tims
                array_img = transform_sample(array_img)
                # print('array_img:', array_img.shape)
                list_CT_image.append(array_img)
            elif 'label.nii' in file:
                nii_label = nib.load(path_data + folder + '/' + file)
                array_la = (nii_label.get_fdata()).transpose(2,0,1)
                array_la = transform_sample(array_la)
                # array_la = np.resize(array_la, (256, 256))
                label_la = label_extract_CT(array_la)
                list_CT_labels.append(label_la)
    print('ct:', len(list_CT_image), len(list_CT_labels))

    list_CT_image, list_CT_labels = ROI_5(list_CT_image, list_CT_labels)
    os.chdir(path_save)
    with open("CTs_128.pkl", 'wb') as fo:
        pickle.dump([list_CT_image, list_CT_labels], fo)
    return



## 其他器官的MRI数据
def MRs():
    path_data = '/.../impor_MR/mr_train/'
    path_save = '/.../'

    # 处理思路：nii转为数组，提取出la，原图和la存为pkl
    list_MR_image, list_MR_labels = [], []
    file_list = os.listdir(path_data)
    list.sort(file_list)
    i=0
    ii = 0
    for file in file_list:
        if 'image.nii' in file:
            i = i + 1
            nii_img = nib.load(path_data + '/' + file)
            if i in [1,2,9,12,13,14]:
                array_img = (nii_img.get_fdata()).transpose(1,2,0)
            elif i in [3,4,5,6,7,8,10,11,15,16,17,18,19,20]:
                array_img = (nii_img.get_fdata()).transpose(1,0,2)
            list_MR_image.append(array_img)
        elif 'label.nii' in file:
            ii = ii +1
            nii_label = nib.load(path_data + '/' + file)
            # print('before--', (nii_label.get_fdata()).shape)
            if ii in [1,2,9,12,13,14]:
                array_labels = (nii_label.get_fdata()).transpose(1,2,0)
                # print('after---', array_labels.shape)
            elif ii in [3,4,5,6,7,8,10,11,15,16,17,18,19,20]:
                array_labels = (nii_label.get_fdata()).transpose(1,0,2)
                # print('after---', array_labels.shape)
            labels = label_extract_CT(array_labels)
            # print('labels:', len(labels))
            list_MR_labels.append(labels)

    print('mr:', len(list_MR_image), len(list_MR_labels))
    list_MR_image, list_MR_labels = ROI_4(list_MR_image, list_MR_labels)
    os.chdir(path_save)
    with open("MRs_128.pkl", 'wb') as fo:
        pickle.dump([list_MR_image, list_MR_labels], fo)  #无bg
    return


def MR(boundary):
    path_data = '/.../impor_MR/mr_train/'
    path_save = '/.../'

    # 处理思路：nii转为数组，提取出la，原图和la存为pkl
    list_MR_image, list_MR_label = [], []
    file_list = os.listdir(path_data)
    list.sort(file_list)
    n, m = 0, 0
    for file in file_list:
        if 'image.nii' in file:
            n=n+1
            nii_img = nib.load(path_data + '/' + file)
            if n in [1,2,9,12,13,14]:
                array_img = (nii_img.get_fdata()).transpose(1,2,0)
            elif n in [3,4,5,6,7,8,10,11,15,16,17,18,19,20]:
                array_img = (nii_img.get_fdata()).transpose(1,0,2)
            list_MR_image.append(array_img)
        elif 'label.nii' in file:
            m=m+1
            nii_label = nib.load(path_data + '/' + file)
            if m in [1, 2, 9, 12, 13, 14]:
                array_label = (nii_label.get_fdata()).transpose(1, 2, 0)
            elif m in [3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 17, 18, 19, 20]:
                array_label = (nii_label.get_fdata()).transpose(1, 0, 2)
            label_la = la_extract(array_label, boundary=boundary)
            list_MR_label.append(label_la)
    print('mr:', len(list_MR_image), len(list_MR_label))
    list_MR_image, list_MR_label = ROI_3(list_MR_image, list_MR_label)
    os.chdir(path_save)
    with open("MRRV_128.pkl", 'wb') as fo:
        pickle.dump([list_MR_image, list_MR_label], fo)  #无bg
    return


if __name__ == '__main__':
    CT()
    CTs()
    MRs()
    MR(boundary=600)  #LV=500,RA=550,RV=600