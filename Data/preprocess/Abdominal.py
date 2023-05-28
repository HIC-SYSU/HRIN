from PIL import Image
import os
import torch
import pickle
import numpy as np
from skimage import measure,color
import skimage
import skimage.morphology as sm


def img_InPhase(people_num):
    Imgs = []
    image_dir = '/.../' + people_num + '/T1DUAL/DICOM_anon/InPhase/'  # T1DUAL
    imgs = os.listdir(image_dir)
    for png in sorted(imgs):
        x = Image.open(image_dir + png)  # 打开图片
        img = np.asarray(x)
        Imgs.append(img)
    Imgs = np.array(Imgs)
    if Imgs.shape[1] > 256:
        x = int(Imgs.shape[1] / 2)
        y = int(Imgs.shape[2] / 2)
        Imgs = Imgs[:, (x - 128):(x + 128), (y - 128):(y + 128)]
    return Imgs


def img_OutPhase(people_num):
    Imgs = []
    image_dir = '/.../' + people_num + '/T1DUAL/DICOM_anon/OutPhase/'  # T1DUAL
    imgs = os.listdir(image_dir)
    for png in sorted(imgs):
        x = Image.open(image_dir + png)  # 打开图片
        img = np.asarray(x)
        Imgs.append(img)
    Imgs = np.array(Imgs)
    if Imgs.shape[1] > 256:
        x = int(Imgs.shape[1] / 2)
        y = int(Imgs.shape[2] / 2)
        Imgs = Imgs[:, (x - 128):(x + 128), (y - 128):(y + 128)]
    return Imgs



def img_T2(people_num):
    Imgs = []
    image_dir = '/.../' + people_num + '/T2SPIR/DICOM_anon/'  # T1DUAL
    imgs = os.listdir(image_dir)
    for png in sorted(imgs):
        x = Image.open(image_dir + png)  # 打开图片
        img = np.asarray(x)
        Imgs.append(img)
    Imgs = np.array(Imgs)
    if Imgs.shape[1] > 256:
        x = int(Imgs.shape[1] / 2)
        y = int(Imgs.shape[2] / 2)
        Imgs = Imgs[:, (x - 128):(x + 128), (y - 128):(y + 128)]
    return Imgs



def getBinaryTensor(img, boundary):
    one = np.ones_like(img)
    zero = np.zeros_like(img)
    return np.where(img == boundary, one, zero)   # 满足条件返回one, 不满足条件返回zero


# 输入一整个病人的label
def label_extract(pngs):
    Background, A1, A2, A3, A4 = [], [] ,[], [] ,[]
    for png in sorted(pngs):
        x = Image.open(png)  # 打开图片
        label = np.asarray(x)
        if label.shape[0]>256:
            x = int(label.shape[0] / 2)
            y = int(label.shape[1] / 2)
            label = label[(x-128):(x+128), (y-128):(y+128)]

        # 提出背景，pixel=0
        background = getBinaryTensor(label, boundary=0)
        Background.append(background)
        # 提出第1个label，pixel=63
        a1 = getBinaryTensor(label, boundary=63)
        A1.append(a1)
        # 提出第2个label，pixel=126
        a2 = getBinaryTensor(label, boundary=126)
        A2.append(a2)
        # 提出第3个label，pixel=189
        a3 = getBinaryTensor(label, boundary=189)
        A3.append(a3)
        # 提出第4个label，pixel=252
        a4 = getBinaryTensor(label, boundary=252)
        A4.append(a4)
    return [np.array(Background), np.array(A1), np.array(A2), np.array(A3), np.array(A4)]



if __name__ == '__main__':
    people_path = '/.../'
    people_s = os.listdir(people_path)
    list_people = []
    for s in sorted(people_s):  # InPhase or OutPhase
        if s[0] != ".":
            # 将原图转为数组
            infor_In = s + '_T1DUAL_InPhase'
            infor_Out = s + '_T1DUAL_OutPhase'
            infor_T2 = s + '_T2SPIR'  #D:\BaiduNetdiskDownload\MR_png\MR_png\8\T2SPIR\DICOM_anon
            img_one_people_In = img_InPhase(s)
            img_one_people_Out = img_OutPhase(s)
            img_one_people_T2 = img_T2(s)

            # 处理标签，将背景以及四个目标分开
            label_image_dir = '/.../' + s + '/T1DUAL/Ground/'  # T1DUAL
            os.chdir(label_image_dir)
            labels = os.listdir(label_image_dir)
            label_one_people = label_extract(labels)  # len(one_people)=5, 都是numpy,  one_people=[Background, A1, A2, A3, A4]

            label_image_dir = '/.../' + s + '/T2SPIR/Ground/'  # T1DUAL
            os.chdir(label_image_dir)
            labels = os.listdir(label_image_dir)
            label_one_people_T2 = label_extract(labels)  # len(one_people)=5, 都是numpy,  one_people=[Background, A1, A2, A3, A4]


            one_people_In = [infor_In, img_one_people_In, ] + label_one_people    # one_people = [病人信息，原图，背景，1，2，3，4]
            one_people_Out = [infor_Out, img_one_people_Out, ] + label_one_people
            one_people_T2 = [infor_T2, img_one_people_T2, ] + label_one_people_T2

            list_people.append(one_people_In)   # [0][2]是label
            list_people.append(one_people_Out)
            list_people.append(one_people_T2)

    path = '/.../'
    os.chdir(path)
    with open("Abdominal.pkl", 'wb') as fo:
        pickle.dump(list_people, fo)