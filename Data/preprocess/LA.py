import os
import torch
import numpy as np
import nibabel as nib
import pickle




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

def ROI(Imgs,Epis,Endos,Walls,Scars):
    temp_img=[]
    temp_epi=[]
    temp_endo=[]
    temp_wall=[]
    temp_scar=[]
    for i in range(len(Imgs)):
        temp=Epis[i]
        shapes=temp.shape
        low,height = z_roi(temp)
        xl, xh, yl, yh =xy_roi(temp)
        temp_img.append(Imgs[i][low:height,xl:xh, yl:yh])
        temp_epi.append(Epis[i][low:height,xl:xh, yl:yh])
        temp_endo.append(Endos[i][low:height,xl:xh, yl:yh])
        temp_wall.append(Walls[i][low:height,xl:xh, yl:yh])
        temp_scar.append(Scars[i][low:height,xl:xh, yl:yh])
    return temp_img,temp_epi,temp_endo,temp_wall,temp_scar


def nii_to_list(filepath):
    Imgs, Epis, Endos, Walls, Scars = [], [], [], [], []
    filenames = os.listdir(filepath)  # 读取nii文件夹
    for filename in range(len(filenames)):
        files = os.listdir(filepath + '/' + str(filename) + '/' + str(filename) + '/')
        for file in files:
            niis = os.listdir(filepath + '/' + str(filename) + '/' + str(filename) + '/' +str(file) + '/')
            for nii in niis:
                # 开始读取nii文件
                img_path = os.path.join(filepath + '/' + str(filename) + '/' + str(filename) + '/' +str(file) + '/', nii)
                img = nib.load(img_path)
                img_data = img.get_data()
                img_data = np.reshape(img_data, [-1, 256, 256])
                if 'LGE' in nii:
                    Imgs.append(img_data)
                elif 'Epi' in nii:
                    Epis.append(img_data)
                elif 'Endo' in nii:
                    Endos.append(img_data)
                elif 'Wall' in nii:
                    Walls.append(img_data)
                elif 'scar' in nii:
                    Scars.append(img_data)
    return Imgs,Epis,Endos,Walls,Scars



filepath = '.../'
Imgs,Epis,Endos,Walls,Scars = nii_to_list(filepath)
temp_img,temp_epi,temp_endo,temp_wall,temp_scar=ROI(Imgs,Epis,Endos,Walls,Scars)
CARMA_Imgs,CARMA_Epis,CARMA_Endos,CARMA_Walls,CARMA_Scars = temp_img,temp_epi,temp_endo,temp_wall,temp_scar

with open("C2.pkl", 'wb') as fo:
    pickle.dump([CARMA_Imgs,CARMA_Epis,CARMA_Endos,CARMA_Walls,CARMA_Scars], fo)
