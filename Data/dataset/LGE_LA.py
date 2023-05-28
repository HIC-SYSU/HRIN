from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import copy


class Abdominaldataset(Dataset):
    def __init__(self, data_root,num_batch,num_split):
        super(Abdominaldataset).__init__()
        self.num_batch = num_batch
        data_root = os.path.join(data_root,'Abdominal_MR_256.pkl')
        with open(data_root, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        self.data = data
        self.support_data = data[:num_split]
        self.query_data = data[num_split:]
        self.length_support = len(self.support_data)
        self.length_query = len(self.query_data)

        self.input_images, self.input_labels = [], []
        for i2 in range(self.length_query):
            i1 = np.random.randint(0, self.length_support - 1)
            images = [self.transform(self.support_data[i1][1]), self.transform(self.query_data[i2][1])]
            labels = [np.stack(self.support_data[i1][2:],axis=0), np.stack(self.query_data[i2][2:],axis=0)]
            self.input_images.append(images)
            self.input_labels.append(labels)

    def __getitem__(self, item):
        imgs = self.input_images[item]
        labels = self.input_labels[item]
        return imgs, labels

    def __len__(self):
        return self.num_batch

    def transform(self,images):
        for ii in range(len(images)):
            tims = copy.copy(images[ii])
            tims = (tims + np.abs(tims)) / 2.0
            tims = (tims - np.min(np.min(np.min(tims)))) * 1.00 / (np.max(np.max(np.max(tims))) - np.min(np.min(np.min(tims))))
            images[ii] = tims
        return images



class LADataSetFineTune(Dataset):
    def __init__(self,data_root,file_name,rate, num_split, phase):
        super(LADataSetFineTune).__init__()

        self.phase = phase
        file_root = os.path.join(data_root, file_name)
        with open(file_root, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        Imgs = data[0]
        Labels = data[1]
        num_rate = int(len(Imgs) * rate)
        self.num_rate = num_split if num_rate > num_split else num_rate

        self.train_Imgs = Imgs[:self.num_rate]
        self.train_labels = Labels[:self.num_rate]
        self.test_Imgs = Imgs[num_split:]
        self.test_labels = Labels[num_split:]

        self.input_images, self.input_labels = [], []
        if self.phase == 'train':
            self.num_batch = len(self.train_Imgs) - int(self.num_rate * 0.5)
            l11 = int(len(self.train_Imgs) / 2)
            l1 = len(self.train_Imgs)
            for i2 in range(l11,l1):
                i1 = np.random.randint(0, l11)
                images = [self.transform(self.train_Imgs[i1]), self.transform(self.train_Imgs[i2])]
                labels = [self.train_labels[i1], self.train_labels[i2]]
                self.input_images.append(images)
                self.input_labels.append(labels)
        elif self.phase == 'test':
            self.num_batch = len(self.test_Imgs)
            l1 = int(len(self.train_Imgs))
            l2 = int(len(self.test_Imgs))
            for i2 in range(l2):
                i1 = np.random.randint(0, l1-1)
                images = [self.transform(self.train_Imgs[i1]), self.transform(self.test_Imgs[i2])]
                labels = [self.train_labels[i1], self.test_labels[i2]]
                self.input_images.append(images)
                self.input_labels.append(labels)

    def __getitem__(self, item):
        imgs = self.input_images[item]
        labels = self.input_labels[item]
        return imgs,labels

    def __len__(self):
        return self.num_batch

    def transform(self, images):
        tims = copy.copy(images)
        tims = (tims + np.abs(tims)) / 2.0
        tims = (tims - np.min(np.min(np.min(tims)))) * 1.00 / (np.max(np.max(np.max(tims))) - np.min(np.min(np.min(tims))))
        images = tims
        return images




