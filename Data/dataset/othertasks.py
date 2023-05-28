from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import copy

class TaskDataset(Dataset):
    def __int__(self ,data_root ,file_name):
        super(TaskDataset).__init__()
        file_root = os.path.join(data_root ,file_name)
        with open(file_root, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        Imgs = data[0]
        Labels = data[1]
        self.images = Imgs
        self.labels = Labels

        self.num_split = int(len(self.images)*0.5)
        self.train_Imgs = Imgs[:self.num_split]
        self.train_labels = [Labels[0][:self.num_split], Labels[1][:self.num_split], Labels[2][:self.num_split], Labels[3][:self.num_split]]
        self.test_Imgs = Imgs[self.num_split:]
        self.test_labels = [Labels[0][self.num_split:], Labels[1][self.num_split:], Labels[2][self.num_split:], Labels[3][self.num_split:]]

    def transform(self, images):
        tims = copy.copy(images)
        tims = (tims + np.abs(tims)) / 2.0
        tims = (tims - np.min(np.min(np.min(tims)))) * 1.00 / (np.max(np.max(np.max(tims))) - np.min(np.min(np.min(tims))))
        images = tims
        return images

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        label = np.stack(label ,axis=0)
        img = self.transform(img)
        return img ,label

    def __len__(self):
        return self.num_split



class TaskDatasetFineTune(Dataset):
    def __int__(self ,data_root ,file_name,phase):
        super(TaskDatasetFineTune).__init__()
        self.phase = phase
        self.num_batch = 10
        file_root = os.path.join(data_root, file_name)
        with open(file_root, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        Imgs = data[0]
        Labels = data[1]

        self.train_Imgs = Imgs[:self.num_batch]
        self.train_labels = Labels[:self.num_batch]
        self.test_Imgs = Imgs[self.num_batch:]
        self.test_labels = Labels[self.num_batch:]

        self.input_images, self.input_labels = [], []
        if self.phase == 'train':
            self.num_batch = len(self.train_Imgs) - int(self.num_batch * 0.5)
            l11 = int(len(self.train_Imgs) / 2)
            l1 = len(self.train_Imgs)
            for i2 in range(l11, l1):
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
                i1 = np.random.randint(0, l1 - 1)
                images = [self.transform(self.train_Imgs[i1]), self.transform(self.test_Imgs[i2])]
                labels = [self.train_labels[i1], self.test_labels[i2]]
                self.input_images.append(images)
                self.input_labels.append(labels)

    def __getitem__(self, item):
        imgs = self.input_images[item]
        labels = self.input_labels[item]
        return imgs, labels

    def __len__(self):
        return self.num_batch

    def transform(self, images):
        tims = copy.copy(images)
        tims = (tims + np.abs(tims)) / 2.0
        tims = (tims - np.min(np.min(np.min(tims)))) * 1.00 / (np.max(np.max(np.max(tims))) - np.min(np.min(np.min(tims))))
        images = tims
        return images