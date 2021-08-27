from __future__ import division
from __future__ import print_function

import os

from skimage import io
from torch.utils.data import Dataset
import random
import pandas as pd
import cv2
import numpy as np

class DogsCatsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.pics_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pics_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.pics_list[idx])
        target = 0 if "cat" in self.pics_list[idx] else 1 
        image = io.imread(img_name) 
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "target": target}

        return sample



class JAFFEDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.pics_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pics_list)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir, self.pics_list[idx])
        if "AN" in self.pics_list[idx]:
            target = 0
        elif "DI" in self.pics_list[idx]:
            target = 1
        elif "FE" in self.pics_list[idx]:
            target = 2
        elif "HA" in self.pics_list[idx]:
            target = 3
        elif "SA" in self.pics_list[idx]:
            target = 4
        elif "SU" in self.pics_list[idx]:
            target = 5
        elif "NE" in self.pics_list[idx]:
            target = 6
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "target": target}

        return sample



class RafDataset(Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        #txt標籤檔資料格式: train_00001.jpg 5, test_0001.jpg 5 檔名+空白+標籤
        NAME_COLUMN = 0 
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)#sep=' ' (separate)以空白分隔
        if phase == 'train': 
            dataset = df[df[NAME_COLUMN].str.startswith('train')] #dataset取開頭為train的資料
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')] #dataset取開頭為test的資料
        file_names = dataset.iloc[:, NAME_COLUMN].values #iloc取所有NAME_COLUMN的資料到file_names
        self.target = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # txt檔中標籤從1開始(1~7)，故減1 => 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0] #file_names格式為train_00001.jpg或test_0001.jpg, 使用split以 "."來分割資料，取[0]:train_00001 or test_00001
            f = f + "_aligned.jpg" #加上"_aligned.jpg" 來符合aligned資料夾圖片的格式 train_00001_aligned.jpg
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(image)
            
        sample = {"image":image, "target":target, "index":idx}
        return sample