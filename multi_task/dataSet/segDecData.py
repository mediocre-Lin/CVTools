"""
@File: dataSet.py
@Author: Keith 
@Time: 2022/03/25 16:13:57
@Contact: 956744413@qq.com
"""
from torch.utils.data.dataset import Dataset
import torch
import os
from  PIL import Image
import torchvision.transforms as transforms
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import albumentations as al
SEED = 2021
random.seed(SEED)
HEIGHT, WIDTH = 1024, 1024
al_transform = al.Compose([
    al.Flip(),
    al.Rotate(limit=90),
    al.OneOf([
        al.CenterCrop(height = HEIGHT, width = WIDTH),
        al.Resize(height = HEIGHT, width = WIDTH)
        ], p = 1),
    al.OneOf([
        al.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False),
        al.RandomContrast()
        ],p = 0.5)
])

class segDec_Dataset(Dataset):
    def __init__(self, img_path, label_path, trans = al_transform,mode='train'):
        self.img_path = img_path
        self.img_label = label_path
        self.transform = trans
        self.mode = mode
        self.to_tensor = transforms.Compose([            
                                        transforms.Resize([HEIGHT,WIDTH]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
        self.mask2tensor = transforms.Compose([            
                                        transforms.Resize([HEIGHT,WIDTH]),
                                        transforms.ToTensor()
                                    ])
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.img_label[index] == 1:
            mask = np.zeros(shape=[HEIGHT,WIDTH])
            mask = Image.fromarray(mask)
        else:
            mask_path = self.img_path[index].replace('img.png','label.png')
            mask = Image.open(mask_path).convert('P')

        if self.mode == 'train':
          img_np = np.array(img)
          mask_np = np.array(mask)/255.0
          augmented = self.transform(image=img_np,mask=mask_np)
          img = Image.fromarray(augmented['image'])
          mask = Image.fromarray(augmented['mask'])

        return self.to_tensor(img), self.mask2tensor(mask),torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_label)





def regular_path(imgs_paths):
    for idx in range(len(imgs_paths)):
        for num in range(1,10):
            imgs_paths[idx] = imgs_paths[idx].replace('\\'+str(num)+'.','\\')
    return imgs_paths
def count_imgs(imgs_path, category):
    img_set = []
    for path in imgs_path:
        for cate in category:
            if cate in path:
                img_set.append(path)
                break
    return img_set

def data_generate(data_path):
    defect_imgs = glob.glob(data_path,+'\\defect\\*\\img.png')
    undefect_imgs = glob.glob(data_path+'undefect\\*.bmp')
    defect_imgs = sorted(defect_imgs)
    undefect_imgs = sorted(undefect_imgs)
    imgs = defect_imgs + undefect_imgs
    label = [0 for i in range(len(defect_imgs))] + [1 for i in range(len(undefect_imgs))]
    x, y = np.array(imgs),np.array(label)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify= y,random_state = 42)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    return X_train,X_test,y_train,y_test