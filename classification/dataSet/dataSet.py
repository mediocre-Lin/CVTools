from torch.utils.data.dataset import Dataset
import torch
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

class camera_Dataset(Dataset):
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
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.mode == 'train':
          img_np = np.array(img)
          augmented = self.transform(image=img_np)
          img = Image.fromarray(augmented['image'])
        return self.to_tensor(img), torch.from_numpy(np.array(self.img_label[index]))

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
    # ????????????
    lens_anomaly =['IR/???','p3/??????','p3/???','p3/??????','p3/??????','p3/??????','p3/??????','p3/??????',
                'p2/??????','p2/???','p2/??????','p2/??????','p2/??????','p2/??????','p2/??????',
                'p1/??????','p1/???','p1/??????','p1/??????','p1/??????','p1/??????','p1/??????'] 
    # ????????????
    glue_anomaly =['???/'] 
    # ?????????
    film_anomaly =['??????/???????????????','????????????','p3/??????']
    # ??????/??????/??????  ???????????????
    no_imaging_anomaly = ['??????/','??????/????????????','??????/','??????/']
    # ??????
    IR_edge_anomaly = ['IR\??????']
    set1 = glob.glob(str(data_path)+ '/*/*/*/*/*.bmp')
    set2 = glob.glob(str(data_path)+ '/*/*/*/*/*/*.bmp')
    set3 = glob.glob(str(data_path)+ '/*/*/*/*/*/*/*.bmp')

    imgs_paths = list(set(set1) | set(set2))
    imgs_paths = list(set(imgs_paths) | set(set3))
    lens_anomaly_imgs = count_imgs(imgs_paths, lens_anomaly)
    glue_anomaly_imgs = count_imgs(imgs_paths, glue_anomaly)
    film_anomaly_imgs = count_imgs(imgs_paths, film_anomaly)
    no_imaging_anomaly_imgs = count_imgs(imgs_paths, no_imaging_anomaly)
    IR_edge_anomaly_imgs = count_imgs(imgs_paths, IR_edge_anomaly)

    lens_anomaly_imgs = sorted(lens_anomaly_imgs)
    glue_anomaly_imgs = sorted(glue_anomaly_imgs)
    film_anomaly_imgs = sorted(film_anomaly_imgs)
    no_imaging_anomaly_imgs = sorted(no_imaging_anomaly_imgs)
    IR_edge_anomaly_imgs = sorted(IR_edge_anomaly_imgs)

    data_all = lens_anomaly_imgs + glue_anomaly_imgs+film_anomaly_imgs+ no_imaging_anomaly_imgs+IR_edge_anomaly_imgs
    
    label = [0 for  i in range(len(lens_anomaly_imgs))] + [1 for  i in range(len(glue_anomaly_imgs))] + [1 for i in range(len(film_anomaly_imgs))] + [1 for i in range(len(no_imaging_anomaly_imgs))]+ [1 for i in range(len(IR_edge_anomaly_imgs))]
    pd_df = pd.DataFrame(columns=['path','label'])
    pd_df['path'] = np.array(data_all)
    pd_df['label'] = np.array(label)
    x,y = np.array(data_all),np.array(label)
    print(x[0])
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify= y,random_state = 42)
    '''??????????????????????????????'''
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    return X_train,X_test,y_train,y_test