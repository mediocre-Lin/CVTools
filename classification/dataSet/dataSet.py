from torch.utils.data.dataset import Dataset
import torch
from  PIL import Image
import torchvision.transforms as transforms
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
SEED = 2021
random.seed(SEED)

class camera_Dataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.img_label = label_path
        self.transform = transforms.Compose([
                                        transforms.Resize((1024, 1024)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img, torch.from_numpy(np.array(self.img_label[index]))

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
    # 镜面异常
    lens_anomaly =['IR/伤','p3/异物','p3/裂','p3/压伤','p3/划伤','p3/脏污','p3/膜裂','p3/脱膜',
                'p2/异物','p2/裂','p2/压伤','p2/划伤','p2/脏污','p2/膜裂','p2/脱膜',
                'p1/异物','p1/裂','p1/压伤','p1/划伤','p1/脏污','p1/膜裂','p1/脱膜'] 
    # 打胶异常
    glue_anomaly =['胶/'] 
    # 膜异常
    film_anomaly =['组装/挡光片歪斜','膜色异常','p3/膜欠']
    # 端面/镜室/底座  非成像异常
    no_imaging_anomaly = ['端面/','组装/花瓣装反','镜室/','底座/']
    # 崩边
    IR_edge_anomaly = ['IR\崩边']
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
    
    label = [0 for  i in range(len(lens_anomaly_imgs))] + [1 for  i in range(len(glue_anomaly_imgs))] + [2 for i in range(len(film_anomaly_imgs))] + [3 for i in range(len(no_imaging_anomaly_imgs))]+ [4 for i in range(len(IR_edge_anomaly_imgs))]
    pd_df = pd.DataFrame(columns=['path','label'])
    pd_df['path'] = np.array(data_all)
    pd_df['label'] = np.array(label)
    x,y = np.array(data_all),np.array(label)
    print(x[0])
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify= y,random_state = 42)
    '''打印各个数据集的形状'''
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    return X_train,X_test,y_train,y_test