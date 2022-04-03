"""
@File: coco_analyze.py
@Author: Keith 
@Time: 2022/04/01 10:28:22
@Contact: 956744413@qq.com
"""
from asyncio import as_completed
import numpy as np
from pycocotools.coco import COCO
from pyparsing import alphas
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(color_codes=True)#导入seaborn包设定颜色
def id2name(coco):
    classes = dict()
    classes_id = []
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']

    for key in classes.keys():
        classes_id.append(key)
    return classes, classes_id

def load_dataset(path):
    dataset = []
    coco = COCO(path)
    classes, classes_id = id2name(coco)
    print(classes)
    print('class_ids:', classes_id)

    img_ids = coco.getImgIds()
    print(len(img_ids))

    for imgId in img_ids:
        i = 0
        img = coco.loadImgs(imgId)[i]
        height = img['height']
        width = img['width']
        i = i + 1
        if imgId % 500 == 0:
            print('process {} images'.format(imgId))
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        # time.sleep(0.2)
        for ann in anns:
            if 'bbox' in ann:
                bbox = ann['bbox']
                '''
                    coco:
                annotation: [x, y, width, height] 
                '''
                ann_width = bbox[2]
                ann_height = bbox[3]
                area = ann_width * ann_height

                # 偏移量
                ann_width_ratio = np.float64(ann_width / width) * 100
                ann_height_ratio = np.float64(ann_height / height) * 100
                w_h_ratio = np.float64(ann_width / ann_height) * 100
                ann_area_ratio = ann_width_ratio * ann_height_ratio / 100
                dataset.append([ann_width_ratio, ann_height_ratio, w_h_ratio, ann_area_ratio])
            else:
                raise ValueError("coco no bbox -- wrong!!!")

    return np.array(dataset)
if __name__ == '__main__':
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小
    train_annFile = 'D:\pixdot\project_camera\\regular_data_two_classes\coco\\annotations\\instances_train2017.json'
    val_annFile = 'D:\pixdot\project_camera\\regular_data_two_classes\coco\\annotations\\instances_val2017.json'

    train_data = load_dataset(train_annFile)
    val_data = load_dataset(val_annFile)
    data = np.concatenate((train_data,val_data))
    print(data[:,3].max())
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
    sns.kdeplot(data[:,0],shade = True)
    sns.distplot(data[:,0], label='缺陷长占比', kde = False, color='#e74c3c', bins = 100, ax=axs[0][0])
    sns.distplot(data[:,1], label='缺陷宽占比', kde = False, color='#2ecc71', bins = 100, ax=axs[0][1])
    sns.distplot(data[:,2], label='缺陷长宽比', kde = False, color='#e74c3c', bins = 100, ax=axs[1][0])
    sns.distplot(data[:,3], label='缺陷面积占比',kde = False, color='red',bins = 100, ax=axs[1][1])
    axs[1][1].set(xlim = (0,10))
    axs[0][0].set_title('缺陷长占比分布', size=20, y=1.05)
    axs[0][1].set_title('缺陷宽占比分布', size=20, y=1.05)
    axs[1][0].set_title('缺陷长宽比分布', size=20, y=1.05)
    axs[1][1].set_title('缺陷面积占比分布', size=20, y=1.05)
    for i in range(2):
        axs[0][i].tick_params(axis='x', labelsize=20)
        axs[1][i].tick_params(axis='x', labelsize=20)
        axs[0][i].tick_params(axis='y', labelsize=20)
        axs[1][i].tick_params(axis='y', labelsize=20)
        axs[0][i].set_xlabel('')
        axs[1][i].set_xlabel('')
        axs[0][i].set_ylabel('')
        axs[1][i].set_ylabel('')
    plt.subplots_adjust(hspace=0.5)
    
    plt.savefig('res.png',dpi = 100)
    plt.show()