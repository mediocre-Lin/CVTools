"""
@File: labelme2seg.py
@Author: Keith 
@Time: 2022/04/01 16:05:00
@Contact: 956744413@qq.com
"""
import glob
import shutil
import os
import random
IMAGE_NAME = 'img.png'
LABEL_NAME = 'label.png'
PATH = 'D:\pixdot\project_camera\glass_defect_data\coco1_camera_defect\\train2017\\'
files = glob.glob(PATH+'*_json')

def labelme2seg(files):
    count = 1
    tarImg_dir = os.path.join(PATH + 'seg','JPEGImages')
    tarLabel_dir= os.path.join(PATH + 'seg','Annotations')
    for file in files:
        ori_img  = os.path.join(file,IMAGE_NAME)
        ori_label = os.path.join(file,LABEL_NAME)
        shutil.copy(ori_img,tarImg_dir + '\\' + str(count) + '.jpg')
        shutil.copy(ori_label,tarLabel_dir + '\\' + str(count) + '.png')
        count += 1
def maske_train_val_list(img_path):
    random.seed(0)
    test_size = 0.3
    data_nums = len(os.listdir(img_path))
    txt_list = []
    for i in range(1,data_nums + 1):
        txt_list.append(f'JPEGImages/{i}.jpg Annotations/{i}.png')
    random.shuffle(txt_list)
    return  txt_list[int(data_nums * test_size) + 1:],txt_list[:int(data_nums * test_size)]
# labelme2seg(files)
# os.mkdir(PATH + 'seg')
# os.mkdir(PATH + 'seg\\' + 'JPEGImages')
# os.mkdir(PATH + 'seg\\' + 'Annotations')
train_list, val_list = maske_train_val_list(PATH + 'seg\\' + 'JPEGImages')
with open(PATH + 'seg\\'+'train_list.txt','w') as tf:
    for train_info in train_list:
        tf.write(train_info+'\n')
    tf.close()
with open(PATH + 'seg\\'+'val_list.txt','w') as vf:
    for val_info in val_list:
        vf.write(val_info+'\n')
    vf.close()