"""
@File: coco_visualize.py
@Author: Keith 
@Time: 2022/03/28 13:26:18
@Contact: 956744413@qq.com
"""

import json
import os
import cv2
import glob
# parent_path = 'D:\pixdot\project_camera\\regular_data_two_classes\coco\\val2017'
# json_file = 'D:\pixdot\project_camera\\regular_data_two_classes\coco\\annotations\instances_val2017.json' # 目标检测生成的文件
# img_paths = glob.glob(parent_path+'/*.jpg')
# with open(json_file) as annos:
#     annotations = json.load(annos)
# imgs_info = annotations['images']
# annotations_info = annotations['annotations']

# id = 2
# img_path = os.path.join(parent_path,imgs_info[id]['file_name'])
# image = cv2.imread(img_path)
# print(img_path)
# for anno in annotations_info:
#     if anno['image_id'] == id:
#         bbox = anno['bbox'] # (x1, y1, w, h)
#         x, y, w, h = bbox
#         image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2) 
# # cv2.imwrite('./res.jpg',image)
# 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
# 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
# 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
import matplotlib.pyplot as plt
img_path ='D:\pixdot\project_camera\\regular_data_two_classes\coco_Dataset\\train2017\\Image_20220315172932317.bmp'
image = cv2.imread(img_path)
bbox = [
                634.0,
                220.0,
                703.0,
                710.0
            ] # (x1, y1, w, h)
x, y, w, h = bbox
image = image[int(y):int(y+h),int(x):int(x+w)]
bbox =[179.0, 365.0, 137.0, 70.0] # (x1, y1, w, h)
x, y, w, h = bbox
image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2) 
# cv2.imwrite('./res.jpg',image)
plt.imshow(image)
plt.show()
# cv2.namedWindow("res", cv2.WINDOW_AUTOSIZE)
# cv2.imshow('demo', anno_image)
# cv2.waitKey(5000)
