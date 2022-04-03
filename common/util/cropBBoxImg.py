'''
@File: cropBBoxImg
@Author: Keith 
@Time: 2022/03/28 19:29:54
@Contact: 956744413@qq.com

'''
import json, os
from PIL import Image

def reset_info(anno,crop_info):
    x, y = crop_info[0], crop_info[1]
    segment_infos = anno['segmentation']
    bbox_info = anno['bbox']
    bbox_info[0] -= x
    bbox_info[1] -= y
    for segment_info in segment_infos:
        for i in range(len(segment_info)):
            if i % 2 == 0:
                segment_info[i] -= x
            else:
                segment_info[i] -= y
    anno['segmentation'] = segment_infos
    anno['bbox'] = bbox_info
    return anno

res_json_path = "D:\pixdot\project_camera\\regular_data_two_classes\coco_Dataset\\annotations\instances_val2017.json"
camera_json_path = 'D:\pixdot\project_camera\\regular_data_two_classes\\val.json'

with open(res_json_path) as f: 
    res_json = json.load(f)
with open(camera_json_path) as cf: 
    camera_json = json.load(cf)

new_annos = []
annos_info = res_json['annotations']
camera_annos_info = camera_json['annotations']

for camera_info in camera_annos_info:
    img_id = camera_info['image_id']
    crop_info = camera_info['bbox']
    for anno_info in annos_info:
        if anno_info['image_id'] == img_id and anno_info['category_id'] == 1:
            new_annos.append(reset_info(anno_info,crop_info))
res_json['annotations'] = new_annos
with open("./crop_val.json",'w',encoding='utf-8') as json_file:
        json.dump(res_json,json_file,ensure_ascii=False)