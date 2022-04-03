"""
@File: extractClass.py
@Author: Keith 
@Time: 2022/03/28 20:30:20
@Contact: 956744413@qq.com
"""

import json
res_json_path = 'D:\pixdot\project_camera\\regular_data_two_classes\coco_Dataset\\annotations\instances_train2017.json'
with open(res_json_path) as f: 
    res_json = json.load(f)
print(res_json.keys())
new_json = {}
new_json['info'] = res_json['info']
new_json['licenses'] = res_json['licenses']
new_json['type'] = res_json['type']
new_json['images'] = res_json['images']
new_json['categories'] = [
        {
            "supercategory": None,
            "id": 0,
            "name": "_background_"
        },
        {
            "supercategory": None,
            "id": 1,
            "name": "camera"
        }]
annotations = res_json['annotations']
annota_new = []
count = 1
for anno in annotations:
    if anno['category_id'] == 2:
        anno['category_id'] = 1
        anno['id'] = count
        count += 1
        annota_new.append(anno)
new_json['annotations'] = annota_new
with open("./train.json",'w',encoding='utf-8') as json_file:
        json.dump(new_json,json_file,ensure_ascii=False)