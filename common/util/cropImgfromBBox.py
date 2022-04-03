'''
@File: cropImgfromBBox
@Author: Keith 
@Time: 2022/03/29 10:58:44
@Contact: 956744413@qq.com
'''
import json, os, glob
import matplotlib.pyplot as plt
from PIL import Image

def get_camera_bbox(json_file):
    shape_infos = json_file['shapes']
    for shape_info in shape_infos:
        if shape_info['label'] == 'camera':
            return shape_info['points'][0],shape_info['points'][1]
def reset_bbox(json_file,bbox):
    shape_infos = json_file['shapes']
    for shape_info in shape_infos:
        if shape_info['label'] == 'defect':
            point_info = shape_info['points']
            for i in range(len(point_info)):
                point_info[i][0] -= bbox[0]
                point_info[i][1] -= bbox[1]
    shape_info['points'] = point_info
    json_file['shapes'] = shape_infos
    return json_file


CLASSES = ['class1','class2','class3','class4','class5','class6','class7','class8','class9']
for i in range(len(CLASSES)):
    json_path = 'D:\pixdot\project_camera\glass_defect_data\\'+f"{CLASSES[i]}"+'\\*.json'
    jsons = glob.glob(json_path)
    for idx in range(len(jsons)):
        with open(jsons[idx]) as f: 
            tar_json = json.load(f)

        bbox_info,bbox_info2 = get_camera_bbox(tar_json)
        res_js = reset_bbox(tar_json,bbox_info)
        img_path = jsons[idx].replace('.json','.bmp')
        img = Image.open(img_path)
        camera_bbox = [int(bbox_info[0]),int(bbox_info[1]),int(bbox_info2[0]),int(bbox_info2[1])]
        region = img.crop(camera_bbox)
        tar_img = img_path.replace('.bmp','.jpg')
        region.save(tar_img,quality=95)
        with open(jsons[idx],'w',encoding='utf-8') as json_file:
                json.dump(res_js,json_file,ensure_ascii=False)
# camera_train_json_path = 'D:\pixdot\project_camera\\regular_data_two_classes\\train.json'
# camera_val_json_path = 'D:\pixdot\project_camera\\regular_data_two_classes\\val.json'

# with open(camera_train_json_path) as f: 
#     train_camera = json.load(f)
# with open(camera_val_json_path) as cf: 
#     val_camera = json.load(cf)

# train_camera_info = train_camera['annotations']
# train_img_info = train_camera['images']

# val_camera_info = val_camera['annotations']
# val_img_info = val_camera['images']

# dir_path = 'D:\pixdot\project_camera\\regular_data_two_classes\coco_Dataset\\train2017'
# out_dir_path = 'D:\pixdot\project_camera\\regular_data_two_classes\crop_Camera_train'


# # for camera_info in train_camera_info:
# #     img_id = camera_info['image_id']
# #     for img_info in train_img_info:
# #         if img_info['id'] == img_id:
# #             img_path = os.path.join(dir_path,img_info['file_name'].replace('.jpg','.bmp'))
# #             img = Image.open(img_path)
# #             bbox = camera_info['bbox']
# #             x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[0])+ int(bbox[2]),int(bbox[1]+bbox[3])
# #             region = img.crop([x1,y1,x2,y2])
# #             res_path = os.path.join(out_dir_path,img_info['file_name'])
# #             region.save(res_path,quality=95)

# dir_path = 'D:\pixdot\project_camera\\regular_data_two_classes\class_1'
# out_dir_path = 'D:\pixdot\project_camera\\regular_data_two_classes\crop_Camera_val'

# for camera_info in val_camera_info:
#     img_id = camera_info['image_id']
#     for img_info in val_img_info:
#         if img_info['id'] == img_id:
#             img_path = os.path.join(dir_path,img_info['file_name'].replace('.jpg','.bmp'))
#             img = Image.open(img_path)
#             bbox = camera_info['bbox']
#             x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[0])+ int(bbox[2]),int(bbox[1]+bbox[3])
#             region = img.crop([x1,y1,x2,y2])
#             res_path = os.path.join(out_dir_path,img_info['file_name'])
#             region.save(res_path,quality=95)

# if __name__ == '__main__':
#     path = 'D:\pixdot\project_camera\\regular_data_two_classes\crop_camera_dataset\\annotations\crop_val.json'
#     with open(path) as f: 
#         data_js = json.load(f)
#     annotations = data_js['annotations']
#     count = 0
#     for anno in annotations:
#         anno['id'] = count
#         count += 1
#     data_js['annotations']  = annotations
#     with open("./val.json",'w',encoding='utf-8') as json_file:
#         json.dump(data_js,json_file,ensure_ascii=False)