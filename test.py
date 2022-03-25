"""
@File: test.py
@Author: Keith 
@Time: 2022/03/25 14:38:38
@Contact: 956744413@qq.com
"""
import glob
import shutil
import os 
if __name__ == '__main__':
    data_path = 'D:\pixdot\project_camera\data_two_classes'
    img_files = glob.glob(data_path+'\\*\\*.bmp')
    json_files = glob.glob(data_path+'\\*\\*.json')
    print(len(img_files))
    for i in range(len(json_files)):
        json_files[i] = json_files[i].replace('.json','.bmp')
    undefect_img = list(set(img_files) - set(json_files))
    for img in undefect_img:
        img_name = img.split('\\')[-1]
        shutil.copy(img,os.path.join('D:\pixdot\project_camera\\regular_data_two_classes\class_2',img_name))
    # print(undefect_img)
    # target_path = 'D:\pixdot\project_camera\\regular_data_two_classes'
    # img_file = glob.glob(data_path+'\\*\\*.bmp')
    # json_file = glob.glob(data_path+'\\*\\*.json')
    # un_defect_files = 
    # for js_file in json_file:
    #     img_file = js_file.replace('.json','.bmp')
    #     js = js_file.split('\\')[-1]
    #     img = img_file.split('\\')[-1]
    #     shutil.copy(js_file, os.path.join('D:\pixdot\project_camera\\regular_data_two_classes\class_1_annotation',js))
    #     shutil.copy(img_file, os.path.join('D:\pixdot\project_camera\\regular_data_two_classes\class_1',img))
