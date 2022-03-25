"""
@File: maskDataSet
@Author: Keith 
@Time: 2022/03/25 15:19:57
@Contact: 956744413@qq.com
"""
import os

path = 'D:\pixdot\project_camera\\regular_data_two_classes\class_1_annotation' 

json_file = os.listdir(path)

os.system("activate base")

for file in json_file:

    os.system("labelme_json_to_dataset.exe %s"%(path + '/' + file))
