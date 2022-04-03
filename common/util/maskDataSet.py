"""
@File: maskDataSet
@Author: Keith 
@Time: 2022/03/25 15:19:57
@Contact: 956744413@qq.com
"""
import os

path = 'D:\pixdot\project_camera\glass_defect_data\coco1_camera_defect\\train2017' 

json_file = os.listdir(path)

os.system("activate base")

for file in json_file:

    os.system("labelme_json_to_dataset.exe %s"%(path + '/' + file))
