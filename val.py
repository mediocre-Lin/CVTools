"""
@File: val.py
@Author: Keith 
@Time: 2022/03/23 13:55:03
@Contact: 956744413@qq.com
"""
import torch, torchvision, warnings, random, argparse, os
import numpy as np
import torch.nn as nn
from common import writeData
from classification.dataSet import camera_Dataset, data_generate
from classification.tools import validate
from classification.model import classifier_model

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
SEED = 2021
torch.manual_seed(SEED)
random.seed(SEED)
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)
num_workers = 0 
opt = None


CLASSES = ['ANNO_1','ANNO_2']

def validater():
    _, val_imgs, _, val_label = data_generate(opt.data)

    val_data = camera_Dataset(val_imgs, val_label,mode = 'val')
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    print("-" * 40)

    print("Counts of val_data :", len(val_data))
    print("distribution of label:", np.bincount(val_label))
    net = opt.model
    num_classes = opt.num_classes
    log_path = opt.log_path
    weight_path = os.path.join(log_path,'Best.pt')

    model = classifier_model(model_cnn=net,num_class = num_classes)
    model = model.to(device)
    chkpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    analyze = {
        'path':log_path,
        'classes':CLASSES
    }

    val_loss, val_acc, val_multi_acc = validate(
        val_dataloader, model, criterion,analyze=analyze)

    writeData(None, val_loss, val_acc, val_multi_acc, 0, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=64)
    parser.add_argument('--num_classes', type=int,
                        default=2)
    parser.add_argument('--data', type=str,
                        default='data/')
    parser.add_argument('--model', type=str,
                        default='resnet18')
    parser.add_argument('--log_path', type=str,
                        default='/content/drive/MyDrive/Keith/Pixdot/CVTools/common/log/classfication/resnet18_03_24ver_9')
    opt = parser.parse_args()

    print(opt)
    validater()
