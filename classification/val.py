"""
@File: val.py
@Author: Keith 
@Time: 2022/03/23 13:55:03
@Contact: 956744413@qq.com
"""
import torch, torchvision, warnings, random, argparse
import torch.nn as nn
from classification.dataSet import camera_Dataset, data_generate
from classification.tools import validate
from common import predict_analyze
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


def validater():
    _, val_imgs, _, val_label = data_generate(opt.data)
    val_data = camera_Dataset(val_imgs, val_label)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    print("-" * 40)
    print("Counts of val_data :", len(val_data))

    model = torchvision.models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    fc_features = model.fc.in_features
    # 修改类别为10，重定义最后一层
    model.fc = nn.Linear(fc_features, 4)
    model = model.to(device)
    chkpt = torch.load(opt.log_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    criterion = nn.CrossEntropyLoss().to(device)

    val_loss, val_acc, val_multi_acc = validate(
        val_dataloader, model, criterion)

    print('val_loss: %5.5s , acc: %5.5s' % (val_loss, val_acc))
    print('class_1_acc:%5.5s\nclass_2_acc:%5.5s\nclass_3_acc:%5.5s\nclass_4_acc:%5.5s\n' % (
        val_multi_acc[0], val_multi_acc[1], val_multi_acc[2], val_multi_acc[3]))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=2)
    parser.add_argument('--data', type=str,
                        default='data/')
    parser.add_argument('--log_path', type=str,
                        default='common/log/classfication/')
    opt = parser.parse_args()

    print(opt)
    validater()
