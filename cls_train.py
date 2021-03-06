"""
@File: train.py
@Author: Keith 
@Time: 2022/03/23 13:54:45
@Contact: 956744413@qq.com
"""
import numpy as np
import torch, torchvision, warnings, random, argparse,os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from classification.dataSet import camera_Dataset, data_generate
from common import writeData,name_gernerate
from classification.tools import train, validate
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


def trainer():
    train_imgs,val_imgs,train_label,val_label = data_generate(opt.data)
    train_data = camera_Dataset(train_imgs, train_label, mode = 'train')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,num_workers=0)
    val_data = camera_Dataset(val_imgs, val_label, mode = 'val')
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size,  shuffle=False,num_workers=0)
    print("-" * 40)
    print("Counts of train_data :", len(train_data))
    print("Counts of val_data :", len(val_data))

    
    epochs = opt.epochs
    lr = opt.lr
    log_path =opt.log_path
    net = opt.model
    num_classes = len(np.unique(val_label))

    model = classifier_model(model_cnn=net,num_class = num_classes)
    model = model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    best_acc = 0.
    for epoch in range(epochs):

        print('\nEpoch:{}/{}:'.format(epoch, epochs))

        train_loss, train_acc, train_multi_acc = train(train_dataloader, model, criterion, optimizer)
        
        writeData(writer, train_loss, train_acc, train_multi_acc, epoch, 'train')
        val_loss, val_acc, val_multi_acc = validate(val_dataloader, model, criterion)
        writeData(writer, val_loss, val_acc, val_multi_acc, epoch, 'val')

        if val_acc > best_acc:
            print("--------save best ........")
            best_acc = val_acc
            chkpt = {
                    'model': model.state_dict(),
                    'best_acc':best_acc
                }
            torch.save(chkpt, log_path + '/Best.pt')
        chkpt = {
                'model': model.state_dict(),
                'last_acc':val_acc
            }
        torch.save(chkpt, log_path + '/last.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,
                        default=30)
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--lr', type=float,
                        default=0.005)
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5)
    parser.add_argument('--model', type=str,
                        default='resnet18')
    parser.add_argument('--data', type=str,
                        default='data/')
    parser.add_argument('--log_path', type=str,
                        default='/content/drive/MyDrive/Keith/Pixdot/CVTools/common/log/classfication/')
    parser.add_argument('--amp_train', action='store_true', help='training with amp, faster trainnig')

    opt = parser.parse_args()
    log_path = name_gernerate(opt.log_path,opt.model)
    log_path = os.path.join(opt.log_path,log_path)
    os.makedirs(log_path)
    print('log and weight will be saved in ',log_path)
    opt.log_path = log_path
    print(opt)
    writer = SummaryWriter(log_path) 
    trainer()
    writer.close()

