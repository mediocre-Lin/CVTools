"""
@File: tools.py
@Author: Keith 
@Time: 2022/03/25 16:16:52
@Contact: 956744413@qq.com
"""
import torch, warnings, random
import numpy as np
from common import multi_class_acc
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast
from common import predict_analyze
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
SEED = 2021
torch.manual_seed(SEED)
random.seed(SEED)
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

def train(train_loader, model, criterion, optimizer,amp_mode = True):
    model.train()
    train_loss = []
    count = 0
    pre_record = None
    label_record = None
    for image, label in tqdm(train_loader):
        count += 1
        image = image.to(device)
        label = label.to(device)
        if not amp_mode:
            out = model(image).to(device)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            with autocast():
                out = model(image)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss.append(loss.item() / label.shape[0])

        _, predicted = torch.max(out.data,
                                    dim = 1)
        if pre_record is None:
            pre_record = predicted.cpu().numpy()
            label_record = label.cpu().numpy()
        else:
            pre_record = np.concatenate((pre_record,predicted.cpu().numpy()))
            label_record = np.concatenate((label_record,label.cpu().numpy()))

    acc = (pre_record == label_record).sum() / len(label_record)
    return np.mean(train_loss), acc, multi_class_acc(pre_record,label_record)


def validate(val_loader, model, criterion, analyze = None):
    model.eval()
    val_loss = []
    count = 0 
    pre_record = None
    label_record = None
    with torch.no_grad():
        for image, label in tqdm(val_loader):
            count += 1
            image = image.to(device)
            label = label.to(device)
            out = model(image).to(device)
            loss = criterion(out, label)
            val_loss.append(loss.item() / label.shape[0])
            _, predicted = torch.max(out.data,
                                     dim = 1)
            if pre_record is None and label_record is None:
                pre_record = predicted.cpu().numpy()
                label_record = label.cpu().numpy()
            else:
                pre_record = np.concatenate((pre_record,predicted.cpu().numpy()))
                label_record = np.concatenate((label_record,label.cpu().numpy()))
    acc = (pre_record == label_record).sum() / len(label_record)
    if not analyze is None:
        res_path = analyze['path']
        class_name = analyze['classes']
        predict_analyze(pre_record,label_record,res_path,class_name = class_name)

    return np.mean(val_loss), acc, multi_class_acc(pre_record,label_record)