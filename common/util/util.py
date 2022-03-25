"""
@File: util.py
@Author: Keith 
@Time: 2022/03/23 13:04:44
@Contact: 956744413@qq.com
"""
import os
import time
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def get_time():
    return time.strftime('%m_%d', time.localtime(time.time()))


def name_gernerate(path, name):
    file_name = name + '_'+get_time() + 'ver_'
    count = 1
    while os.path.exists(os.path.join(path, file_name+str(count))):
        count += 1
    return file_name+str(count)

def multi_class_acc(pre, label, class_num=None):
    
    class_num = len(np.unique(label)) if class_num is None else class_num
    acc = np.zeros(class_num)
    for _class in range(class_num):
        class_idx = label == _class
        acc[_class] =  (pre[class_idx] == label[class_idx]).sum().item() / ((class_idx).sum().item() + 1e-5)
    return acc


def plot_confusion_matrix(cm, path, class_name, title='Confusion Matrix'):

    classes = class_name
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='red',
                     fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(name_gernerate(path, title)+'.png', format='png')
    plt.show()


def predict_analyze(predict, label, res_path, class_num=4, class_name=None):
    if not class_name is None:
        classes = class_name
    else:
        classes = ['class_'+str(i) for i in range(1, class_num+1)]
    pre = np.array(predict)
    label = np.array(label)
    correct_idx = (pre == label)
    wrong_idx = ~correct_idx
    print('correct num:', sum(correct_idx).item())
    print('wrong num:', sum(wrong_idx).item())
    confus_mat = confusion_matrix(pre, label)
    print(confus_mat)
    plot_confusion_matrix(confus_mat, res_path, classes)

def writeData(writer = None,loss = 0 ,acc= 0,multi_acc = 0,epoch = 0 ,mode='train'):
    if mode == 'test':
      print('%s_loss: %5.5s , acc: %5.5s'%(mode, loss, acc))
      for i, c_acc in enumerate(multi_acc):
        print('calss_%s : %5.5s'%(i+1,c_acc))
    else:
      # loss
      writer.add_scalar('Loss/'+mode, loss, epoch)
      # acc
      writer.add_scalar('acc/'+mode, acc, epoch)
      # multi_acc
      for i in range(len(multi_acc)):
        writer.add_scalar('acc/'+mode+'/'+'class:'+str(i), multi_acc[i], epoch)
      
      print('%s_loss: %5.5s , acc: %5.5s'%(mode, loss, acc))
      for i, c_acc in enumerate(multi_acc):
        print('calss_%s : %5.5s'%(i+1,c_acc))
      writer.flush()

def load_weight(model,weight_path,device = 'cpu'):
    model = model.to(device)
    chkpt = torch.load(weight_path,map_location = device)
    model.load_state_dict(chkpt['model'])
    return model

if __name__ == '__main__':
    pre = [0, 1, 2, 3, 3, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0]
    label = [1, 2, 3, 0, 3, 1, 2, 3, 3, 3, 1, 1, 0, 0, 1]
    predict_analyze(pre, label, './',
                    class_name=['anno_1', 'anno_2', 'anno_3', 'anno_4'])
