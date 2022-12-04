# -*- coding: utf-8 -*-
# file name: ninapro_utils
# author: ultralyj
# e-mail: 1951578@tongji.edu.cn
# version: v1.0
# date: 2022-12-3
# brief: only for test

import h5py
import numpy as np
import torch
from emg_net import UEMGNET
from ninapro_utils import *
from torch.autograd import Variable
from torchsummary import summary
import torchvision.models as models

import time
# 数据集的地址
db7_path = 'E:/db7/'
# 测试者人数
num_subject = 22
BATCH_SIZE = 128

def read_data(filename):
    print('open h5 file:%s...'%(filename),end='',flush=True)
    with h5py.File(filename,'r') as f:
        imageData   = f['featureData'][:]
        imageLabel  = f['featureLabel'][:] 
        print('[ok]',end='')
        print('data:', imageData.shape,', label:',end='')
        print(imageLabel.shape, flush=True)
        return imageData,imageLabel

def merge_emg_data(b,e):
    for i in range(b, e + 1):
        d0,l0 = read_data('data/s{0}.h5'.format(i))
        if (i==b):
            d = d0
            l = l0
        else:
            d = np.concatenate((d,d0),axis=0)
            l = np.concatenate((l,l0),axis=0)
    return d,l

def get_dataLoader(data,label):
    data_t = torch.tensor(data)
    label_t = torch.tensor(label)
    torch_dataset = torch.utils.data.TensorDataset(data_t, label_t)
    torch_loader = torch.utils.data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=64,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        drop_last = True
    )
    return torch_loader

def save_model(model,filename):
    torch.save(model.state_dict(), filename)

def testAccuracy(device, model, test_loader):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            emg_data, labels = data
            emg_data = Variable(emg_data.to(device))    # torch.Size([64, 1, 200, 12])
            labels = Variable(labels.to(device))        # torch.Size([64])
            # run the model on the test set to predict labels
            outputs = model(emg_data)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train(device, num_epochs, train_loader, test_loader):
    best_accuracy = 0
    model = UEMGNET(40).to(device)
    summary(model,(1,200,12))
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (emg_data, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            emg_data = Variable(emg_data.to(device))    # torch.Size([64, 1, 200, 12])
            labels = Variable(labels.to(device))        # torch.Size([64])

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(emg_data)
            # compute the loss based on model output and real labels
            loss = loss_func(outputs, labels.long())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy(device, model, test_loader)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        save_model(model, "model/checkpoint_epoch{1}_{0}.pth".format(epoch, (int(time.time())%1000000)))
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, "model/best_acc.pth")
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # pretreatment_ninapro(db7_path,num_subject)
    # 选前16个受试者作为训练集，17-19作为测试集,20作为验证集
    data_train,label_train = merge_emg_data(1,16)
    data_test,label_test = merge_emg_data(17,19)
    print('[train]:%d, [test]:%d'%(len(label_train),len(label_test)))
    
    # 数据升维
    data_train = np.expand_dims(data_train,axis=1)
    data_test = np.expand_dims(data_test,axis=1)

    # 生成data_loader
    train_loader = get_dataLoader(data_train, label_train)
    test_loader = get_dataLoader(data_test, label_test)
    # 模型训练
    train(device, 20, train_loader, test_loader)


    print('---finish---') 
