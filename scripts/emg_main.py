# -*- coding: utf-8 -*-
# file name: emg_main
# author: ultralyj
# e-mail: 1951578@tongji.edu.cn
# version: v1.0
# date: 2022-12-5
# brief: main body

import h5py
import numpy as np
import torch
from emg_net import *
from ninapro_utils import *
from torch.autograd import Variable
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as tt

import time
# 数据集的地址
db7_path = 'E:/db7/'
# 测试者人数
num_subject = 22
BATCH_SIZE = 32



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
        batch_size=128,      # mini batch size
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

class FeatureExtractor(object):
    def __init__(self):
        # 获取类内的所有方法名字
        self.method_list = dir(self)

    def __call__(self, sample):
        feature_list = []
        # 由于继承了nn.module, 里面有一些方法不是计算特征的，要忽略掉
        for method in self.method_list:
            if method[0:2] == 'f_':
                func = getattr(self, method)
                feature_list.append(func(sample[0,:,:]))
        feature = np.array(feature_list)
        return feature


    @staticmethod
    def f_RMS(d):
        return np.sqrt(np.mean(np.square(d), axis=0))

    @staticmethod
    def f_MAV(d):
        return np.mean(np.abs(d), axis=0)

    @staticmethod  # 过零点次数
    def f_ZC(d):
        nZC = np.zeros(d.shape[1])
        th = np.mean(d, axis=0)
        th = np.abs(th)
        for i in range(1, d.shape[0]):
            for j in range(d.shape[1]):
                if d[i - 1, j] < th[j] < d[i, j]:
                    nZC[j] += 1
                elif d[i - 1, j] > th[j] > d[i, j]:
                    nZC[j] += 1
        return nZC / d.shape[0]

    @staticmethod  # slope sign change
    def f_SSC(d):
        nSSC = np.zeros(d.shape[1])
        th = np.mean(d, axis=0)
        th = np.abs(th)
        for i in range(2, d.shape[0]):
            diff1 = d[i] - d[i - 1]
            diff2 = d[i - 1] - d[i - 2]
            for j in range(d.shape[1]):
                if np.abs(diff1[j]) > th[j] and np.abs(diff2[j]) > th[j] and (diff1[j] * diff2[j]) < 0:
                    nSSC[j] += 1
        return nSSC / d.shape[0]

    @staticmethod
    def f_VAR(d):
        feature = np.var(d, axis=0)
        return feature

def train(device, num_epochs, train_loader, test_loader):
    best_accuracy = 0
    model = FCN(input_size=(5,12), class_num=40).to(device)
    summary(model,(1,5,12))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (emg_data, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            emg_data = Variable(emg_data.to(device))    # torch.Size([64, 1, 200, 12])
            labels = Variable(labels.to(device))        # torch.Size([64])

            emg_data = emg_data.to(torch.float32)
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
            if i % 100 == 99:    
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
def train3(device, num_epochs, train_loader, test_loader):
    best_accuracy = 0
    model = FCN(input_size=(5,12), class_num=40).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_accs = []
    train_loss = []
    test_accs = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,(inputs, labels) in enumerate(train_loader,0):#0是下标起始位置默认为0
            # data 的格式[[inputs, labels]]       
    #         inputs,labels = data
            inputs = Variable(inputs.to(device))    # torch.Size([64, 1, 200, 12])
            labels = Variable(labels.to(device))        # torch.Size([64]) 
            inputs = inputs.to(torch.float32)
            #初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()         

            #前向+后向+优化     
            outputs = model(inputs)
            loss = loss_func(outputs,labels.long())
            loss.backward()
            optimizer.step()

            # loss 的输出，每个一百个batch输出，平均的loss
            running_loss += loss.item()
            if i%100 == 99:
                print('[%d,%5d] loss :%.3f' %
                    (epoch+1,i+1,running_loss/100),end='',flush=True)
                running_loss = 0.0
            train_loss.append(loss.item())

            # 训练曲线的绘制 一个batch中的准确率
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)# labels 的长度
            correct = (predicted == labels).sum().item() # 预测正确的数目
            train_accs.append(100*correct/total)
            if i%100 == 99:
                print(' acc=%d'%(100*correct/total))
        accuracy = testAccuracy(device, model, test_loader)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "model/best_epoch{1}_{0}.pth".format(epoch, (int(time.time())%1000000)))


class EMG_DataSet(torch.utils.data.Dataset):   # 创建一个叫做DogVsCatDataset的Dataset，继承自父类torch.utils.data.Dataset
    def __init__(self, emg, labels, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels = labels
        self.emgs = emg
        self.transform = transform
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        if self.transform:
            emg_f = self.transform(self.emgs[idx])
        emg_f = torch.tensor(emg_f)
        label = torch.tensor(self.labels[idx])
        return emg_f, label

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # # pretreatment_ninapro(db7_path,num_subject)
    # # 选前16个受试者作为训练集，17-19作为测试集,20作为验证集
    # data,label = read_data('data/raw_split.h5')
    # #data_test,label_test = merge_emg_data(17,19)
    # #print('[train]:%d, [test]:%d'%(len(data),len(label_test)))
    # # 数据升维
    # label = label.flatten()
    # #data_test = np.expand_dims(data_test,axis=1)
    
    # gain = 1e6
    # data*=gain
    # data = np.expand_dims(data,axis=1)

    # print('label shape:',label.shape,' ,data shape: ', data.shape)
    # data_t = torch.tensor(data)
    # label_t = torch.tensor(label)
    # torch_dataset = torch.utils.data.TensorDataset(data_t, label_t)
    
    # n_test = int(len(label)*0.2)
    # train_dataset = torch.utils.data.TensorDataset(data_t, label_t)
    # test_dataset = torch.utils.data.TensorDataset(data_t[n_test:,:,:], label_t[n_test:])

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,      # torch TensorDataset format
    #     batch_size=BATCH_SIZE,      # mini batch size
    #     shuffle=True,               # 要不要打乱数据 (打乱比较好)
    #     num_workers=2,              # 多线程来读数据
    #     pin_memory=True,
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,      # torch TensorDataset format
    #     batch_size=BATCH_SIZE,      # mini batch size
    #     shuffle=True,               
    #     num_workers=2,              # 多线程来读数据
    #     pin_memory=True,
    # )

    d0,l0 = read_data('../data/ss1_f.h5')
    print('[data]:%d, [labe]:%d'%(len(d0),len(l0)),d0.shape)
    
    # 矩阵数据预处理
    d0 = np.expand_dims(d0,axis=1)
    d0 = d0.astype(np.float32)
    l0 = l0.flatten()
    np.swapaxes(d0,1,2)
    print('reshaped data:',d0.shape,d0.dtype)
    print('reshaped label:',l0.shape,l0.dtype)

    ttemg = tt.Compose([FeatureExtractor()])


    test_size = int(len(l0)*0.2)
    torch_dataset = EMG_DataSet(d0, l0, transform=ttemg)
    torch_dataset.__getitem__(2)
    train_dataset,test_dataset = torch.utils.data.random_split(dataset=torch_dataset,lengths = [len(l0)-test_size,test_size])
    print('get dataset...[ok]')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=32,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        pin_memory=True,
        drop_last = True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=128,      # mini batch size
        shuffle=True,               
        num_workers=2,              # 多线程来读数据
        pin_memory=True,
        drop_last = True
    )
    
    train3(device, 50, train_loader, test_loader)


    print('---finish---') 
