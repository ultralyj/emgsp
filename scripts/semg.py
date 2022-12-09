import scipy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as F
from emg_net import *
import time
def get_feature_dict(filename):
    """将ninapro_feature的MAT文件加载为字典

    Args:
        path: mat文件路径
        filename: mat文件名

    Returns:
        数据集字典
        [feat_set, featStim, featRep]
    """
    # 读取MAT文件
    print('load file: ' + filename + '...', end= '', flush=True)
    dict_feature=scipy.io.loadmat(filename)
    if (dict_feature != ()):
        #print(ninapro_data.keys())
        print('[ok]:%d'%(len(dict_feature['featStim'])), flush=True)
    # 返回字典
    return dict_feature

def get_semg(filename = "../feature/feature_S1s20.mat"):
    feature_dict = get_feature_dict(filename)
    index = []
    for i in range(len(feature_dict['featStim'])):
        if feature_dict['featStim'][i]!=0:
            index.append(i)
    emg_feature = feature_dict['feat_set'][index,:,:]
    labels = feature_dict['featStim'][index,:]
    return emg_feature,labels

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
    model = FCN(input_size=(12,10), class_num=40).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    
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
            torch.save(model.state_dict(), "../model/best_epoch{1}_{0}.pth".format(epoch, (int(time.time())%1000000)))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    emg_feature,labels = get_semg()

    # 矩阵数据预处理
    emg_feature = np.expand_dims(emg_feature,axis=1)
    emg_feature = emg_feature.astype(np.float32)
    labels = labels.astype(np.int8)
    labels = labels.flatten() -1

    print('reshaped data:',emg_feature.shape,emg_feature.dtype)
    print('reshaped label:',labels.shape,labels.dtype)

    print('get dataloader...', end='',flush=True)
    emg_feature_t = torch.tensor(emg_feature)
    labels_t = torch.tensor(labels)
    torch_dataset = torch.utils.data.TensorDataset(emg_feature_t, labels_t)

    # 划分数据集与训练集
    test_size = int(len(labels)*0.2)
    train_dataset,test_dataset = torch.utils.data.random_split(dataset=torch_dataset,lengths = [len(labels)-test_size,test_size])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=64,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        drop_last = True,
    
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=32,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        drop_last = True,
    )
    print('[ok]')
    print('begin to train.....')
    # 模型训练
    train(device, 50, train_loader, test_loader)

if __name__ == "__main__":
    main()
    print("finish")