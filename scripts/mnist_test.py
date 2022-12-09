import torch
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
import numpy as np
import torch
from emg_net import GengNet,NinaProNet, UEMGNET, CNN
from ninapro_utils import *
from torch.autograd import Variable
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as tt
import time



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
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_accs = []
    train_loss = []
    test_accs = []

    for epoch in range(3):
        running_loss = 0.0
        for i,data in enumerate(train_loader,0):#0是下标起始位置默认为0
            # data 的格式[[inputs, labels]]       
    #         inputs,labels = data
            inputs,labels = data[0].to(device), data[1].to(device)
            #初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()         

            #前向+后向+优化     
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            # loss 的输出，每个一百个batch输出，平均的loss
            running_loss += loss.item()
            if i%100 == 99:
                print('[%d,%5d] loss :%.3f' %
                    (epoch+1,i+1,running_loss/100),flush=True)
                running_loss = 0.0
            train_loss.append(loss.item())

            # 训练曲线的绘制 一个batch中的准确率
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)# labels 的长度
            correct = (predicted == labels).sum().item() # 预测正确的数目
            train_accs.append(100*correct/total)
            print(' acc=%d'%(100*correct/total))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    data_tf = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])]
    )
    
    train_dataset = datasets.MNIST(root='./MNIST',train=True,transform=data_tf,download=False)
    testData = datasets.MNIST(root='./MNIST',train = False,transform = data_tf)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=64,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testData,
                                            batch_size=1000,
                                            shuffle=True)
    print("begin")
    train(device, 20, train_loader=train_loader, test_loader= test_loader)
    print('finish')
