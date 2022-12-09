import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import models
class UEMGNET(nn.Module):
    def __init__(self, classes):
        super(UEMGNET,self).__init__()
        self.classes = classes
                # 第一层卷积网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=(21,3),stride=(1,1),padding=(10,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(10,1),stride=(10,1))
        )

        # 第二层卷积网络
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,2),stride=(3,2))
        )

        # 第三层卷积网络
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )

        # 降维函数
        self.flatten1 = nn.Flatten()
        self.dropout1 = nn.Dropout1d(p=0.5)
        self.fc1 = nn.Linear(1152,128)
        self.dropout2 = nn.Dropout1d(p=0.5)
        self.fc2 = nn.Linear(128,classes)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten1(out)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = F.softmax(self.fc2(out),dim=1)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(12288,1024)#两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,40)
#         self.dp = nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
#         x = self.fc3(x)
#         self.dp(x)
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)  
#         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(24576,100)
        self.mlp2 = torch.nn.Linear(100,40)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x



class NinaProNet(nn.Module):
    def __init__(self, class_num=None, base_features=16, window_length=256, input_channels=10):
        super(NinaProNet, self).__init__()
        self.class_num = class_num
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=base_features * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=base_features * 2,
                      out_channels=base_features * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=base_features * 4,
                      out_channels=base_features * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=base_features * 4,
                      out_channels=base_features * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(base_features * 4 * int(window_length / 8), 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.mlp2 = nn.Linear(100, self.class_num)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = F.softmax(x, dim=1)
        return x

class GengNet(nn.Module):
    def __init__(self, class_num=None, base_features=64, window_length=256, input_channels=6):
        super(GengNet, self).__init__()
        self.class_num = class_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # for EMG images, the channels is 1. not the signal channels: input_channels
                      out_channels=base_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base_features,
                      out_channels=base_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base_features,
                      out_channels=base_features,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=base_features,
                      out_channels=base_features,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.fcn1 = nn.Sequential(
            nn.Linear(base_features * window_length * input_channels, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fcn2 = nn.Linear(128, self.class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.fcn1(x.view(x.size(0), -1))
        x = self.fcn2(x)
        x = F.softmax(x, dim=1)
        return x

class FCN(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()
        self.class_num = class_num
        self.input_size = input_size
        self.fcn1 = nn.Sequential(
            nn.Linear(in_features=input_size[0] * input_size[1], out_features=10000),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(10000),
            nn.ReLU())
        self.fcn2 = nn.Sequential(
            nn.Linear(in_features=10000, out_features=1000),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(1000),
            nn.ReLU())
        self.fcn3 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(100),
            nn.ReLU())
        self.fcn4 = nn.Linear(in_features=100, out_features=self.class_num)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = x.view(x.size(0), -1)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        x = self.fcn4(x)
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model = models.resnet18(num_classes=40)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)
    summary(model,(1,12,10))
