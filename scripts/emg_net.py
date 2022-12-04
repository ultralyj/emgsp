import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = UEMGNET(40).to(device)
# summary(model,(1,200,12))