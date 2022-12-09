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
from emg_net import NinaProNet
from ninapro_utils import *
from torch.autograd import Variable
from torchsummary import summary
import torchvision.models as models

import time
# 数据集的地址
db7_path = 'E:/db7/'
# 测试者人数
num_subject = 22


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("The model will be running on", device, "device")
    pretreatment_ninapro(db7_path, num_subject)
    for i in range(1,num_subject+1):
        emg_dataset_split('data/s{0}_f.h5'.format(i),'data/ss{0}_f.h5'.format(i))        
    print('---finish---') 
    