# -*- coding: utf-8 -*-
# file name: ninapro_utils
# author: ultralyj
# e-mail: 1951578@tongji.edu.cn
# version: v1.0
# date: 2022-12-3
# brief: process ninapro dataset from MATLAB format 

'''
数据定义
- subject: subject number
- exercise: exercise number
- acc (36 columns): three-axes accelerometers of the 12 sensors
- gyro (36 columns): three-axes gyroscopes of the 12 sensors
- mag (36 columns): three-axes magnetometers of the 12 sensors
- emg (12 columns): sEMG signal of the 12 sensors
- glove (22 columns): uncalibrated signal from the 22 sensors of the cyberglove
- stimulus (1 column): the movement repeated by the subject.
- restimulus (1 column): again the movement repeated by the subject. In this case the duration of the movement label is refined a-posteriori in order to correspond to the real movement
- repetition (1 column): repetition of the stimulus
- rerepetition (1 column): repetition of restimulus
'''

import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py

def get_ninapro(path, filename):
    """将ninapro的MAT文件加载为字典

    Args:
        path: mat文件路径
        filename: mat文件名

    Returns:
        数据集字典
        # dict_keys(['__header__', '__version__', '__globals__', 
        # 'subject', 'exercise', 'emg', 'acc', 'gyro', 'mag', 
        # 'stimulus', 'glove', 'repetition', 'restimulus', 'rerepetition'])
    """
    # 读取MAT文件
    print('load file: ' + filename + '...', end= '', flush=True)
    ninapro_data=scipy.io.loadmat(path+filename)
    if (ninapro_data != ()):
        #print(ninapro_data.keys())
        print('[ok]:%d'%(len(ninapro_data['restimulus'])), flush=True)
    # 返回字典
    return ninapro_data

def get_stimulus_range(dataset, stimulus):
    """获取某一动作的数据范围

    Args:
        dataset: 数据集字典
        stimulus: 刺激的序号

    Returns:
        [r_begin, r_end]: 数据范围的上下标
    """
    if(stimulus == 0):
        print('[WARNING] the stimulus number maybe error')
        return [-1, -1]
    i = 0
    length = len(dataset['restimulus'])
    while (i<length):
        if(dataset['restimulus'][i] == stimulus):
            break
        i+= 1
    r_begin = i
    while (i<length):
        if(dataset['restimulus'][i] == stimulus + 1):
            break
        i+= 1
    r_end = i - 1
    return [r_begin, r_end]
    
def get_repetition_range(dataset, stimulus, repetition):
    """获取某一次动作的数据范围

    Args:
        dataset: 数据集字典
        stimulus: 刺激的序号
        repetition: 次数

    Returns:
        [r_begin, r_end]: 某一次刺激的数据范围的上下标
    """
    [b, e] = get_stimulus_range(dataset, stimulus)
    i = b
    while (i <= e):
        if (dataset['rerepetition'][i] == repetition):
            break
        i+= 1
    r_begin = i
    while (i <= e):
        if (dataset['rerepetition'][i] != repetition):
            break
        i+= 1
    r_end = i - 1
    return [r_begin, r_end]

def merge_exercise(E1, E2):
    """合并两个exercises，返回值只包含刺激的序号和EMG数据

    Args:
        E1: exerciese1的字典
        E2: exerciese2的字典
       
    Returns:
        [emg, label]: EMG数据（12列）和对应刺激的序号
    """
    print('merge dataset: s%de%d, s%de%d...'%(E1['subject'],E1['exercise'],E2['subject'],E2['exercise']),end='',flush=True)
    E1_emg = E1['emg']
    E2_emg = E2['emg']

    E1_label = E1['restimulus']
    E2_label = E2['restimulus']

    index1 =[]
    for i in range(len(E1_label)):
        if E1_label[i]!=0:
            index1.append(i)
    label1 = E1_label[index1,:]
    emg1 = E1_emg[index1,:]

    index2 =[]
    for i in range(len(E2_label)):
        if E2_label[i]!=0:
            index2.append(i)
    label2 = E2_label[index2,:]
    emg2 = E2_emg[index2,:]

    emg = np.vstack((emg1,emg2))
    label = np.vstack((label1,label2))
    label = label-1

    print('[ok]:emg shape:',end='')
    print(emg.shape,flush=True)
    return emg, label

def generate_image(emg,label):
    """将EMG数据切片，准备送入神经网络

    Args:
        emg: EMG数据（12列）
        label: 对应刺激的序号
       
    Returns:
        [imagedata, label]: 切片数据和对应刺激的序号
    """

    emg = emg*20000

    imageData=[]
    imageLabel=[]
    imageLength=200
    classes = 40

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if(label[j,:]==i):
                index.append(j)
                
        iemg = emg[index,:]
        length = math.floor((iemg.shape[0]-imageLength)/imageLength)
        #print("class ",i," number of sample: ",iemg.shape[0],length)
        print('\rgenerate emg matrix data...',end='',flush=True)
        print('[%d/%d]'%(i,classes),end='',flush=True)
        for j in range(length):
            subImage = iemg[imageLength*j:imageLength*(j+1),:]
            imageData.append(subImage)
            imageLabel.append(i)    
    imageData = np.array(imageData)
    imageLabel = np.array(imageLabel)
    print('\rgenerate emg matrix data...[ok]',imageData.shape,', ',imageLabel.shape,end='\n',flush=True)
    return imageData, imageLabel

def save_h5data(filename, data ,label):
    """将提取的数据暂存为h5文件

    Args:
        filename: 存储文件路径
        data: 存储数据
        label: 数据标签
    """
    with h5py.File(filename,'w') as f:
        f.create_dataset('featureData', data = data)  
        f.create_dataset('featureLabel', data = label)  
        f.close() 

def pretreatment_ninapro(db7_path, num_subject):
    """预处理数据集，仅提取EMG数据，并保存为h5py文件

    Args:
        db7_path: 数据集路径
        num_subject: 测试人数
    """
    for i in range(1, num_subject + 1):
        print('---subject%d---'%i,flush=True)
        # 读取exerciese1和exerciese2的数据
        e1_raw = get_ninapro(db7_path + 'Subject_{0}/'.format(i),'S{0}_E1_A1.mat'.format(i))
        e2_raw = get_ninapro(db7_path + 'Subject_{0}/'.format(i),'S{0}_E2_A1.mat'.format(i))
        # 合并数据
        emg, label = merge_exercise(e1_raw,e2_raw)
        # 截断数据生成训练所需的EMG数据片段
        emg_image, emg_label = generate_image(emg, label)
        # 保存EMG数据和label
        save_h5data('data/s{0}.h5'.format(i),emg_image, emg_label)