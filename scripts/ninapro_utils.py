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

def find_sample_range(dataset, stimulus):
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
    