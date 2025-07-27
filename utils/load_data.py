'''
-------------------------------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Learning socially compatible autonomous driving under safety-critical scenarios"
Author: Gaoyuan Kuang, Qingfan Wang, Jiajie Shen, Jinghe Lin, Xin Gao, Kun Ren, Jianqiang Wang, Shuo Feng, Bingbing Nie
Corresponding author: Shuo Feng (fshuo@tsinghua.edu.cn), Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------------------------------
'''

import os
import torch
import random
import numpy as np



########## initial  ###########
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def AIS_3_cal(HIC):
    AIS_3 = []
    for i in range(len(HIC)):
        hic = np.zeros(3)
        hic[0] = 1. / (1 + np.exp(1.54 + 200 / HIC[i] - 0.00650 * HIC[i]))  # AIS≥1的公式
        hic[1] = 1. / (1 + np.exp(3.39 + 200 / HIC[i] - 0.00372 * HIC[i]))  # AIS≥3的公式

        hic_i = 2
        try:
            while hic[int(hic_i - 1)] < 0.2:
                hic_i = hic_i - 1
        except:
            hic_i = 0
        AIS_3.append(hic_i)

    return np.array(AIS_3)


def AIS_cal(HIC):
    AIS = []
    for i in range(len(HIC)):
        hic = np.zeros(5)
        hic[0] = 1. / (1 + np.exp(1.54 + 200 / HIC[i] - 0.00650 * HIC[i]))  # AIS≥1的公式
        hic[1] = 1. / (1 + np.exp(2.49 + 200 / HIC[i] - 0.00483 * HIC[i]))  # AIS≥2的公式
        hic[2] = 1. / (1 + np.exp(3.39 + 200 / HIC[i] - 0.00372 * HIC[i]))  # AIS≥3的公式
        hic[3] = 1. / (1 + np.exp(4.90 + 200 / HIC[i] - 0.00351 * HIC[i]))  # AIS≥4的公式
        hic[4] = 1. / (1 + np.exp(7.82 + 200 / HIC[i] - 0.00429 * HIC[i]))  # AIS≥5的公式

        hic_i = 5
        try:
            while hic[int(hic_i - 1)] < 0.2:
                hic_i = hic_i - 1
        except:
            hic_i = 0
        AIS.append(hic_i)

    return np.array(AIS)


def load_data():
    # Load the data.
    x_acc = np.load('data/data_crashpulse.npy')
    x_att = np.load('data/data_features.npy')

    x_acc[:, 0] = np.round((x_acc[:, 0] - x_acc[:, 0].min()) / (x_acc[:, 0].max() - x_acc[:, 0].min()) * 199)
    x_acc[:, 1] = np.round((x_acc[:, 1] - x_acc[:, 1].min()) / (x_acc[:, 1].max() - x_acc[:, 1].min()) * 199)

    x_att[:, 0] = np.round((x_att[:, 0] - x_att[:, 0].min()) / (x_att[:, 0].max() - x_att[:, 0].min()) * 29)
    x_att[:, 1] = np.round((x_att[:, 1] - x_att[:, 1].min()) / (x_att[:, 1].max() - x_att[:, 1].min()) * 19)
    x_att[:, 2] = x_att[:, 2] - 1
    x_att[:, 3] = (x_att[:, 3] + 30) / 5
    x_att[:, 4] = np.round((x_att[:, 4] - x_att[:, 4].min()) / (x_att[:, 4].max() - x_att[:, 4].min()) * 5)
    x_att[:, 5] = x_att[:, 5]
    x_att[:, 6] = x_att[:, 6]
    x_att[:, 7] = x_att[:, 7]

    y_HIC = x_att[:, 8]
    y_AIS = AIS_cal(y_HIC)

    x_att = x_att[:, :8]
    x_acc = torch.from_numpy(x_acc).long()
    x_att = torch.from_numpy(x_att).long()
    y_HIC = torch.from_numpy(y_HIC).float()
    y_AIS = torch.from_numpy(y_AIS).long()

    # print(len(x_acc))
    shuffle = np.random.permutation(len(x_acc))
    x_acc = x_acc[shuffle]
    x_att = x_att[shuffle]
    y_HIC = y_HIC[shuffle]
    y_AIS = y_AIS[shuffle]

    train_num, val_num, test_num = 5000, 500, 222
    x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra = x_acc[:train_num], x_att[:train_num], y_HIC[:train_num], \
                                                 y_AIS[:train_num]
    x_acc_val, x_att_val, y_HIC_val, y_AIS_val = x_acc[train_num: train_num + val_num], \
                                                 x_att[train_num: train_num + val_num], \
                                                 y_HIC[train_num: train_num + val_num], \
                                                 y_AIS[train_num: train_num + val_num]
    x_acc_tes, x_att_tes, y_HIC_tes, y_AIS_tes = x_acc[-test_num:], x_att[-test_num:], y_HIC[-test_num:], \
                                                 y_AIS[-test_num:]

    return (x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra), (x_acc_val, x_att_val, y_HIC_val, y_AIS_val), \
           (x_acc_tes, x_att_tes, y_HIC_tes, y_AIS_tes)
