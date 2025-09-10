'''
-------------------------------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Learning socially compatible autonomous driving under safety-critical scenarios"
Author: Gaoyuan Kuang, Qingfan Wang, Jiajie Shen, Jinghe Lin, Xin Gao, Kun Ren, Jianqiang Wang, Shuo Feng, Bingbing Nie
Corresponding author: Shuo Feng (fshuo@tsinghua.edu.cn), Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------------------------------
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import warnings

# Define the random seed.
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# AIS calculation using empirical formula
def AIS_3_cal(HIC):
    AIS_3 = []
    for i in range(len(HIC)):
        hic = np.zeros(3)
        _HIC = min(max(HIC[i], 1), 2000)
        hic[0] = 1. / (1 + np.exp(1.54 + 200 / _HIC - 0.00650 * _HIC))  # AIS≥1
        hic[1] = 1. / (1 + np.exp(3.39 + 200 / _HIC - 0.00372 * _HIC))  # AIS≥3
        hic_i = int(2)
        while hic[int(hic_i - 1)] < 0.2: # We adopted 0.2 as an empirically defined probability threshold for classification decisions.
            hic_i = hic_i - 1
            if hic_i == 0:
                break

        AIS_3.append(hic_i)
    return np.array(AIS_3)

def AIS_cal(HIC, prob_output=False):
    AIS = []
    AIS_prob = []
    for i in range(len(HIC)):
        hic = np.zeros(5)
        ais_prob = np.zeros(6)
        _HIC = min(max(HIC[i], 1), 2000)
        hic[0] = 1. / (1 + np.exp(1.54 + 200 / _HIC - 0.00650 * _HIC))  # P(AIS≥1)
        ais_prob[0] =  1 - hic[0]

        hic[1] = 1. / (1 + np.exp(2.49 + 200 / _HIC - 0.00483 * _HIC))  # P(AIS≥2)
        ais_prob[1] = hic[0] - hic[1]

        hic[2] = 1. / (1 + np.exp(3.39 + 200 / _HIC - 0.00372 * _HIC))  # P(AIS≥3)
        ais_prob[2] = hic[1] - hic[2]

        hic[3] = 1. / (1 + np.exp(4.90 + 200 / _HIC - 0.00351 * _HIC))  # P(AIS≥4)
        ais_prob[3] = hic[2] - hic[3]

        hic[4] = 1. / (1 + np.exp(7.82 + 200 / _HIC - 0.00429 * _HIC))  # P(AIS≥5)
        ais_prob[4] = hic[3] - hic[4]

        ais_prob[5] = hic[4]

        hic_i = int(5)
        while hic[int(hic_i - 1)] < 0.2 :
            hic_i = hic_i - 1
            if hic_i == 0:
                break

        AIS.append(hic_i)
        AIS_prob.append(ais_prob)
    if prob_output:
        return np.array(AIS), np.array(AIS_prob)
    else:
        return np.array(AIS)

class CrashDataset(Dataset):
    def __init__(self, acc_file='./data/data_crashpulse.npy', att_file='./data/data_features.npy', transform=None):
        """
        Args:
            acc_file (str): File path of the crash pulse data (x_acc)
            att_file (str): File path of the feature data (x_att)
            transform (callable, optional): Optional data transformations
        """
        self.x_acc = np.load(acc_file)  # npdarray (5777, 2, 150)
        self.x_att = np.load(att_file)  # npdarray (5777, 9)
        
        # Data preprocessing: normalization, discretization, etc.
        self.x_acc[:, 0] = np.round((self.x_acc[:, 0] - self.x_acc[:, 0].min()) / (self.x_acc[:, 0].max() - self.x_acc[:, 0].min()) * 199)  # X-direction crash pulse
        self.x_acc[:, 1] = np.round((self.x_acc[:, 1] - self.x_acc[:, 1].min()) / (self.x_acc[:, 1].max() - self.x_acc[:, 1].min()) * 199)  # Y-direction crash pulse

        self.x_att[:, 0] = np.round((self.x_att[:, 0] - self.x_att[:, 0].min()) / (self.x_att[:, 0].max() - self.x_att[:, 0].min()) * 29)  # ego vehicle speed: 23.0~140.0km/h to 0~29(int)
        self.x_att[:, 1] = np.round((self.x_att[:, 1] - self.x_att[:, 1].min()) / (self.x_att[:, 1].max() - self.x_att[:, 1].min()) * 19)  # leading vehicle speed 10.0~120.0km/h to 0~19(int)
        self.x_att[:, 2] = self.x_att[:, 2] - 1  # six types of collision Overlap rate: 1~6 to 0~5(int)
        self.x_att[:, 3] = (self.x_att[:, 3] + 30) / 5  # collision angle: -30~30° to 0~12(int)
        self.x_att[:, 4] = np.round((self.x_att[:, 4] - self.x_att[:, 4].min()) / (self.x_att[:, 4].max() - self.x_att[:, 4].min()) * 5) # Mass of the leading vehicle: 900.0-3900.0kg to 0-5(int)
        # x_att[:, 5]-x_att[:, 7]:Belt usage(0,1), Airbag usage(0,1), Occupant size(0,1,2) are discrete variables, no processing is required

        # Target variable
        self.y_HIC = self.x_att[:, 8]
        self.y_AIS = AIS_cal(self.y_HIC)

        # Actual input features
        self.x_att = self.x_att[:, :8]

        self.transform = transform

    def Loss_expect_cal(self):
        # Subsequently, HIC corresponds to the MSE loss, and AIS corresponds to the classification loss.
        # Here, the expectations of both losses are calculated separately, assuming that HIC_pred and HIC_true are independently and identically distributed (i.i.d.), and AIS_pred and AIS_true are also i.i.d.
        _HIC = np.clip(self.y_HIC, 0, 2500)
        MSEloss_expect = 2 * np.std(_HIC) ** 2
        AIS_prob = np.bincount(self.y_AIS) / len(self.y_AIS)
        Classification_loss_expect = -np.log(np.sum(AIS_prob ** 2))
        return MSEloss_expect, Classification_loss_expect

    def __len__(self):
        return len(self.x_acc)

    def __getitem__(self, idx):
        # Retrieve data by index
        x_acc = self.x_acc[idx]
        x_att = self.x_att[idx]
        y_HIC = self.y_HIC[idx]
        y_AIS = self.y_AIS[idx]
        
        if self.transform:
            x_acc = self.transform(x_acc)
            x_att = self.transform(x_att)

        return torch.tensor(x_acc, dtype=torch.long), \
            torch.tensor(x_att, dtype=torch.long), \
            torch.tensor(y_HIC, dtype=torch.float32), \
            torch.tensor(y_AIS, dtype=torch.long)

if __name__ == '__main__':
    import time
    # TEST
    start_time = time.time()
    dataset = CrashDataset()
    print("Dataset loading time:", time.time() - start_time)

    train_size = 5000
    val_size = 500
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    batch_start_time = time.time()
    for i, batch in enumerate(train_loader):
        
        x_acc, x_att, y_HIC, y_AIS = batch
        if i == 0:
            print("x_acc shape:", x_acc.shape)
            print("x_att shape:", x_att.shape)
            print("The last three columns of the first four rows of x_att", x_att[:8, 5:])
            print("y_HIC shape:", y_HIC.shape)
            print("y_AIS shape:", y_AIS.shape)
    print("batch time:", time.time() - batch_start_time)