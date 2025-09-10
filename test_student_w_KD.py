'''
-------------------------------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Learning socially compatible autonomous driving under safety-critical scenarios"
Author: Gaoyuan Kuang, Qingfan Wang, Jiajie Shen, Jinghe Lin, Xin Gao, Kun Ren, Jianqiang Wang, Shuo Feng, Bingbing Nie
Corresponding author: Shuo Feng (fshuo@tsinghua.edu.cn), Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------------------------------
'''

# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import  DataLoader
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, mean_squared_error # root_mean_squared_error

from utils import models
from utils.dataset_prepare import CrashDataset, AIS_cal, AIS_3_cal
import warnings

warnings.filterwarnings('ignore')

# Define the random seed.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    model.eval()
    all_HIC_preds = []
    all_HIC_trues = []
    
    with torch.no_grad():
        for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
            batch_x_att = batch_x_att.to(device)
            batch_pred_HIC, _, _ = model(batch_x_att)
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.numpy())

    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)

    return HIC_preds, HIC_trues

if __name__ == "__main__":
    # Hyperparameter settings and dataset loading
    Emb_size = 128
    Batch_size = 64
    dataset = CrashDataset()
    test_dataset1 = torch.load("./data/val_dataset.pt")
    test_dataset2 = torch.load("./data/test_dataset.pt")
    # test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])
    test_dataset = test_dataset1
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    # Loading the model
    model = models.student_model(Emb_size).to(device)
    model.load_state_dict(torch.load("./ckpt/student_w_KD_best.pth"))

    # Test model
    HIC_preds, HIC_trues = test(model, test_loader)
    HIC_preds[HIC_preds > 2500] = 2500  # An upper bound is set, since excessively high HIC values (>2000, generally indicating critical injury) have no practical significance.
    HIC_trues[HIC_trues > 2500] = 2500

    # HIC evaluation metrics
    # rmse = root_mean_squared_error(HIC_trues, HIC_preds)
    rmse = mean_squared_error(HIC_trues, HIC_preds, squared=False)

    mae = mean_absolute_error(HIC_trues, HIC_preds)
    r2 = r2_score(HIC_trues, HIC_preds)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Coefficient of Determination (R2):", r2)

    # AIS-6C evaluation metrics
    pred, true = AIS_cal(HIC_preds), AIS_cal(HIC_trues)
    accu_6c = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
    conf_mat_6c = confusion_matrix(true, pred)
    G_mean_6c = geometric_mean_score(true, pred)
    report_6c = classification_report_imbalanced(true, pred, digits=3)
    print('AIS-6C Accuracy: ' + str(np.around(accu_6c, 1)) + '%')
    print(conf_mat_6c)
    print(G_mean_6c)
    print(report_6c)

    # AIS-3C evaluation metrics
    pred, true = AIS_3_cal(HIC_preds), AIS_3_cal(HIC_trues)
    accu_3c = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
    conf_mat_3c = confusion_matrix(true, pred)
    G_mean_3c = geometric_mean_score(true, pred)
    report_3c = classification_report_imbalanced(true, pred, digits=3)
    print('AIS-3C Accuracy: ' + str(np.around(accu_3c, 1)) + '%')
    print(conf_mat_3c)
    print(G_mean_3c)
    print(report_3c)

    markdown_content = f"""
# Student Model with KD Results

## HIC Metrics
- **RMSE**: {rmse:.4f}
- **MAE**: {mae:.4f}
- **R2**: {r2:.4f}

## AIS-6C Metrics
- **Accuracy**: {accu_6c:.2f}%
- **G-Mean**: {G_mean_6c:.4f}
- **Confusion Matrix**:
```
{conf_mat_6c}
```
- **Classification Report**:
```
{report_6c}
```

## AIS-3C Metrics
- **Accuracy**: {accu_3c:.2f}%
- **G-Mean**: {G_mean_3c:.4f}
- **Confusion Matrix**:
```
{conf_mat_3c}
```
- **Classification Report**:
```
{report_3c}
```
"""

    file_path = "./results/student_model_w_KD_results.md"
    with open(file_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Results written to {file_path}")

    # Plotting
    import os
    os.makedirs('results', exist_ok=True)

    x1, x2 = HIC_trues, HIC_preds
    y = AIS_cal(HIC_trues)

    cdict = {
        0: '#277DA1',
        1: '#4D908E',
        2: '#90BE6D',
        3: '#F9C74F',
        4: '#F3722C',
        5: '#F94144'
    }

    fig, ax = plt.subplots(figsize=(4, 4))

    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(x1[ix], x2[ix], c=cdict.get(int(g), '#999999'),
                   label=f"{int(g)}", s=30, alpha=0.6)

        leg = ax.legend(title="AIS", fontsize=16, framealpha=0.5,
                    ncol=2, columnspacing=0.02, handletextpad=0.02,
                    prop={'family': 'Arial', 'size': 16},loc="upper left")

    plt.setp(leg.get_title(), fontsize=16, family='Arial')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks([0, 1250, 2500], fontsize=16, fontfamily='Arial')
    plt.yticks([0, 1250, 2500], fontsize=16, fontfamily='Arial', rotation=90)
    plt.xlim((-100, 2600))
    plt.ylim((-100, 2600))
    plt.xlabel("Target HIC", family='Arial', fontsize=19)
    plt.ylabel("Predicted HIC", family='Arial', fontsize=19)
    plt.subplots_adjust(left=0.18, bottom=0.18, top=0.94, right=0.94,
                        wspace=0.25, hspace=0.25)
    plt.savefig('results/student_model_w_KD.png', dpi=300)
    plt.show()

