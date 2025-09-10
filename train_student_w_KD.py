'''
-------------------------------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Learning socially compatible autonomous driving under safety-critical scenarios"
Author: Gaoyuan Kuang, Qingfan Wang, Jiajie Shen, Jinghe Lin, Xin Gao, Kun Ren, Jianqiang Wang, Shuo Feng, Bingbing Nie
Corresponding author: Shuo Feng (fshuo@tsinghua.edu.cn), Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------------------------------
'''

import os
import time
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from utils import models, dataset_prepare
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, mean_squared_error #root_mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Random seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train(student_model, teacher_model, loader, optimizer, criterion, ratio_E, ratio_D, device):
    student_model.train()
    teacher_model.eval()
    loss_batch = []

    for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
        batch_x_acc = batch_x_acc.to(device)
        batch_x_att = batch_x_att.to(device)
        batch_y_HIC = batch_y_HIC.to(device)

        # Forward propagation of the Student and Teacher model
        pred_HIC_s, pred_D_s, pred_E_s = student_model(batch_x_att) # (batch_size,), (batch_size, 16), (batch_size, num_channels[-1]=128)
        with torch.no_grad():
            pred_HIC_t, pred_D_t, pred_E_t = teacher_model(batch_x_acc, batch_x_att[:, 5:]) # (batch_size,), (batch_size, 16), (batch_size, num_channels[-1]=128)

        # Distillation loss and Prediction loss
        loss_pred = criterion(pred_HIC_s, batch_y_HIC)
        loss_KD_E = criterion(pred_E_s, pred_E_t)
        loss_KD_D = criterion(pred_D_s, pred_D_t)

        loss = loss_pred + ratio_E * loss_KD_E + ratio_D * loss_KD_D

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())

    return np.mean(loss_batch)

def valid(student_model, teacher_model, loader, criterion, ratio_E, ratio_D, device):
    student_model.eval()
    teacher_model.eval()
    loss_batch = []
    all_HIC_preds = []
    all_HIC_trues = []
    all_AIS_trues = []

    with torch.no_grad():
        for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
            batch_x_acc, batch_x_att, batch_y_HIC = batch_x_acc.to(device), batch_x_att.to(device), batch_y_HIC.to(device)

            pred_HIC_s, pred_D_s, pred_E_s = student_model(batch_x_att)
            pred_HIC_t, pred_D_t, pred_E_t = teacher_model(batch_x_acc, batch_x_att[:, 5:])

            loss_pred = criterion(pred_HIC_s, batch_y_HIC)
            loss_KD_E = criterion(pred_E_s, pred_E_t)
            loss_KD_D = criterion(pred_D_s, pred_D_t)

            loss = loss_pred + ratio_E * loss_KD_E + ratio_D * loss_KD_D

            loss_batch.append(loss.item())

            all_HIC_preds.append(pred_HIC_s.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
            all_AIS_trues.append(batch_y_AIS.numpy())

    avg_loss = np.mean(loss_batch)
    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)
    AIS_preds = dataset_prepare.AIS_cal(HIC_preds)
    AIS_trues = np.concatenate(all_AIS_trues)
    accuracy = 100. * (1 - np.count_nonzero(AIS_preds - AIS_trues) / len(AIS_trues))
    # rmse = root_mean_squared_error(HIC_trues, HIC_preds)
    rmse = mean_squared_error(HIC_trues, HIC_preds, squared=False)

    # conf_mat = confusion_matrix(AIS_trues, AIS_preds)
    # G_mean = geometric_mean_score(AIS_trues, AIS_preds)
    # report = classification_report_imbalanced(AIS_trues, AIS_preds, digits=3)

    return avg_loss, accuracy, rmse

if __name__ == "__main__":

    # Training Hyperparameters
    Epochs = 500
    Batch_size = 128
    Learning_rate = 0.005
    Learning_rate_min = 1e-6
    patience = 10  # Early stopping
    ratio_E = 20000
    ratio_D = 4000

    # Model Hyperparameters
    Emb_size_teacher = 128  # Embedding size of the teacher model
    Emb_size = 128 # Embedding size of the student model
    Level_Size = 5
    K_size = 5
    Hidden_size = 64
    Dropout = 0.3
    Num_Chans_teacher = [Hidden_size] * (Level_Size - 1) + [Emb_size_teacher]
    Num_Chans_student = [Hidden_size] * (Level_Size - 1) + [Emb_size]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset Loading
    dataset = dataset_prepare.CrashDataset()
    train_size = 5000
    val_size = 500
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, len(dataset) - train_size - val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    # Load the teacher model
    teacher_model = models.teacher_model(Emb_size_teacher, Num_Chans_teacher, kernel_size=K_size, dropout=Dropout).to(device)
    teacher_weights_path = './ckpt/teacher_best.pth'
    if os.path.exists(teacher_weights_path):
        teacher_model.load_state_dict(torch.load(teacher_weights_path))
        teacher_model.eval()
        print("Teacher model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Teacher model weights not found at {teacher_weights_path}.")

    # Load the student model
    student_model = models.student_model(Emb_size).to(device)
    if Emb_size == Emb_size_teacher:
        pretrained_dict = torch.load(teacher_weights_path)
        model_dict = student_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        student_model.load_state_dict(model_dict)
        print("Pretrained weights partially loaded into Student model.")
    else:
        print("Embedding sizes do not match between Teacher and Student models. No weights transferred.")

    # Load the optimizer and learning rate scheduler
    optimizer = optim.AdamW(student_model.parameters(), lr=Learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # Define the loss function
    criterion = nn.MSELoss().to(device)
    
    Best_accu = 0
    LossCurve_val, LossCurve_train = [], []

    # Training loop
    for epoch in range(Epochs):
        epoch_start_time = time.time()

        # Training
        train_loss = train(student_model, teacher_model, train_loader, optimizer, criterion, ratio_E, ratio_D, device)
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f}")

        # Validation
        val_loss, val_accuracy, val_rmse =  valid(student_model, teacher_model, val_loader, criterion, ratio_E, ratio_D, device)
        LossCurve_val.append(val_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.1f}% | RMSE: {val_rmse:.1f}")

        # Learning rate adjustment
        scheduler.step()

        # Save the best model
        if val_accuracy > Best_accu:
            Best_accu = val_accuracy
            if Best_accu > 80:
                torch.save(student_model.state_dict(), './ckpt/student_w_KD_best.pth')
                print(f"Best model saved with val accuracy: {Best_accu:.1f}%")

        # Early stopping logic
        if len(LossCurve_val) > patience:
            recent_losses = LossCurve_val[-patience:]
            if all(recent_losses[i] < recent_losses[i + 1] for i in range(len(recent_losses) - 1)):
                print(f"Early Stop at epoch: {epoch + 1}! Best Val accuracy: {Best_accu:.1f}%")
                break

        print(f"Epoch {epoch+1}/{Epochs} | Time: {time.time()-epoch_start_time:.2f}s")

    print(f"Training Finished!! Best Val accuracy: {Best_accu:.1f}%")
