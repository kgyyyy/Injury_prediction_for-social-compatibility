'''
-------------------------------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Learning socially compatible autonomous driving under safety-critical scenarios"
Author: Gaoyuan Kuang, Qingfan Wang, Jiajie Shen, Jinghe Lin, Xin Gao, Kun Ren, Jianqiang Wang, Shuo Feng, Bingbing Nie
Corresponding author: Shuo Feng (fshuo@tsinghua.edu.cn), Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------------------------------
'''

import os
import time
import torch.utils.data as Data
import torch
import random
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix

from utils import backbones
from utils.load_data import load_data, AIS_cal, AIS_3_cal

import warnings
warnings.filterwarnings('ignore')


########## initial  ###########
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


##  1-Model training
if __name__ == "__main__":
    log = open('main_KD.txt', mode='a', encoding='utf-8')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    (x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra), (x_acc_val, x_att_val, y_HIC_val, y_AIS_val), (x_acc_tes, x_att_tes, y_HIC_tes, y_AIS_tes) = load_data()

    num_i = 0
    Epochs = 100
    for Batch_size in [32, 64, 128]:
        for Emb_size in [64, 128, 256]:
            for (ratio_E, ratio_D) in [(40000, 4000), (80000, 8000), (20000, 2000), (20000, 4000), ]:
                for Learning_rate in [0.01, 0.003, 0.001, 0.0003, 0.0001]:
                    print('')
                    print(num_i)
                    print(num_i, file=log)
                    num_i = num_i + 1
                    print(Batch_size, Emb_size, ratio_E, ratio_D, Learning_rate)
                    print(Batch_size, Emb_size, ratio_E, ratio_D, Learning_rate, file=log)

                    Level_Size, K_size, Emb_size_, Hidden_size, Dropout = 5, 5, 128, 64, 0.3
                    Num_Chans = [Hidden_size] * (Level_Size - 1) + [Emb_size_]
                    model_t = backbones.TCN(Emb_size_, Num_Chans, K_size, dropout=Dropout, emb_dropout=Dropout).cuda()
                    model_t.load_state_dict(torch.load('params/Best/C2S_best'))

                    model_s = backbones.S2S(Emb_size, )
                    model_s = model_s.to(device)

                    if Emb_size == Emb_size_:
                        pretrained_dict = torch.load('params/Best/C2S_best')
                        model_dict = model_s.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model_s.load_state_dict(model_dict)

                    criterion = nn.MSELoss().cuda()
                    optimizer = torch.optim.Adam(model_s.parameters(), lr=Learning_rate)

                    dataset_train = Data.TensorDataset(x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra)
                    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=Batch_size, shuffle=True)
                    dataset_val = Data.TensorDataset(x_acc_val, x_att_val, y_HIC_val, y_AIS_val)
                    loader_val = Data.DataLoader(dataset=dataset_val, batch_size=500, shuffle=True)

                    LossCurve_val, LossCurve_train = [], []
                    Best_accu = 0

                    for epoch in range(Epochs):
                        loss_batch = []
                        epoch_start_time = time.time()
                        model_s.train()
                        model_t.eval()
                        for step, (batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS) in enumerate(loader_train):
                            batch_x_acc = batch_x_acc.to(device)
                            batch_x_att = batch_x_att.to(device)
                            batch_y_HIC = batch_y_HIC.to(device)
                            batch_y_AIS = batch_y_AIS.to(device)

                            pred_HIC_s, pred_D_s, pred_E_s = model_s(batch_x_att)
                            pred_HIC_t, pred_D_t, pred_E_t = model_t(batch_x_acc, batch_x_att)
                            pred_HIC_t, pred_D_t, pred_E_t = pred_HIC_t.detach(), pred_D_t.detach(), pred_E_t.detach()

                            loss_pred = criterion(pred_HIC_s, batch_y_HIC)
                            loss_KD_E = criterion(pred_E_s, pred_E_t) if Emb_size == Emb_size_ else 0
                            loss_KD_D = criterion(pred_D_s, pred_D_t)
                            loss = loss_pred + ratio_E * loss_KD_E + ratio_D * loss_KD_D
                            # print(loss_pred, loss_KD_E, loss_KD_D)
                            # print(loss_pred, ratio_E * loss_KD_E, ratio_D * loss_KD_D)

                            loss_batch.append(np.around(loss.detach().data.cpu().numpy(), decimals=3))

                            optimizer.zero_grad()
                            loss.backward(retain_graph=True)
                            optimizer.step()

                        loss = np.around(np.mean(loss_batch), decimals=3)
                        LossCurve_train.append(loss)
                        print('TRAIN | Epoch: ', epoch + 1, '/', Epochs, '| Loss:', loss, '| Epoch duration:',
                              round(time.time() - epoch_start_time, 2), 's')
                        print('TRAIN | Epoch: ', epoch + 1, '/', Epochs, '| Loss:', loss, '| Epoch duration:',
                              round(time.time() - epoch_start_time, 2), 's', file=log)

                        loss_batch = []
                        epoch_start_time = time.time()
                        model_s.eval()
                        for step, (batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS) in enumerate(loader_val):
                            batch_x_acc = batch_x_acc.to(device)
                            batch_x_att = batch_x_att.to(device)
                            batch_y_HIC = batch_y_HIC.to(device)
                            batch_y_AIS = batch_y_AIS.to(device)

                            pred_HIC_s, pred_D_s, pred_E_s = model_s(batch_x_att)
                            pred_HIC_t, pred_D_t, pred_E_t = model_t(batch_x_acc, batch_x_att)
                            pred_HIC_t, pred_D_t, pred_E_t = pred_HIC_t.detach(), pred_D_t.detach(), pred_E_t.detach()

                            loss_pred = criterion(pred_HIC_s, batch_y_HIC)
                            loss_KD_E = criterion(pred_E_s, pred_E_t) if Emb_size == Emb_size_ else 0
                            loss_KD_D = criterion(pred_D_s, pred_D_t)
                            loss = loss_pred + ratio_E * loss_KD_E + ratio_D * loss_KD_D

                            loss_batch.append(np.around(loss.detach().data.cpu().numpy(), decimals=3))

                        loss = np.around(np.mean(loss_batch), decimals=3)
                        LossCurve_val.append(loss)
                        print('Val | Loss:', loss, '| Val duration:', round(time.time() - epoch_start_time, 3), 's')
                        print('Val | Loss:', loss, '| Val duration:', round(time.time() - epoch_start_time, 3), 's',
                              file=log)

                        ## 6 classification
                        pred, true = AIS_cal(pred_HIC_s.cpu().detach().numpy()), batch_y_AIS.cpu().detach().numpy()
                        accu = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
                        conf_mat = confusion_matrix(true, pred)
                        # G_mean = geometric_mean_score(true, pred)
                        # report = classification_report_imbalanced(true, pred, digits=3)

                        print('Val | Accuracy: ' + str(np.around(accu, 1)) + '%')
                        print(conf_mat)
                        print('')

                        print('Val | Accuracy: ' + str(np.around(accu, 1)) + '%', file=log)
                        # print('Val | G-mean: ' + str(np.around(G_mean, 3)), file=log)
                        print(conf_mat, file=log)
                        # print(report, file=log)
                        print('', file=log)

                        # Early stopping
                        if epoch == 10 and np.around(accu, 1) == 34.6:
                            print('Early Stop!')
                            print('Early Stop!', file=log)
                            break

                        if accu > Best_accu:
                            Best_accu = accu
                            torch.save(model_s.state_dict(), 'params/KD_%d' % num_i)


                        ## 3 classification
                        pred, true = AIS_3_cal(pred_HIC_s.cpu().detach().numpy()), AIS_3_cal(batch_y_HIC.cpu().detach().numpy())
                        accu = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
                        conf_mat = confusion_matrix(true, pred)
                        # G_mean = geometric_mean_score(true, pred)
                        # report = classification_report_imbalanced(true, pred, digits=3)

                        print('Val | Accuracy_3: ' + str(np.around(accu, 1)) + '%')
                        print(conf_mat)
                        print('')

                        print('Val | Accuracy_3: ' + str(np.around(accu, 1)) + '%', file=log)
                        # print('Val | G-mean: ' + str(np.around(G_mean, 3)), file=log)
                        print(conf_mat, file=log)
                        # print(report, file=log)
                        print('', file=log)


                        # for p in optimizer.param_groups:
                        #     p['lr'] = Learning_rate / np.sqrt((epoch + 5) / 5)

                        if epoch > 0 and epoch % 35 == 0:
                            for p in optimizer.param_groups:
                                p['lr'] *= 0.33


                        # Early stopping
                        if len(LossCurve_val) > 5 and LossCurve_val[-1] > LossCurve_val[-2] > LossCurve_val[-3] > LossCurve_val[-4]> LossCurve_val[-5]:
                            print('Early Stop!')
                            print('Early Stop!', file=log)
                            break

    log.close()



# #  2-performance evaluation
# if __name__ == "__main__":
#     # Load the data.
#     (x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra), (x_acc_val, x_att_val, y_HIC_val, y_AIS_val), \
#     (x_acc_tes, x_att_tes, y_HIC_tes, y_AIS_tes) = load_data()
#
#     Batch_size, Emb_size, ratio_E, ratio_D, Learning_rate = 32, 64, 20000, 2000, 0.001
#
#     model_s = backbones.S2S(Emb_size, ).to(device)
#     model_s.load_state_dict(torch.load('params/Best/KD_best_13'))
#
#
#     dataset_val = Data.TensorDataset(x_acc_val, x_att_val, y_HIC_val, y_AIS_val)
#     loader_val = Data.DataLoader(dataset=dataset_val, batch_size=500, shuffle=True)
#     dataset_test = Data.TensorDataset(x_acc_val, x_att_val, y_HIC_val, y_AIS_val)
#     loader_test = Data.DataLoader(dataset=dataset_test, batch_size=222, shuffle=True)
#
#
#     model_s.eval()
#     for step, (batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS) in enumerate(loader_val):
#         batch_x_acc = batch_x_acc.to(device)
#         batch_x_att = batch_x_att.to(device)
#         batch_y_HIC = batch_y_HIC.to(device)
#         batch_y_AIS = batch_y_AIS.to(device)
#         prediction, _, _ = model_s(batch_x_att)
#
#     prediction = prediction.cpu().detach().numpy()
#     batch_y_HIC = batch_y_HIC.cpu().detach().numpy()
#     batch_y_AIS = batch_y_AIS.cpu().detach().numpy()
#     prediction[prediction > 2500] = 2500
#     batch_y_HIC[batch_y_HIC > 2500] = 2500
#
#     # HIC
#     from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
#
#     print(np.sqrt(mean_squared_error(batch_y_HIC, prediction)))
#     print(mean_absolute_error(batch_y_HIC, prediction))
#     print(r2_score(batch_y_HIC, prediction))
#     # print(mean_absolute_percentage_error(batch_y_HIC, prediction))
#     print('')
#
#     import matplotlib.pyplot as plt
#
#     x1, x2 = batch_y_HIC, prediction
#     y = batch_y_AIS
#
#     cdict = {1: '#277DA1', 2: '#4D908E', 3: '#90BE6D', 4: '#F9C74F', 5: '#F3722C', 6: '#F94144'}  # colour scheme
#     fig, ax = plt.subplots(figsize=(6, 6))
#     j = 1
#     for g in np.unique(y):
#         ix = np.where(y == g)
#         ax.scatter(x1[ix], x2[ix], c=cdict[j], label="y = " + str(g), s=30, alpha=0.6)
#         j = j + 1
#     ax.legend(['AIS 0', 'AIS 1', 'AIS 2', 'AIS 3', 'AIS 4', 'AIS 5', ], fontsize=16, framealpha=0.5)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.xlim((-100, 2600))
#     plt.ylim((-100, 2600))
#     plt.xlabel("HIC_Target", family='Arial', fontsize=19)
#     plt.ylabel("HIC_Prediction", family='Arial', fontsize=19)
#     # plt.rcParams.update({'font.size': 50, 'font.family': 'Arial'})
#     # plt.axis('equal')
#     plt.subplots_adjust(left=0.18, bottom=0.18, top=0.94, right=0.94, wspace=0.25, hspace=0.25)
#     # plt.show()
#     plt.savefig('image/HIC_KD.png')
#
#     # 6 classification
#     pred, true = AIS_cal(prediction), batch_y_AIS
#     accu = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
#     conf_mat = confusion_matrix(true, pred)
#     G_mean = geometric_mean_score(true, pred)
#     report = classification_report_imbalanced(true, pred, digits=3)
#
#     print('Val | Accuracy: ' + str(np.around(accu, 1)) + '%')
#     print(conf_mat)
#     print(G_mean)
#     print(report)
#     print('')
#
#     # 3 classification
#     pred, true = AIS_3_cal(prediction), AIS_3_cal(batch_y_HIC)
#     accu = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
#     conf_mat = confusion_matrix(true, pred)
#     G_mean = geometric_mean_score(true, pred)
#     report = classification_report_imbalanced(true, pred, digits=3)
#
#     print('Val | Accuracy_3: ' + str(np.around(accu, 1)) + '%')
#     print(conf_mat)
#     print(G_mean)
#     print(report)
#     print('')
#
#
#
##  3-Model inference efficiency evaluation
# if __name__ == "__main__":
#     # Load the data.
#     (x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra), (x_acc_val, x_att_val, y_HIC_val, y_AIS_val), \
#     (x_acc_tes, x_att_tes, y_HIC_tes, y_AIS_tes) = load_data()
#
#     # Load the data.
#     (x_acc_tra, x_att_tra, y_HIC_tra, y_AIS_tra), (x_acc_val, x_att_val, y_HIC_val, y_AIS_val), \
#     (x_acc_tes, x_att_tes, y_HIC_tes, y_AIS_tes) = load_data()
#
#     Batch_size, Emb_size, ratio_E, ratio_D, Learning_rate = 32, 64, 20000, 2000, 0.001
#
#     model = backbones.S2S(Emb_size, ).cuda()
#     model.load_state_dict(torch.load('params/Best/KD_best_13'))
#
#     dataset_val = Data.TensorDataset(x_acc_val, x_att_val, y_HIC_val, y_AIS_val)
#     loader_val = Data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=True)
#
#     model.eval()
#     Time = []
#     for step, (batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS) in enumerate(loader_val):
#         batch_x_acc = batch_x_acc.cuda()
#         batch_x_att = batch_x_att.cuda()
#         batch_y_HIC = batch_y_HIC.cuda()
#         batch_y_AIS = batch_y_AIS.cuda()
#         start_time = time.time()
#         prediction, _, _ = model(batch_x_att)
#         if step > 10:
#             Time.append(time.time() - start_time)
#
#     print('Time mean:', np.mean(Time))
#     print('Time std:', np.std(Time))
#
#     from thop import profile
#     macs, params = profile(model, inputs=(x_att_val[:1, :].cuda(), ))
#     print("FLOPs(G): %.2fM" % (macs / 1e6), "Params(M): %.2fM" % (params / 1e6))
