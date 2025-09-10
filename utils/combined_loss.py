'''
-------------------------------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Learning socially compatible autonomous driving under safety-critical scenarios"
Author: Gaoyuan Kuang, Qingfan Wang, Jiajie Shen, Jinghe Lin, Xin Gao, Kun Ren, Jianqiang Wang, Shuo Feng, Bingbing Nie
Corresponding author: Shuo Feng (fshuo@tsinghua.edu.cn), Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------------------------------
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, threshold=1000,  mse_overweight=1.0,  mse_norm_coef = 1200000, class_norm_coef = 1.25):
        """
            Initialize the CombinedLoss class.

            Args:
                threshold (float): Threshold for HIC weight allocation.
                mse_overweight (float): Overweight factor for the MSE loss (hyperparameter).
                mse_norm_coef (float): Normalization coefficient for the MSE loss.
                class_norm_coef (float): Normalization coefficient for the classification loss.
        """
        super(CombinedLoss, self).__init__()
        self.threshold = threshold
        self.ktanh = 0.01
        self.ksigmoid = 0.004
        self.mse_overweight = mse_overweight
        self.mse_norm_coef = mse_norm_coef
        self.class_norm_coef = class_norm_coef

    @staticmethod
    def AIS_prob_cal(HIC):
        """
        Compute the mapping from HIC to AIS classification probabilities.

        Args:
            HIC (torch.Tensor): Input tensor of HIC values with shape [batch_size].

        Returns:
            torch.Tensor: Output AIS probability logits with shape [batch_size, 6].
        """
        _HIC = torch.clamp(HIC, min=5.0, max=2000.0)

        hic = torch.zeros((_HIC.size(0), 5), device=HIC.device)
        hic[:, 0] = 1.0 / (1.0 + torch.exp(1.54 + 200.0 / _HIC - 0.00650 * _HIC))  # P(AIS≥1)
        hic[:, 1] = 1.0 / (1.0 + torch.exp(2.49 + 200.0 / _HIC - 0.00483 * _HIC))  # P(AIS≥2)
        hic[:, 2] = 1.0 / (1.0 + torch.exp(3.39 + 200.0 / _HIC - 0.00372 * _HIC))  # P(AIS≥3)
        hic[:, 3] = 1.0 / (1.0 + torch.exp(4.90 + 200.0 / _HIC - 0.00351 * _HIC))  # P(AIS≥4)
        hic[:, 4] = 1.0 / (1.0 + torch.exp(7.82 + 200.0 / _HIC - 0.00429 * _HIC))  # P(AIS≥5)

        ais_prob = torch.zeros((_HIC.size(0), 6), device=HIC.device)
        ais_prob[:, 0] = 1.0 - hic[:, 0]                   # P(AIS=0)
        ais_prob[:, 1] = hic[:, 0] - hic[:, 1]             # P(AIS=1)
        ais_prob[:, 2] = hic[:, 1] - hic[:, 2]             # P(AIS=2)
        ais_prob[:, 3] = hic[:, 2] - hic[:, 3]             # P(AIS=3)
        ais_prob[:, 4] = hic[:, 3] - hic[:, 4]             # P(AIS=4)
        ais_prob[:, 5] = hic[:, 4]                         # P(AIS=5)

        return ais_prob

    def weighted_function(self, x):
        """
        HIC weight allocation function that generates weights for both classification
        and MSE losses.

        Args:
            x (torch.Tensor): Tensor of HIC values with shape [batch_size].

        Returns:
            torch.Tensor: Classification loss weights and MSE loss weights with shape [batch_size].
        """
        tanh_part = 0.5 * (torch.tanh(self.ktanh * (x - self.threshold)) + 1)
        sigmoid_part = torch.sigmoid(self.ksigmoid * (x - self.threshold))
        y_class = torch.where(x <= self.threshold, tanh_part, sigmoid_part)
        y_mse = 1 - y_class
        return y_class, y_mse

    @staticmethod
    def cross_entropy_loss_from_prob(pred_ais_probs, true_ais):
        """
        Compute the cross-entropy loss based on empirical probabilities.

        Args:
            pred_ais_probs (torch.Tensor): Empirical probabilities with shape [batch_size, num_classes=6].
            true_ais (torch.Tensor): Ground truth class labels with shape [batch_size].

        Returns:
            batch_loss (torch.Tensor): Sample-wise cross-entropy loss with shape [batch_size].
        """

        true_class_probs = pred_ais_probs[torch.arange(true_ais.size(0)), true_ais]
        # loss = -log(P(AIS_i=AIS_ture_i))
        batch_loss = -torch.log(true_class_probs + 1e-12)
        return batch_loss

    def forward(self, pred_hic, true_hic, true_ais):
        """
        Compute the combined loss.

        Args:
            pred_hic (torch.Tensor): Model-predicted HIC values with shape [batch_size].
            true_hic (torch.Tensor): Ground truth HIC values with shape [batch_size].
            true_ais (torch.Tensor): Ground truth AIS classification labels with shape [batch_size].

        Returns:
            torch.Tensor: Combined loss value.
            torch.Tensor: Unweighted MSE loss (normalized).
            torch.Tensor: Unweighted classification loss (normalized).
        """

        # Call the weighting function to compute weights
        y_class, y_mse = self.weighted_function(true_hic)  # Both shapes are [batch_size]

        # Compute weighted classification loss
        pred_ais_probs = self.AIS_prob_cal(pred_hic)  # Shape: [batch_size, 6]
        class_losses = self.cross_entropy_loss_from_prob(pred_ais_probs, true_ais)  # Sample-level classification loss
        weighted_class_loss = (class_losses * y_class).mean()  # Mean classification loss after applying sample weights

        # Compute weighted MSE loss
        mse_losses = F.mse_loss(pred_hic, true_hic, reduction='none')  # Sample-level MSE loss
        weighted_mse_loss = (mse_losses * y_mse).mean()  # Mean MSE loss after applying sample weights

        # Combined loss: mse_overweight controls the relative importance of the MSE loss
        combined_loss = self.mse_overweight * weighted_mse_loss / self.mse_norm_coef + weighted_class_loss / self.class_norm_coef

        return combined_loss, mse_losses.mean() / self.mse_norm_coef, class_losses.mean() / self.class_norm_coef


if __name__ == '__main__':
    import numpy as np
    # Test the CombinedLoss class
    AIS_prob = np.array([0.477, 0.124, 0.122, 0.092, 0.051, 0.134], dtype=np.float32) 
    print(-np.log(np.sum(AIS_prob ** 2))) # 1.249
    criterion = CombinedLoss(class_norm_coef=-np.log(np.sum(AIS_prob ** 2)))   
    pred_hic = torch.tensor([0, 100, 1000.0, 2600.0], dtype=torch.float32)
    true_hic = torch.tensor([0, 300, 900.0, 1500.0], dtype=torch.float32)
    true_ais = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    loss = criterion(pred_hic, true_hic, true_ais)
    print(loss)
    #loss.backward()
