'''
@article{xu2024extremecast,
  title={ExtremeCast: Boosting Extreme Value Prediction for Global Weather Forecast},
  author={Xu, Wanghan and Chen, Kang and Han, Tao and Chen, Hao and Ouyang, Wanli and Bai, Lei},
  journal={arXiv preprint arXiv:2402.01295},
  year={2024}
}
'''

import torch
import numpy as np

# Download npy files from https://drive.google.com/drive/folders/10pKdaag08BhtUd0OyNGJUeitCN3zlB-B?usp=sharing
SEDI_u10 = torch.from_numpy(np.load("./SEDI_u10.npy"))
SEDI_v10 = torch.from_numpy(np.load("./SEDI_v10.npy"))
SEDI_t2m = torch.from_numpy(np.load("./SEDI_t2m.npy"))
SEDI_msl = torch.from_numpy(np.load("./SEDI_msl.npy"))
SEDI_tp6h = torch.from_numpy(np.load("./SEDI_tp6h.npy"))
SEDI_t850 = torch.from_numpy(np.load("./SEDI_t850.npy"))
SEDI_q850 = torch.from_numpy(np.load("./SEDI_q850.npy"))
SEDI_z500 = torch.from_numpy(np.load("./SEDI_z500.npy"))

SEDI_q_list = [SEDI_u10, SEDI_v10, SEDI_t2m, SEDI_msl, SEDI_tp6h, SEDI_z500, SEDI_q850, SEDI_t850]


@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qs = 50
    qlim = 4
    qcut = 1
    n, c, h, w = pred.size()
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=target.dtype)
    P_tar = torch.quantile(target.view(n,c,h*w), q=qtile, dim=-1)
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=pred.dtype)
    P_pred = torch.quantile(pred.view(n,c,h*w), q=qtile, dim=-1)
    return torch.mean(torch.mean((P_pred - P_tar)/P_tar, dim=0), dim=0)


def RQE(pred_real, gt_real):
    return top_quantiles_error_torch(pred_real, gt_real).tolist()
    

def SEDI(pred_real, gt_real, month):
    sedi_list_all = []
    for ii in range(8):
        SEDI_q = SEDI_q_list[ii]
    
        sedi_list = []

        for i in range(4):
            SEDI_th = SEDI_q[month-1][i]

            gt_ex = (gt_real[:, ii] > SEDI_th).float()
            pred_ex = (pred_real[:, ii] > SEDI_th).float()

            FP = torch.sum((pred_ex-gt_ex == 1).float(), dim=(-2, -1)) # pred = 1; tar = 0
            TN = torch.sum((pred_ex+gt_ex == 0).float(), dim=(-2, -1)) # pred = 0; tar = 0
            TP = torch.sum((pred_ex+gt_ex == 2).float(), dim=(-2, -1)) # pred = 1; tar = 1
            FN = torch.sum((gt_ex-pred_ex == 1).float(), dim=(-2, -1)) # pred = 0; tar = 1

            FP = torch.where(FP == 0.0, torch.ones_like(FP), FP) # torch.Size([B])
            TN = torch.where(TN == 0.0, torch.ones_like(TN), TN)
            TP = torch.where(TP == 0.0, torch.ones_like(TP), TP)
            FN = torch.where(FN == 0.0, torch.ones_like(FN), FN)

            F = FP/(FP+TN)
            H = TP/(TP+FN)

            SEDI = (torch.log(F)-torch.log(H)-torch.log(1-F)+torch.log(1-H))/ \
                   (torch.log(F)+torch.log(H)+torch.log(1-F)+torch.log(1-H))
            
            SEDI = torch.mean(SEDI, dim=0)
            
            sedi_list.append(SEDI.item())
        sedi_list_all.append(sedi_list)
    
    return sedi_list_all


if __name__ == "__main__":
    # The 8 channels are u10, v10, t2m, msl, tp6h, z500, q850, t850 respectively.
    pred_real = torch.randn(4, 8, 721, 1440) # [Batch, Channel, H, W]
    tar_real = torch.randn(4, 8, 721, 1440)
    # "_real" represents the real value, not the normalized value. For example, for t2m, its value is about 270.
    month = 1 # January

    rqe = RQE(pred_real, tar_real) # A list of length 8, represents the RQE of u10, v10, t2m, msl, tp6h, z500, q850, t850 respectively.
    sedi = SEDI(pred_real, tar_real, month) # A list of length 8x4, represents the 90th, 95th, 98th, 99.5th SEDI of u10, v10, t2m, msl, tp6h, z500, q850, t850 respectively.

    print(rqe)
    print(sedi)
