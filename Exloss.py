import torch
from torch.functional import F

def Exloss(pred, target, up_th=0.9, down_th=0.1, lamda_underestimate=1.2, lamda_overestimate=1.0, lamda=1.0):
    '''
    up_th: percentile threshold of maximum value
    down_th: percentile threshold of minimum value
    lamda_underestimate: The penalty when underestimating is greater than the penalty when overestimating
    lamda_overestimate: Penalty for overestimation
    lamda: weight of Exloss and MSE
    '''
    
    mse_loss = torch.mean((pred-target)**2)

    N, C, H, W = pred.shape
    # Get the 90% and 10% quantiles in target as the thresholds for extreme maximum and minimum values, denoted as tar_up and tar_down
    tar_up =  torch.quantile(target.view(N, C, H*W), q=up_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1
    tar_down =  torch.quantile(target.view(N, C, H*W), q=down_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1

    target_up_area = F.relu(target-tar_up) # The part of target that is greater than tar_up
    target_down_area = -F.relu(tar_down-target) # The part of target that is smaller than tar_down
    pred_up_area = F.relu(pred-tar_up) # The part of pred that is greater than tar_up
    pred_down_area = -F.relu(tar_down-pred) # The part of pred that is smaller than tar_down

    # Increase the loss weight for the underestimated part of pred (the maximum value prediction is too small, the minimum value prediction is too large)
    loss_up = lamda_underestimate*(target_up_area-pred_up_area)*F.relu(target_up_area-pred_up_area)+\
              lamda_overestimate*(pred_up_area-target_up_area)*F.relu(pred_up_area-target_up_area)
    loss_down = lamda_overestimate*(target_down_area-pred_down_area)*F.relu(target_down_area-pred_down_area)+\
                lamda_underestimate*(pred_down_area-target_down_area)*F.relu(pred_down_area-target_down_area)
    loss_up = torch.mean(loss_up)
    loss_down = torch.mean(loss_down)
    ex_loss = (loss_up + loss_down)/(1-up_th+down_th)

    loss_all = mse_loss + lamda*ex_loss

    # print("all_loss:", loss_all.item(), "mse_loss:", mse_loss.item(), "ex_loss", ex_loss.item())

    return loss_all

if __name__ == "__main__":
    pred = torch.randn(1,69,721,1440)
    target = torch.randn(1,69,721,1440)
    print(Exloss(pred, target))