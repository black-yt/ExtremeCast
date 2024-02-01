import torch

def SortIndex(input):
    '''
    Get the sorting ranking corresponding to each element in tensor.
    Args:
        input (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: Tensor of the same size as input. The ensemble boosted predictions.
    Example:
        input=torch.tensor([1.2, 1.5, 0.8, 0.9])
        return torch.tensor([2, 3, 0, 1])
    '''
    sorted_indices = torch.argsort(input, dim=-1)
    ranks = torch.argsort(sorted_indices, dim=-1)
    return ranks


def ExEnsemble(pred, ensembles_nums=50, ensembles_scale=1, device='gpu'):
    '''
    Apply ExEnsemble to the predictions.
    Args:
        pred (torch.Tensor): Tensor of size [B, C, H, W]. The input predictions.
        ensembles_nums (int): Number of ensembles (default: 50).
        ensembles_scale (float or torch.Tensor): Scaling factor for ensembles. It can be a real number or a tensor of size [B, C, H, W] (default: 1).
    Returns:
        torch.Tensor: Tensor of size [B, C, H, W]. The ensemble boosted predictions.
    '''

    if device == 'cpu':
        original_device = pred.device
        pred = pred.cpu()
        try:
            ensembles_scale = ensembles_scale.cpu()
        except:
            pass

    B, C, H, W = pred.shape

    scale = ensembles_scale * torch.ones_like(pred) # [B, C, H, W]

    # SortIndex()
    idx = SortIndex(pred.flatten(2,3)) # [B, C, H*W]
    
    # Sample()
    pred = pred.unsqueeze(2) # [B, C, 1, H, W]
    scale = scale.unsqueeze(2) # [B, C, 1, H, W]
    disturbance = torch.randn(B, C, ensembles_nums, H, W, device=pred.device) * scale 
    ens = pred + disturbance # [B, C, ensembles_nums, H, W]

    # Sort()
    sorted_ens, _ = torch.sort(ens.flatten(2,4)) # [B, C, ensembles_nums*H*W]
    sorted_ens = sorted_ens.reshape(B, C, H*W, ensembles_nums) # [B, C, H*W, ensembles_nums]

    # Partition() and Median()
    k = int(0.5 * ensembles_nums) # ensembles_nums / 2
    sorted_ens_mid, _ = torch.kthvalue(sorted_ens, k, -1) # [B, C, H*W]

    # GetByIndex()
    ens_from_idx = torch.gather(sorted_ens_mid, dim=-1, index=idx) # [B, C, H*W]
    out = ens_from_idx.reshape(B, C, H, W) # [B, C, H, W]

    if device == 'cpu':
        return out.to(original_device)

    return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    pred = torch.randn(1, 69, 721, 1440).to(device)
    std = torch.randn(1, 69, 721, 1440).to(device)
    out = ExEnsemble(pred, 50, std)
    print(out.shape)