import torch
from BoostEns import ExEnsemble
import numpy as np
from utils import load_model_d, load_model_g, merge_pred, normalize_numpy, inverse_normalize_torch, diffusion_inverse_transform, get_scale
from pic import pic_process

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Load model
    model_d = load_model_d()
    model_g = load_model_g()

    # Load data
    input1 = np.load('2017-12-31T18:00:00.npy') # [69, 721, 1440]
    input2 = np.load('2018-01-01T00:00:00.npy') # [69, 721, 1440]
    target = np.load('2018-01-01T06:00:00.npy') # [69, 721, 1440]
    climat = np.load('climatology-2018-01-01T06:00:00.npy') # [69, 721, 1440]

    # Add batch dimension
    input1 = np.expand_dims(input1, axis=0) # [1, 69, 721, 1440]
    input2 = np.expand_dims(input2, axis=0) # [1, 69, 721, 1440]
    target = np.expand_dims(target, axis=0) # [1, 69, 721, 1440]
    climat = np.expand_dims(climat, axis=0) # [1, 69, 721, 1440]

    # Normalize
    input1 = normalize_numpy(input1)
    input2 = normalize_numpy(input2)
    target = normalize_numpy(target)
    climat = normalize_numpy(climat)
    
    # Run Model_d
    print("[1] Model_d")
    input_data = np.concatenate([input1, input2], axis=1, dtype=np.float32)
    output_data = model_d.run(None, {'input':input_data})[0]
    model_d_pred = torch.from_numpy(output_data).to(device)

    # Run Model_g
    print("[2] Model_g")
    model_g = model_g.to(device)
    with torch.no_grad():
        model_output = model_g.sample(condition = model_d_pred[:,:69]).detach() #1,3,128,256
    diffusion_out = diffusion_inverse_transform(model_output)
    climat = torch.from_numpy(climat).to(device)
    model_g_pred = merge_pred(diffusion_out, model_d_pred, climat)

    # Run ExEnsemble
    print("[3] ExEnsemble")
    scale = get_scale(model_g_pred)
    # Setting device='gpu' can speed up ExEnsemble, but it also requires more memory.
    ens_pred = ExEnsemble(pred=model_g_pred[:,:69], ensembles_scale=scale, device='cpu')

    # Check mse to ensure correct operation. When running correctly, the MSE is about 0.0051
    target = torch.from_numpy(target).to(device)
    print("MSE:", float(torch.mean((ens_pred[:,:69]-target)**2).cpu()))

    # Inverse Normalize
    model_d_pred = inverse_normalize_torch(model_d_pred[:,:69])
    model_g_pred = inverse_normalize_torch(model_g_pred[:,:69])
    ens_pred = inverse_normalize_torch(ens_pred[:,:69])

    # Visualization
    pic_process(model_d_pred, model_g_pred, ens_pred)