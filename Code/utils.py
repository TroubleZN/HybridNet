import numpy as np
import torch

def simulate_growth_curves(nsamples=1366, ndays=130, device='cpu'):
    days = np.arange(ndays)

    # Height: sigmoid (max 150 cm)
    H = 150 / (1 + np.exp(-0.1 * (days - 45)))

    # Tassel Length: sigmoid (max 25 cm)
    TL = 25 / (1 + np.exp(-0.2 * (days - 70)))

    # HIR: sigmoid (max 30%)
    HIR = 0.3 / (1 + np.exp(-0.15 * (days - 65)))

    # Add noise and tile to all samples
    H_mat = np.tile(H, (nsamples, 1)) + np.random.normal(0, 3.0, size=(nsamples, ndays))
    TL_mat = np.tile(TL, (nsamples, 1)) + np.random.normal(0, 1.0, size=(nsamples, ndays))
    HIR_mat = np.tile(HIR, (nsamples, 1)) + np.random.normal(0, 0.005, size=(nsamples, ndays))

    # Clip to valid range
    H_mat = np.clip(H_mat, 0, None)
    TL_mat = np.clip(TL_mat, 0, None)
    HIR_mat = np.clip(HIR_mat, 0, None)

    # Convert to torch tensors
    H_tensor = torch.tensor(H_mat, dtype=torch.float32, device=device)
    TL_tensor = torch.tensor(TL_mat, dtype=torch.float32, device=device)
    HIR_tensor = torch.tensor(HIR_mat, dtype=torch.float32, device=device)

    return H_tensor, TL_tensor, HIR_tensor