#%%
import pandas as pd
import torch
from Code.DHCorn_hour import dhcorn
from Code.make_g import make_treno
from Code.utils import simulate_growth_curves
import matplotlib.pyplot as plt
import numpy as np

#%% Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = 'cpu'

print(f"当前设备: {device}")

#%% 直接加载并赋值
beta       = torch.load('result/beta.pt').to(device)
env        = torch.load('result/env.pt').to(device)
FT         = torch.load('result/FT.pt').to(device)
G          = torch.load('result/G.pt').to(device)
geno       = torch.load('result/geno.pt').to(device)
Height     = torch.load('result/Height.pt').to(device)
HIR_mat    = torch.load('result/HIR.pt').to(device)
mg         = torch.load('result/mg.pt').to(device)
TL         = torch.load('result/TL.pt').to(device)
train_idx  = torch.load('result/train_idx.pt')
test_idx   = torch.load('result/test_idx.pt')

#%%
N = len(HIR)
treno, Utreno, Ltreno = make_treno()
Utreno = torch.tensor(Utreno, device=device)
Ltreno = torch.tensor(Ltreno, device=device)


g = torch.matmul(G, beta[:, 0, :]).reshape(N, 4, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, 4, 89)
geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno

ph = dhcorn(env, mg, geno, device=device)
