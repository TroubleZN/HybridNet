from model import dhcorn
from sorghum import simu_sorghum

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


# torch.autograd.set_detect_anomaly(True)
# torch.autograd.set_detect_anomaly(False)

# 设置设备为 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# 参数
nsamples = 2000
H = 4320
genostage = 3
genoparam = 76
envivar = 13

# 构造输入数据
envi = (torch.randn(nsamples, H, envivar) * 10 + 50).to(device)  # 环境变量
mg = torch.cat([
    torch.randint(low=160, high=170, size=(nsamples, 1)),          # 收获日 160 ~ 180
    torch.rand(nsamples, 1) * 3 + 3,                               # 密度 3.0 ~ 6.0
    torch.randint(low=0, high=5, size=(nsamples, 1))
], dim=1).to(device)
geno = torch.rand(nsamples, genostage, genoparam).to(device)       # 遗传参数

def generate_ideal_height_curve(nsamples=2000, device='cpu'):
    days = torch.arange(180, device=device).float()  # 0~179天
    max_height = torch.normal(mean=2.0, std=0.2, size=(nsamples, 1)).to(device)  # 每株植物最终高度在 1.8~2.2 米之间

    # 生长曲线函数：sigmoid 形式
    growth_rate = 0.1
    turning_point = 90.0  # 拐点在90天左右

    height_curve = max_height * torch.sigmoid(growth_rate * (days - turning_point))  # [nsamples, 180]
    return height_curve

target_height = generate_ideal_height_curve(nsamples=nsamples, device='cuda')

#%%
# 运行 dhcorn 模型
# ph = dhcorn(envi, mg, geno, device=device)


#%%
import gc
# 可训练参数 geno 初始化
geno = torch.rand(nsamples, 3, genoparam, device=device, requires_grad=True)
# geno = geno.repeat(nsamples, 1, 1)

optimizer = torch.optim.Adam([geno], lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()

    ph = dhcorn(envi, mg, geno, device=device)
    loss = torch.nn.functional.mse_loss(ph["Height"], target_height) / max(target_height)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1:02d}, Height MSE Loss = {loss.item():.4f}")

    # 显式删除变量，释放前一轮图
    del ph, loss
    torch.cuda.empty_cache()
    gc.collect()


