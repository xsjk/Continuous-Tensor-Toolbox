import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import math
import open3d as o3d
import lightning as L
from model import *
from utils import scatter_comparison
import os

L.seed_everything(42)

data_all = ["data/heartp0.05"]

#################
# Here are the hyperparameters.
LR_REAL = 0.000001
THRES = 0.01
DOWN = 5
MAX_ITER = 500
OMAGA = 4
GAMMA_0 = 300
GAMMA_1 = 0.3
GAMMA_2 = 0.3
#################

RAND_NUM = 30
ADD_BORDER = 0.1

SIZE_PC = 6

dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gradient(y, x):
    return torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), create_graph=True)[0]


for data in data_all:
    pcd = o3d.io.read_point_cloud(data + ".pcd")
    X_np = np.array(pcd.points)
    X_Input = torch.from_numpy(X_np).to(device, dtype)

    n, ndim = X_np.shape

    r_1 = n // DOWN
    r_2 = n // DOWN
    r_3 = n // DOWN

    model = LRTF((r_1, r_2, r_3), [n, n, n], omega_0=OMAGA).to(device, dtype)
    optimizier = optim.Adam(model.parameters(), lr=LR_REAL)
    
    for iter in range(MAX_ITER):

        X_Random = torch.vstack([
            torch.empty((RAND_NUM), dtype=dtype, device=device).uniform_(
                torch.min(X_Input[:, i]) - ADD_BORDER,
                torch.max(X_Input[:, i]) + ADD_BORDER,
            )
            for i in range(ndim)
        ]).T.requires_grad_()

        X_Out = model.diag(*X_Input.T)
        X_Out_off = model(*X_Random.T)

        loss_1 = GAMMA_0 * torch.norm(X_Out, p=1)
        loss_2 = GAMMA_1 * torch.norm(gradient(X_Out_off, X_Random).norm(dim=-1) - RAND_NUM**2, p=1)
        loss_3 = GAMMA_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)), p=1)
        
        loss = loss_1 + loss_2 + loss_3

        optimizier.zero_grad()
        loss.backward(retain_graph=True)
        optimizier.step()

        if iter % 200 == 0:

            with torch.no_grad():

                print("iteration:", iter)
                number = 60
                range_ = torch.from_numpy(np.array(range(number))).to(device, dtype)
                X_In = torch.vstack([
                    torch.linspace(torch.min(X_Input[:, i]) - ADD_BORDER, 
                                   torch.max(X_Input[:, i]) + ADD_BORDER, 
                                   number, device=device, dtype=dtype)
                    for i in range(ndim)
                ]).T

                idx = torch.where(torch.abs(model(*X_In.T)) < THRES)
                points = torch.vstack([X_In[idx[i], i] for i in range(ndim)]).T.cpu().numpy()
                scatter_comparison(X_np, points, SIZE_PC)
                plt.savefig(f"result/{os.path.split(data)[-1]}_{iter}.png", bbox_inches="tight")
                plt.close()
