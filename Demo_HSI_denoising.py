from model import *
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from skimage.metrics import peak_signal_noise_ratio
import lightning as L
from utils import plot_comparison

data_all = ["data/om1"]
c_all = ["case2"]

###################
# Here are the hyperparameters.
MAX_ITER = 5001
W_DECAY = 0.1
LR_REAL = 0.0001
PHI = 5 * 10e-6
MU = 1.2
GAMMA = 0.1
DOWN = 4
OMEGA = 2
###################

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
L.seed_everything(42)


for data in data_all:
    for c in c_all:
        file_name = f"{data}{c}.mat"
        mat = scipy.io.loadmat(file_name)
        X_np = mat["Nhsi"]
        X = torch.from_numpy(X_np).to(device, dtype)
        [n_1, n_2, n_3] = X.shape

        mid_channel = n_2
        r_1 = int(n_1 / DOWN)
        r_2 = int(n_2 / DOWN)
        r_3 = int(n_3 / DOWN)
        kernel_size = (r_1, r_2, r_3)

        file_name = data + "gt.mat"
        mat = scipy.io.loadmat(file_name)
        gt_np = mat["Ohsi"]
        gt_np = np.clip(gt_np, 0, 1)
        assert np.all(gt_np <= 1) and np.all(gt_np >= 0)

        gt = torch.from_numpy(gt_np).to(device, dtype)

        mask = torch.ones_like(X)
        mask[X == 0] = 0

        U_input = torch.arange(1, n_1 + 1, dtype=dtype, device=device)
        V_input = torch.arange(1, n_2 + 1, dtype=dtype, device=device)
        W_input = torch.arange(1, n_3 + 1, dtype=dtype, device=device)

        model = LRTF(kernel_size, hidden_size=[r_2, r_2], omega_0=OMEGA).to(device, dtype)
        optimizier = optim.Adam(model.parameters(), lr=LR_REAL, weight_decay=W_DECAY)

        ps_best = 0
        for iter in range(MAX_ITER):
            X_Out = model(U_input, V_input, W_input)
            if iter == 0:
                D = torch.zeros_like(X)
                S = X - X_Out.detach()

            V = soft_thres(S + D / MU, GAMMA / MU)
            S = (2 * X - 2 * X_Out.detach() + MU * V - D) / (2 + MU)

            loss = (
                torch.norm((X - X_Out - S) * mask, 2)
                + PHI * torch.norm(X_Out[1:, :, :] - X_Out[:-1, :, :], 1)
                + PHI * torch.norm(X_Out[:, 1:, :] - X_Out[:, :-1, :], 1)
            )

            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

            D += MU * (S - V)

            if iter % 100 == 0:
                X_Out_np = X_Out.cpu().detach().numpy()

                ps = peak_signal_noise_ratio(gt_np, X_Out_np)
                print("iteration:", iter, "PSNR", ps)

                show = [15, 25, 30]
                plot_comparison(gt_np[..., show], X_Out_np[..., show])
                plt.savefig(
                    f"result/{os.path.split(data)[-1]}p{c}_{iter}.png",
                    bbox_inches="tight",
                )
                plt.close()
