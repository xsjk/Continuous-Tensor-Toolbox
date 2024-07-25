from model import *
import torch
import os
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import lightning as L
from skimage.metrics import peak_signal_noise_ratio
from utils import plot_comparison

data_all = ["data/plane"]
c_all = ["2"]

###################
# Here are the hyperparameters.
W_DECAY = 3
LR_REAL = 0.0001
MAX_ITER = 3001
DOWN = [2, 2, 1]
OMEGA = 2
###################

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
L.seed_everything(42)


for data in data_all:
    for c in c_all:
        file_name = f"{data}p{c}.mat"
        mat = scipy.io.loadmat(file_name)
        X_np = mat["Nhsi"]
        X = torch.from_numpy(X_np).to(device, dtype)
        [n_1, n_2, n_3] = X.shape

        mid_channel = int(n_2)
        r_1 = int(n_1 / DOWN[0])
        r_2 = int(n_2 / DOWN[1])
        r_3 = int(n_3 / DOWN[2])
        kernel_size = (r_1, r_2, r_3)

        file_name = data + "gt.mat"
        mat = scipy.io.loadmat(file_name)
        gt_np: np.ndarray = mat["Ohsi"]
        assert np.all(gt_np <= 1) and np.all(gt_np >= 0)

        gt = torch.from_numpy(gt_np).to(device, dtype)

        mask = torch.ones_like(X)
        mask[X == 0] = 0

        U_input = torch.arange(1, n_1 + 1, dtype=dtype, device=device)
        V_input = torch.arange(1, n_2 + 1, dtype=dtype, device=device)
        W_input = torch.arange(1, n_3 + 1, dtype=dtype, device=device)

        model = LRTF(kernel_size, hidden_size=[r_2, r_2], omega_0=OMEGA).to(device, dtype)
        optimizier = optim.Adam(model.parameters(), lr=LR_REAL, weight_decay=W_DECAY)

        for iter in range(MAX_ITER):
            X_Out = model(U_input, V_input, W_input)
            loss = torch.linalg.vector_norm((X_Out - X) * mask, 2)

            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

            if iter % 100 == 0:
                X_Out_np: np.ndarray = X_Out.cpu().detach().numpy()

                ps = peak_signal_noise_ratio(gt_np.astype(X_Out_np.dtype), X_Out_np)
                print("iteration:", iter, "PSNR", ps)
                
                plot_comparison(gt_np, X_Out_np)
                plt.savefig(f"result/{os.path.split(data)[-1]}p{c}_{iter}.png", bbox_inches="tight")
                plt.close()
