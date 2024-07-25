import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import numpy as np

######################
# Plotting functions #
######################

def plot_comparison(im_true: np.ndarray, im_pred: np.ndarray, title_true: str = "Ground truth", title_pred: str = "Prediction", figsize: tuple[int, int] = (9, 6)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(np.clip(im_true, 0, 1), cmap="gray")
    ax[0].set_title(title_true)
    ax[0].axis("off")
    ax[1].imshow(np.clip(im_pred, 0, 1), cmap="gray")
    ax[1].set_title(title_pred)
    ax[1].axis("off")

def scatter_comparison(points_true: torch.Tensor, points_pred: torch.Tensor, size_pc: int = 6):
    assert points_true.shape[1] == 3
    assert points_pred.shape[1] == 3
    
    fig = plt.figure(figsize=(9, 5))
    ax = plt.subplot(121, projection="3d")
    ax.scatter(*points_true.T, s=size_pc)
    ax.view_init(elev=30, azim=90)

    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(*points_pred.T, s=size_pc)
    ax.view_init(elev=30, azim=90)

