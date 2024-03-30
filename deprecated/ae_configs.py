import torch
import torch.nn as nn
import torch.optim as optim

from data import *
from models import CAE

from copy import deepcopy

base = dict()

exp1 = deepcopy(base)
exp1.update(
    lr=1e-4,
    batch_size=256,
    dataset_class=CIFARImages,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        patch_size=12,
    ),
    model_class=CAE,
    model_args=dict(
        bottleneck_sizes=[128, 32, 8, 2],
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.CrossEntropyLoss(),
)

exp2 = deepcopy(base)
exp2.update(
    lr=1e-3,
    batch_size=128,
    dataset_class=NewCaltech256Images,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        patch_size=28,
    ),
    model_class=CAE,
    model_args=dict(
        output_channels=[8, 16, 32, 64],
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 1],
        paddings=[1, 1, 1, 1],
        bottleneck_sizes=[128, 64],
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.CrossEntropyLoss(),
)

exp3 = deepcopy(base)
exp3.update(
    lr=1e-3,
    batch_size=128,
    dataset_class=NewCaltech256Images,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        patch_size=28,
    ),
    model_class=CAE,
    model_args=dict(
        output_channels=[8, 16, 32],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 1],
        paddings=[1, 1, 1],
        bottleneck_sizes=[64],
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)

exp4 = deepcopy(base)
exp4.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Images,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        patch_size=28,
    ),
    model_class=CAE,
    model_args=dict(
        output_channels=[8, 16, 32],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 1],
        paddings=[1, 1, 1],
        bottleneck_sizes=[128, 64, 32],
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)


AE_CONFIGS = dict(
    exp1=exp1,
    exp2=exp2,
    exp3=exp3,
    exp4=exp4,
)