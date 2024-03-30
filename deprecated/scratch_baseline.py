import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np
import tqdm

class ThreeRandomOneShift(Dataset):
    def __init__(self, train, patch_size, eps_shift, num_training=10e3, num_testing=5e3):
        super().__init__()

        self.train = train

        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))

        self.patch_size = patch_size
        self.eps_shift = eps_shift
        self.image_size = (self.patch_size[0] + 4*self.eps_shift, self.patch_size[1] + 4*self.eps_shift)

        self.shifts = torch.cartesian_prod(torch.arange(-self.eps_shift, self.eps_shift + 1), torch.arange(-self.eps_shift, self.eps_shift + 1))
                        
    def __getitem__(self, index):
        torch.manual_seed(self.seeds[index])

        shift = self.shifts[torch.randint(len(self.shifts), size=(1,)).item()]

        return self.shift_by(index, shift)

    def shift_by(self, index, shift):        
        shift_x, shift_y = shift[0].long(), shift[1].long()

        torch.manual_seed(self.seeds[index])

        img = torch.rand(*self.image_size)

        center_x_range = torch.arange(2*self.eps_shift, 2*self.eps_shift + self.patch_size[1]).long()
        center_y_range = torch.arange(2*self.eps_shift, 2*self.eps_shift + self.patch_size[0]).long()

        i1 = img[torch.meshgrid(center_y_range, center_x_range, indexing="ij")]
        i2 = img[torch.meshgrid(center_y_range - shift_y, center_x_range - shift_x, indexing="ij")]
        i3 = img[torch.meshgrid(center_y_range - 2*shift_y, center_x_range - 2*shift_x, indexing="ij")]
        
        return i1, i2, i3, shift

    def __len__(self):
        return len(self.seeds)


class Normalize(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return nn.functional.normalize(x, p=2.0, dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, std, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.normal(torch.Tensor([0, 0, 0]).to(x.device), torch.Tensor([self.std, self.std, self.std]).to(x.device))
        else:
            return x

class Rand8x8_LR1(nn.Module):
    def __init__(self, std):
        super().__init__()

        self.v = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 3),
            GaussianNoise(std),
            Normalize(),
        )

        self.A_v_L = nn.Sequential(
            nn.Linear(3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        self.A_v_R = nn.Sequential(
            nn.Linear(3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, i1, i2):
        images = torch.cat([i1, i2], dim=1)

        v = self.v(images)

        v_L = self.A_v_L(v).reshape(-1,  8, 8)
        v_R = self.A_v_R(v).reshape(-1,  8, 8)

        return torch.sigmoid(torch.bmm(torch.bmm(v_L, i2.reshape(-1, 8, 8)), v_R))


def loss_fn(pred, gt):
    mse = nn.MSELoss()

    # return 5 - 5 * torch.exp(-mse(pred.squeeze(), gt.squeeze()) / (2 * 0.5**2))

    return mse(pred, gt)


def run(std):
    model = Rand8x8_LR1(std=std).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = ThreeRandomOneShift(train=True, patch_size=(8, 8), eps_shift=1, num_training=30e3, num_testing=10e3)
    test_dataset = ThreeRandomOneShift(train=False, patch_size=(8, 8), eps_shift=1, num_training=30e3, num_testing=10e3)

    train_loader = DataLoader(train_dataset, batch_size=768, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=768, shuffle=False, pin_memory=True, num_workers=8)
    
    train_losses = []
    test_losses = []

    for epoch in tqdm.tqdm(range(800)):
        """
        train
        """
        model.train()

        losses = []
        for i1, i2, i3, _  in train_loader:
            i1 = i1.cuda()
            i2 = i2.cuda()
            i3 = i3.cuda()

            optimizer.zero_grad()

            pred_i3 = model(i1, i2)

            loss = loss_fn(pred_i3, i3)

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
        train_losses.append(np.mean(losses))


        if (epoch % 5 == 0):
            """
            eval
            """
            model.eval()

            losses = []
            with torch.no_grad():
                for i1, i2, i3, _  in test_loader:
                    i1 = i1.cuda()
                    i2 = i2.cuda()
                    i3 = i3.cuda()

                    pred_i3 = model(i1, i2)

                    loss = loss_fn(pred_i3, i3)

                    losses.append(loss.detach().cpu().item())
            test_losses.append(np.mean(losses))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes = axes.flatten()

    axes[0].plot(train_losses, color='green', label='train loss')
    axes[0].plot(test_losses, color='red', label='test loss')
    axes[0].legend()

    model.eval()

    eps = 1
    coords = torch.cartesian_prod(
        torch.arange(-eps, eps+1),
        torch.arange(-eps, eps+1)
    )

    colorlist = ['maroon', 'darkorange', 'darkkhaki', 'limegreen', 'aquamarine', 'royalblue', 'blueviolet', 'darkmagenta', 'slategray']

    colorlabels = {tuple(coord.numpy()) : colorlist[i] for i, coord in enumerate(coords)}

    points, colors = [], []

    for _ in range(1000):
        with torch.no_grad():
            index = np.random.randint(0, len(train_dataset), size=(1,))[0]
            i1, i2, i3, gt_v = train_dataset[index]

            i1 = i1.cuda().unsqueeze(0)
            i2 = i2.cuda().unsqueeze(0)
            i3 = i3.cuda().unsqueeze(0)
        
            images = torch.cat([i1, i2], dim=1)

            pred_v = model.v(images).squeeze().detach().cpu().numpy()

            points.append(pred_v)
            colors.append(colorlabels[tuple(gt_v.numpy())])

    points = np.array(points).T
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]

    points = np.vstack((np.arctan2(y, x), z))

    axes[1].scatter(*points, color=colors, s=8, alpha=1.0)

    fig.savefig("./baseline_visualization_std={}".format(str(std)[-1]))


if __name__ == "__main__":
    run(std=0.0)
    run(std=0.1)
    run(std=0.2)
    run(std=0.3)
    run(std=0.4)
    run(std=0.5)
    run(std=0.6)
    run(std=0.7)
    run(std=0.8)
    run(std=0.9)