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
import random
import argparse

from kmeans_pytorch import kmeans


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_center(x, eps_shift):
    assert x.dim() == 3

    return x[:, eps_shift:-eps_shift, eps_shift:-eps_shift]


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
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 3),
            GaussianNoise(std),
            Normalize(),
        )

        self.v_projector = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            Normalize(),
        )

        self.A_v_L = nn.Sequential(
            nn.Linear(3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        self.A_v_R = nn.Sequential(
            nn.Linear(3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def get_v(self, i1, i2):
        images = torch.cat([i1, i2], dim=1)

        v = self.v(images)
        v_projected = self.v_projector(v)

        return v, v_projected

    def get_i3(self, v, i2):
        v_L = self.A_v_L(v).reshape(-1,  8, 8)
        v_R = self.A_v_R(v).reshape(-1,  8, 8)

        return torch.sigmoid(torch.bmm(torch.bmm(v_L, i2.reshape(-1, 8, 8)), v_R))

    def forward(self, i1, i2):
        v, v_projected = self.get_v(i1, i2)

        pred_i3 = self.get_i3(v, i2)

        return v, v_projected, pred_i3


def attract_loss(pred, gt, eps_shift):
    mse = nn.MSELoss()

    return mse(get_center(pred, eps_shift), get_center(gt, eps_shift))


def repel_loss(x, y):
    # mse = nn.MSELoss()

    # return 5 * torch.exp(-mse(x, y) / (2 * 0.6**2))

    return nn.functional.cosine_similarity(x, y).mean()


def run(std, eps_shift):
    model = Rand8x8_LR1(std=std).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_training = int(3e3 * ((2 * eps_shift + 1)**2))
    num_testing = int(num_training * 0.3)

    train_dataset = ThreeRandomOneShift(train=True, patch_size=(8, 8), eps_shift=eps_shift, num_training=num_training, num_testing=num_testing)
    test_dataset = ThreeRandomOneShift(train=False, patch_size=(8, 8), eps_shift=eps_shift, num_training=num_training, num_testing=num_testing)

    train_loader = DataLoader(train_dataset, batch_size=768, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=768, shuffle=False, pin_memory=True, num_workers=6)

    train_losses = {'image loss' : [], 'v loss' : []}
    test_losses = {'image loss' : [], 'v loss' : []}

    print("Model size: {}".format(count_parameters(model)))

    for epoch in tqdm.tqdm(range(800)):
        """
        train
        """
        model.train()

        losses = {'image loss' : [], 'v loss' : []}
        for i1, i2, i3, _  in train_loader:
            i1 = i1.cuda()
            i2 = i2.cuda()
            i3 = i3.cuda()

            optimizer.zero_grad()

            _, v_projected, pred_i3 = model(i1, i2)

            image_loss = attract_loss(pred_i3, i3, eps_shift)

            repel_loss_inds = torch.combinations(torch.arange(len(v_projected)), r=2)
            repel_loss_inds1, repel_loss_inds2 = repel_loss_inds[:, 0], repel_loss_inds[:, 1]
            v_loss = repel_loss(v_projected[repel_loss_inds1], v_projected[repel_loss_inds2])

            loss = image_loss + v_loss

            loss.backward()
            optimizer.step()
            
            losses['image loss'].append(image_loss.detach().cpu().item())
            losses['v loss'].append(v_loss.detach().cpu().item())

        train_losses['image loss'].append(np.mean(losses['image loss']))
        train_losses['v loss'].append(np.mean(losses['v loss']))

        if (epoch % 5 == 0):
            """
            eval
            """
            model.eval()

            losses = {'image loss' : [], 'v loss' : []}
            with torch.no_grad():
                for i1, i2, i3, _  in test_loader:
                    i1 = i1.cuda()
                    i2 = i2.cuda()
                    i3 = i3.cuda()

                    _, v_projected, pred_i3 = model(i1, i2)

                    image_loss = attract_loss(pred_i3, i3, eps_shift)

                    repel_loss_inds = torch.combinations(torch.arange(len(v_projected)), r=2)
                    repel_loss_inds1, repel_loss_inds2 = repel_loss_inds[:, 0], repel_loss_inds[:, 1]
                    v_loss = repel_loss(v_projected[repel_loss_inds1], v_projected[repel_loss_inds2])

                    losses['image loss'].append(image_loss.detach().cpu().item())
                    losses['v loss'].append(v_loss.detach().cpu().item())

            test_losses['image loss'].append(np.mean(losses['image loss']))
            test_losses['v loss'].append(np.mean(losses['v loss']))

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), width_ratios=[2, 2], height_ratios=[3, 2, 2], constrained_layout=True)
    axes = axes.flatten()

    axes[0].plot(train_losses['image loss'], color='darkblue', label='train image loss')
    axes[0].plot(train_losses['v loss'], color='darkorchid', label='train v loss')
    axes[0].plot(test_losses['image loss'], color='cornflowerblue', label='test image loss')
    axes[0].plot(test_losses['v loss'], color='plum', label='test v loss')
    axes[0].legend()
    
    model.eval()

    coords = torch.cartesian_prod(
        torch.arange(-eps_shift, eps_shift+1),
        torch.arange(-eps_shift, eps_shift+1)
    )

    #colorlist = ['maroon', 'darkorange', 'darkkhaki', 'limegreen', 'aquamarine', 'royalblue', 'blueviolet', 'darkmagenta', 'slategray']
    colorlist = list(mp.colors.cnames.keys())
    random.seed(0)
    random.shuffle(colorlist)
    colorlist = colorlist[:len(coords)]

    colorlabels = {tuple(coord.numpy()) : colorlist[i] for i, coord in enumerate(coords)}

    points, colors = [], []

    for _ in range(1000):
        with torch.no_grad():
            index = np.random.randint(0, len(train_dataset), size=(1,))[0]
            i1, i2, i3, gt_v = train_dataset[index]

            i1 = i1.cuda().unsqueeze(0)
            i2 = i2.cuda().unsqueeze(0)
            i3 = i3.cuda().unsqueeze(0)

            pred_v, _, pred_i3 = model(i1, i2)

            points.append(pred_v.squeeze().cpu().numpy())
            colors.append(colorlabels[tuple(gt_v.numpy())])

    # string = ''
    # for key, val in colorlabels.items():
    #     string += '{}: {}\n'.format(key, val)
    # with open("./baseline_kmeans_eps={}_std={}.txt".format(str(eps_shift), str(std)[-1]), "w") as f:
    #     print(string, file=f)
            
    im = axes[2].imshow(get_center(i3, eps_shift).squeeze().detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title("true i3")
    im = axes[3].imshow(get_center(pred_i3, eps_shift).squeeze().detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
    axes[3].set_title("pred i3 given pred v")

    points = np.array(points)

    x = points[:, 0].T
    y = points[:, 1].T
    z = points[:, 2].T

    axes[1].scatter(*np.vstack((np.arctan2(y, x), z)), color=colors, s=8, alpha=1.0)

    """
    k-means
    """
    points = torch.from_numpy(points).cuda()
 
    cluster_ids_x, cluster_centers = kmeans(
        X=points, num_clusters=(2 * eps_shift + 1)**2, distance='cosine', device='cuda', tol=1e-3,
    )
    cluster_centers = cluster_centers.cuda()

    with torch.no_grad():
        losses = []
        for i1, i2, i3, _  in train_loader:
            i1 = i1.cuda()
            i2 = i2.cuda()
            i3 = i3.cuda()

            pred_v, _ = model.get_v(i1, i2)

            v_centroid = cluster_centers[
                (
                    torch.linalg.norm(
                        cluster_centers.repeat(pred_v.shape[0], 1, 1) - pred_v.unsqueeze(1), ord=2, dim=2
                    )**2
                ).argmin(dim=1)
            ]

            pred_i3 = model.get_i3(v_centroid, i2)

            losses.append(attract_loss(pred_i3, i3, eps_shift).detach().cpu().item())

    im = axes[4].imshow(get_center(i3[0].unsqueeze(0), eps_shift).squeeze().detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
    axes[4].set_title("true i3")
    im = axes[5].imshow(get_center(pred_i3[0].unsqueeze(0), eps_shift).squeeze().detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
    axes[5].set_title("pred i3 given v centroids")
    
    cluster_centers = cluster_centers.detach().cpu().numpy()
    x = cluster_centers[:, 0].T
    y = cluster_centers[:, 1].T
    z = cluster_centers[:, 2].T

    axes[1].scatter(*np.vstack((np.arctan2(y, x), z)), color='black', s=50, alpha=1.0)
    axes[1].set_title("(using centroids as 'v') loss on training set: {}".format(np.mean(losses)))

    fig.colorbar(im, ax=axes[2:])
    fig.savefig("./batch_attract_mse_repel_cosim_kmeans_eps={}_std={}".format(str(eps_shift), str(std)[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eps_shift', required=True, type=int)
    args = parser.parse_args()

    run(std=0.0, eps_shift=args.eps_shift)
    run(std=0.1, eps_shift=args.eps_shift)
    run(std=0.2, eps_shift=args.eps_shift)
    run(std=0.3, eps_shift=args.eps_shift)
    run(std=0.4, eps_shift=args.eps_shift)
    run(std=0.5, eps_shift=args.eps_shift)
    run(std=0.6, eps_shift=args.eps_shift)
    run(std=0.7, eps_shift=args.eps_shift)
    run(std=0.8, eps_shift=args.eps_shift)
    run(std=0.9, eps_shift=args.eps_shift)