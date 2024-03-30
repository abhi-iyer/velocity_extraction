from imports import *
from utils import *


class Gauss_FC_FC(nn.Module):
    def __init__(self, seed, v_dim, num_output_labels):
        super().__init__()

        self.v = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, v_dim),
            nn.Tanh(),
        )

        self.reduce_i2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.expand_v = nn.Sequential(
            nn.Linear(v_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Unflatten(dim=1, unflattened_size=(1, 16, 16)),
            nn.Conv2d(1, num_output_labels, kernel_size=1, stride=1),
        )

        torch.manual_seed(seed)
        self.apply(he_init)

    def get_v(self, i1, i2):
        images = torch.cat([i1, i2], dim=1)

        return self.v(images)

    def get_i3(self, v, i2):
        reduced_i2 = self.reduce_i2(i2)
        expanded_v = self.expand_v(v)
        
        return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

    def forward(self, i1, i2):
        v = self.get_v(i1, i2)

        pred_i3 = self.get_i3(v, i2)

        return v, pred_i3


class Blob_FC_FC(nn.Module):
    def __init__(self, seed, v_dim, num_output_labels):
        super().__init__()

        self.v = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, v_dim),
            nn.Tanh(),
        )

        self.reduce_i2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.expand_v = nn.Sequential(
            nn.Linear(v_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Unflatten(dim=1, unflattened_size=(1, 16, 16)),
            nn.Conv2d(1, num_output_labels, kernel_size=1, stride=1),
        )

        torch.manual_seed(seed)
        self.apply(he_init)

    def get_v(self, i1, i2):
        images = torch.cat([i1, i2], dim=1)

        return self.v(images)

    def get_i3(self, v, i2):
        reduced_i2 = self.reduce_i2(i2)
        expanded_v = self.expand_v(v)
        
        return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

    def forward(self, i1, i2):
        v = self.get_v(i1, i2)

        pred_i3 = self.get_i3(v, i2)

        return v, pred_i3


class Bird_FC_FC(nn.Module):
    def __init__(self, seed, v_dim, num_output_labels):
        super().__init__()

        self.v = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(2 * 32*12, 32*12),
            nn.BatchNorm1d(32*12),
            nn.ReLU(),
            nn.Linear(32*12, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, v_dim),
            nn.Tanh(),
        )

        self.reduce_i2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(32*12, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.expand_v = nn.Sequential(
            nn.Linear(v_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32*12),
            nn.Unflatten(dim=1, unflattened_size=(1, 32, 12)),
            nn.Conv2d(1, num_output_labels, kernel_size=1, stride=1),
        )

        torch.manual_seed(seed)
        self.apply(he_init)

    def get_v(self, i1, i2):
        images = torch.cat([i1, i2], dim=1)

        return self.v(images)

    def get_i3(self, v, i2):
        reduced_i2 = self.reduce_i2(i2)
        expanded_v = self.expand_v(v)
        
        return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

    def forward(self, i1, i2):
        v = self.get_v(i1, i2)

        pred_i3 = self.get_i3(v, i2)

        return v, pred_i3


class Frequency_FC_FC(nn.Module):
    def __init__(self, seed, v_dim, num_output_labels):
        super().__init__()

        block = lambda in_dim, out_dim : nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.v = nn.Sequential(
            nn.Flatten(start_dim=1),
            block(2 * 100, 2 * 100),
            block(2 * 100, 256),
            block(256, 384),
            block(384, 384),
            block(384, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
            block(32, 16),
            block(16, 8),
            block(8, 4),
            nn.Linear(4, v_dim),
            nn.Tanh(),
        )

        self.reduce_i2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            block(100, 84),
            block(84, 64),
            block(64, 32),
        )

        self.expand_v = nn.Sequential(
            block(v_dim, 4),
            block(4, 8),
            block(8, 16),
            block(16, 32),
        )

        self.combine = nn.Sequential(
            block(64, 64),
            block(64, 128),
            block(128, 128),
            nn.Linear(128, 100),
            nn.Unflatten(dim=1, unflattened_size=(1, 10, 10)), # reshape to 1 x 10 x 10 instead of 1 x 100 
            nn.Conv2d(1, num_output_labels, kernel_size=1, stride=1), # because 1d conv becomes more efficient
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(1, 100)),
        )

        torch.manual_seed(seed)
        self.apply(he_init)

    def get_v(self, i1, i2):
        images = torch.cat([i1, i2], dim=1)

        return self.v(images)

    def get_i3(self, v, i2):
        reduced_i2 = self.reduce_i2(i2)
        expanded_v = self.expand_v(v)
        
        return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

    def forward(self, i1, i2):
        v = self.get_v(i1, i2)

        pred_i3 = self.get_i3(v, i2)

        return v, pred_i3



'''
baselines
'''

class Autoencoder(nn.Module):
    def __init__(self, seed, input_shape, layer_sizes):
        super().__init__()

        block = lambda in_dim, out_dim : nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
        )

        for i in range(len(layer_sizes) - 1):
            if (i + 1) == (len(layer_sizes) - 1):
                self.encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            else:
                self.encoder.append(block(layer_sizes[i], layer_sizes[i + 1]))

        
        layer_sizes = layer_sizes[::-1]

        self.decoder = nn.Sequential()

        for i in range(len(layer_sizes) - 1):
            if (i + 1) == (len(layer_sizes) - 1):
                self.decoder.append(nn.Sequential(
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                    nn.Unflatten(dim=1, unflattened_size=input_shape),
                ))
            else:
                self.decoder.append(block(layer_sizes[i], layer_sizes[i + 1]))


        torch.manual_seed(seed)
        self.apply(he_init)

    def encoder_forward(self, images):
        return self.encoder(images)

    def decoder_forward(self, latent):
        return self.decoder(latent)

    def forward(self, images):
        return self.decoder_forward(self.encoder_forward(images))
