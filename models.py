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
    def __init__(self, seed, input_shape, layer_sizes, v_dim):
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
    

class VariationalAutoencoder(nn.Module):
    def __init__(self, seed, input_shape, layer_sizes, v_dim):
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
                self.mu = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                self.logvar = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
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
        x = self.encoder(images)

        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)

        z = mu + torch.randn_like(std) * std 

        return z, mu, logvar

    def decoder_forward(self, latent):
        return self.decoder(latent)

    def forward(self, images):
        z, mu, logvar = self.encoder_forward(images)

        pred_images = self.decoder_forward(z)

        return pred_images, mu, logvar


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, self.kernel_size, padding=self.padding)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.hidden_dim, *shape, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, *shape, device=self.conv.weight.device))
    

class MCNet_32x12(nn.Module):
    def __init__(self, seed, v_dim):
        super().__init__()

        # shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        

        # motion encoder
        self.convlstm = ConvLSTMCell(32, 32, 3)
        self.motion_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, v_dim),
        )


        # content encoder
        self.content_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, v_dim),
        )


        # combination layers
        self.combination_fc = nn.Sequential(
            nn.Linear(v_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),            
        )
        self.combination_conv = nn.Sequential(
            nn.Upsample(size=(8, 4), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(16, 8), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(32, 12), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


        torch.manual_seed(seed)
        self.apply(he_init)

    
    def motion_encoder(self, x, hidden_state):
        x = self.encoder(x)

        h_next, c_next = self.convlstm(x, hidden_state)

        x = self.motion_fc(h_next.view(h_next.size(0), -1))

        return x, (h_next, c_next)


    def content_encoder(self, x):
        x = self.encoder(x)

        x = self.content_fc(x.view(x.size(0), -1))

        return x

    
    def combination_layers(self, x):
        bs, L, _ = x.shape

        x = x.flatten(start_dim=0, end_dim=1)

        x = self.combination_fc(x).view(-1, 32, 4, 2)
        x = self.combination_conv(x)

        x = x.view(bs, L, 1, 32, 12)

        return x

    
    def forward(self, x):
        bs, L, _, _, _ = x.shape

        motion_inputs = x[:, 1:, :] - x[:, :-1, :]

        hidden_state = self.convlstm.init_hidden(bs, (4, 2))

        reconstruction_in = []

        for t in range(L - 1):
            motion_feature, hidden_state = self.motion_encoder(motion_inputs[:, t, :], hidden_state)

            reconstruction_in.append(motion_feature)

        reconstruction_in.append(self.content_encoder(x[:, -1, :]))

        reconstruction_in = torch.stack(reconstruction_in, dim=1)
        reconstruction_out = self.combination_layers(reconstruction_in)
        
        return reconstruction_in[:, :-1, :], reconstruction_out


class MCNet_16x16(nn.Module):
    def __init__(self, seed, v_dim):
        super().__init__()

        # shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        

        # motion encoder
        self.convlstm = ConvLSTMCell(32, 32, 3)
        self.motion_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, v_dim),
        )


        # content encoder
        self.content_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, v_dim),
        )


        # combination layers
        self.combination_fc = nn.Sequential(
            nn.Linear(v_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),            
        )
        self.combination_conv = nn.Sequential(
            nn.Upsample(size=(4, 4), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


        torch.manual_seed(seed)
        self.apply(he_init)

    
    def motion_encoder(self, x, hidden_state):
        x = self.encoder(x)

        h_next, c_next = self.convlstm(x, hidden_state)

        x = self.motion_fc(h_next.view(h_next.size(0), -1))

        return x, (h_next, c_next)


    def content_encoder(self, x):
        x = self.encoder(x)

        x = self.content_fc(x.view(x.size(0), -1))

        return x

    
    def combination_layers(self, x):
        bs, L, _ = x.shape

        x = x.flatten(start_dim=0, end_dim=1)

        x = self.combination_fc(x).view(-1, 32, 2, 2)
        x = self.combination_conv(x)

        x = x.view(bs, L, 1, 16, 16)

        return x

    
    def forward(self, x):
        bs, L, _, _, _ = x.shape

        motion_inputs = x[:, 1:, :] - x[:, :-1, :]

        hidden_state = self.convlstm.init_hidden(bs, (2, 2))

        reconstruction_in = []

        for t in range(L - 1):
            motion_feature, hidden_state = self.motion_encoder(motion_inputs[:, t, :], hidden_state)

            reconstruction_in.append(motion_feature)

        reconstruction_in.append(self.content_encoder(x[:, -1, :]))

        reconstruction_in = torch.stack(reconstruction_in, dim=1)
        reconstruction_out = self.combination_layers(reconstruction_in)
        
        return reconstruction_in[:, :-1, :], reconstruction_out


class MCNet_1x100(nn.Module):
    def __init__(self, seed, v_dim):
        super().__init__()

        # shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        

        # motion encoder
        self.convlstm = ConvLSTMCell(32, 32, 3)
        self.motion_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, v_dim),
        )


        # content encoder
        self.content_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, v_dim),
        )


        # combination layers
        self.combination_fc = nn.Sequential(
            nn.Linear(v_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),            
        )
        self.combination_conv = nn.Sequential(
            nn.Upsample(size=(4, 4), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(10, 10), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


        torch.manual_seed(seed)
        self.apply(he_init)

    
    def motion_encoder(self, x, hidden_state):
        x = self.encoder(x)

        h_next, c_next = self.convlstm(x, hidden_state)

        x = self.motion_fc(h_next.view(h_next.size(0), -1))

        return x, (h_next, c_next)


    def content_encoder(self, x):
        x = self.encoder(x)

        x = self.content_fc(x.view(x.size(0), -1))

        return x

    
    def combination_layers(self, x):
        bs, L, _ = x.shape

        x = x.flatten(start_dim=0, end_dim=1)

        x = self.combination_fc(x).view(-1, 32, 2, 2)
        x = self.combination_conv(x)

        x = x.view(bs, L, 1, 10, 10)

        return x

    
    def forward(self, x):
        bs, L, C, _, _ = x.shape

        x = x.view(bs, L, C, 10, 10)

        motion_inputs = x[:, 1:, :] - x[:, :-1, :]

        hidden_state = self.convlstm.init_hidden(bs, (2, 2))

        reconstruction_in = []

        for t in range(L - 1):
            motion_feature, hidden_state = self.motion_encoder(motion_inputs[:, t, :], hidden_state)

            reconstruction_in.append(motion_feature)

        reconstruction_in.append(self.content_encoder(x[:, -1, :]))

        reconstruction_in = torch.stack(reconstruction_in, dim=1)
        reconstruction_out = self.combination_layers(reconstruction_in).view(bs, L, C, 1, 100)
        
        return reconstruction_in[:, :-1, :], reconstruction_out