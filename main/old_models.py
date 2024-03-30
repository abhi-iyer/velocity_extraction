'''
21 x 21 images
'''

# class Conv_Conv(nn.Module):
#     def __init__(self,):
#         super().__init__()

#         self.i1_head = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1,1)),
#             nn.Flatten(start_dim=1),
#         )

#         self.i2_head = deepcopy(self.i1_head) 
        
#         self.v = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Linear(16, 2),
#             nn.Tanh(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
        

#         ## A is a UNet architecture 
        
#         self.block = lambda in_ch, middle_ch, out_ch, k, s : nn.Sequential(
#             torch.nn.Conv2d(in_ch, middle_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, middle_ch, kernel_size=k, stride=s, padding=1, groups=middle_ch),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, out_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(out_ch),
#             torch.nn.ReLU(),
#         )
#         self.upsample = lambda size : nn.Upsample(size=size, mode='bilinear', align_corners=True)

#         self.encoder1 = self.block(in_ch=1, middle_ch=12, out_ch=9, k=3, s=2)
#         self.encoder2 = self.block(in_ch=9, middle_ch=36, out_ch=27, k=3, s=2)
#         self.encoder3 = self.block(in_ch=27, middle_ch=108, out_ch=81, k=3, s=2)
#         self.encoder4 = self.block(in_ch=81, middle_ch=324, out_ch=243, k=3, s=2)
#         self.pool = nn.AdaptiveAvgPool2d((1,1))
        
#         self.A_fc = nn.Sequential(
#             nn.Linear(243 + 64, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#         )

#         self.decoder4 = self.block(in_ch=64 + 243, middle_ch=256, out_ch=128, k=3, s=1)
#         self.decoder3 = self.block(in_ch=128 + 81, middle_ch=128, out_ch=64, k=3, s=1)
#         self.decoder2 = self.block(in_ch=64 + 27, middle_ch=64, out_ch=32, k=3, s=1)
#         self.decoder1 = self.block(in_ch=32 + 9, middle_ch=16, out_ch=1, k=3, s=1)
        
#         self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

#     def get_v(self, i1, i2):
#         v = self.v(torch.cat((self.i1_head(i1), self.i2_head(i2)), dim=1))

#         return v

#     def get_i3(self, v, i2):
#         spatial_shapes = [tuple(i2.shape[2:])]

#         enc1 = self.encoder1(i2)
#         spatial_shapes.append(tuple(enc1.shape[2:]))

#         enc2 = self.encoder2(enc1)
#         spatial_shapes.append(tuple(enc2.shape[2:]))

#         enc3 = self.encoder3(enc2)
#         spatial_shapes.append(tuple(enc3.shape[2:]))

#         enc4 = self.encoder4(enc3)
#         spatial_shapes.append(tuple(enc4.shape[2:]))

#         bottleneck = self.A_fc(
#             torch.cat([
#                 self.pool(enc4).flatten(start_dim=1), 
#                 self.expand_v(v)
#             ], dim=1)
#         ).unflatten(dim=1, sizes=(64, 2, 2))

#         dec4 = self.decoder4(torch.cat([self.upsample(size=spatial_shapes[-1])(bottleneck), enc4], dim=1))
#         dec3 = self.decoder3(torch.cat([self.upsample(size=spatial_shapes[-2])(dec4), enc3], dim=1))
#         dec2 = self.decoder2(torch.cat([self.upsample(size=spatial_shapes[-3])(dec3), enc2], dim=1))
#         dec1 = self.decoder1(torch.cat([self.upsample(size=spatial_shapes[-4])(dec2), enc1], dim=1))

#         return self.conv(self.upsample(size=spatial_shapes[-5])(dec1)).squeeze()

#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3


# class FC_FC(nn.Module):
#     def __init__(self,):
#         super().__init__()

#         self.v = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(2 * (21)**2, 2 * (21)**2),
#             nn.BatchNorm1d(2 * (21)**2),
#             nn.ReLU(),
#             nn.Linear(2 * (21)**2, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 2),
#             nn.Tanh(),
#         )

#         self.reduce_i2 = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(21**2, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )

#         self.combine = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 324),
#             nn.BatchNorm1d(324),
#             nn.ReLU(),
#             nn.Linear(324, 21**2),
#             nn.Unflatten(dim=1, unflattened_size=(21, 21)),
#         )

#     def get_v(self, i1, i2):
#         images = torch.cat([i1, i2], dim=1)

#         return self.v(images)

#     def get_i3(self, v, i2):
#         reduced_i2 = self.reduce_i2(i2)
#         expanded_v = self.expand_v(v)
        
#         return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3


# class Conv_FC(nn.Module):
#     def __init__(self,):
#         super().__init__()

#         self.i1_head = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1,1)),
#             nn.Flatten(start_dim=1),
#         )

#         self.i2_head = deepcopy(self.i1_head) 
        
#         self.v = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Linear(16, 2),
#             nn.Tanh(),
#         )

#         self.reduce_i2 = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(21**2, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
        
#         self.combine = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 324),
#             nn.BatchNorm1d(324),
#             nn.ReLU(),
#             nn.Linear(324, 21**2),
#             nn.Unflatten(dim=1, unflattened_size=(21, 21)),
#         )

#     def get_v(self, i1, i2):
#         v = self.v(torch.cat((self.i1_head(i1), self.i2_head(i2)), dim=1))

#         return v

#     def get_i3(self, v, i2):
#         reduced_i2 = self.reduce_i2(i2)
#         expanded_v = self.expand_v(v)

#         return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3


# class FC_Conv(nn.Module):
#     def __init__(self,):
#         super().__init__()

#         self.v = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(2 * (21)**2, 2 * (21)**2),
#             nn.BatchNorm1d(2 * (21)**2),
#             nn.ReLU(),
#             nn.Linear(2 * (21)**2, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 2),
#             nn.Tanh(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
        

#         ## A is a UNet architecture 
        
#         self.block = lambda in_ch, middle_ch, out_ch, k, s : nn.Sequential(
#             torch.nn.Conv2d(in_ch, middle_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, middle_ch, kernel_size=k, stride=s, padding=1, groups=middle_ch),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, out_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(out_ch),
#             torch.nn.ReLU(),
#         )
#         self.upsample = lambda size : nn.Upsample(size=size, mode='bilinear', align_corners=True)

#         self.encoder1 = self.block(in_ch=1, middle_ch=12, out_ch=9, k=3, s=2)
#         self.encoder2 = self.block(in_ch=9, middle_ch=36, out_ch=27, k=3, s=2)
#         self.encoder3 = self.block(in_ch=27, middle_ch=108, out_ch=81, k=3, s=2)
#         self.encoder4 = self.block(in_ch=81, middle_ch=324, out_ch=243, k=3, s=2)
#         self.pool = nn.AdaptiveAvgPool2d((1,1))
        
#         self.A_fc = nn.Sequential(
#             nn.Linear(243 + 64, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#         )

#         self.decoder4 = self.block(in_ch=64 + 243, middle_ch=256, out_ch=128, k=3, s=1)
#         self.decoder3 = self.block(in_ch=128 + 81, middle_ch=128, out_ch=64, k=3, s=1)
#         self.decoder2 = self.block(in_ch=64 + 27, middle_ch=64, out_ch=32, k=3, s=1)
#         self.decoder1 = self.block(in_ch=32 + 9, middle_ch=16, out_ch=1, k=3, s=1)
        
#         self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

#     def get_v(self, i1, i2):
#         images = torch.cat([i1, i2], dim=1)

#         return self.v(images)

#     def get_i3(self, v, i2):
#         spatial_shapes = [tuple(i2.shape[2:])]

#         enc1 = self.encoder1(i2)
#         spatial_shapes.append(tuple(enc1.shape[2:]))

#         enc2 = self.encoder2(enc1)
#         spatial_shapes.append(tuple(enc2.shape[2:]))

#         enc3 = self.encoder3(enc2)
#         spatial_shapes.append(tuple(enc3.shape[2:]))

#         enc4 = self.encoder4(enc3)
#         spatial_shapes.append(tuple(enc4.shape[2:]))

#         bottleneck = self.A_fc(
#             torch.cat([
#                 self.pool(enc4).flatten(start_dim=1), 
#                 self.expand_v(v)
#             ], dim=1)
#         ).unflatten(dim=1, sizes=(64, 2, 2))

#         dec4 = self.decoder4(torch.cat([self.upsample(size=spatial_shapes[-1])(bottleneck), enc4], dim=1))
#         dec3 = self.decoder3(torch.cat([self.upsample(size=spatial_shapes[-2])(dec4), enc3], dim=1))
#         dec2 = self.decoder2(torch.cat([self.upsample(size=spatial_shapes[-3])(dec3), enc2], dim=1))
#         dec1 = self.decoder1(torch.cat([self.upsample(size=spatial_shapes[-4])(dec2), enc1], dim=1))

#         return self.conv(self.upsample(size=spatial_shapes[-5])(dec1)).squeeze()

#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3




'''
16 x 16 images
'''

# class Conv_Conv(nn.Module):
#     def __init__(self,):
#         super().__init__()

#         self.i1_head = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1,1)),
#             nn.Flatten(start_dim=1),
#         )

#         self.i2_head = deepcopy(self.i1_head) 
        
#         self.v = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Linear(16, 2),
#             nn.Tanh(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
        

#         ## A is a UNet architecture 
        
#         self.block = lambda in_ch, middle_ch, out_ch, k, s : nn.Sequential(
#             torch.nn.Conv2d(in_ch, middle_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, middle_ch, kernel_size=k, stride=s, padding=1, groups=middle_ch),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, out_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(out_ch),
#             torch.nn.ReLU(),
#         )
#         self.upsample = lambda size : nn.Upsample(size=size, mode='bilinear', align_corners=True)

#         self.encoder1 = self.block(in_ch=1, middle_ch=12, out_ch=9, k=3, s=2)
#         self.encoder2 = self.block(in_ch=9, middle_ch=36, out_ch=27, k=3, s=2)
#         self.encoder3 = self.block(in_ch=27, middle_ch=108, out_ch=81, k=3, s=2)
#         self.encoder4 = self.block(in_ch=81, middle_ch=324, out_ch=243, k=3, s=2)
#         self.pool = nn.AdaptiveAvgPool2d((1,1))
        
#         self.A_fc = nn.Sequential(
#             nn.Linear(243 + 64, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#         )

#         self.decoder4 = self.block(in_ch=64 + 243, middle_ch=256, out_ch=128, k=3, s=1)
#         self.decoder3 = self.block(in_ch=128 + 81, middle_ch=128, out_ch=64, k=3, s=1)
#         self.decoder2 = self.block(in_ch=64 + 27, middle_ch=64, out_ch=32, k=3, s=1)
#         self.decoder1 = self.block(in_ch=32 + 9, middle_ch=16, out_ch=1, k=3, s=1)
        
#         self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

#     def get_v(self, i1, i2):
#         v = self.v(torch.cat((self.i1_head(i1), self.i2_head(i2)), dim=1))

#         return v

#     def get_i3(self, v, i2):
#         spatial_shapes = [tuple(i2.shape[2:])]

#         enc1 = self.encoder1(i2)
#         spatial_shapes.append(tuple(enc1.shape[2:]))

#         enc2 = self.encoder2(enc1)
#         spatial_shapes.append(tuple(enc2.shape[2:]))

#         enc3 = self.encoder3(enc2)
#         spatial_shapes.append(tuple(enc3.shape[2:]))

#         enc4 = self.encoder4(enc3)
#         spatial_shapes.append(tuple(enc4.shape[2:]))

#         bottleneck = self.A_fc(
#             torch.cat([
#                 self.pool(enc4).flatten(start_dim=1), 
#                 self.expand_v(v)
#             ], dim=1)
#         ).unflatten(dim=1, sizes=(64, 2, 2))

#         dec4 = self.decoder4(torch.cat([self.upsample(size=spatial_shapes[-1])(bottleneck), enc4], dim=1))
#         dec3 = self.decoder3(torch.cat([self.upsample(size=spatial_shapes[-2])(dec4), enc3], dim=1))
#         dec2 = self.decoder2(torch.cat([self.upsample(size=spatial_shapes[-3])(dec3), enc2], dim=1))
#         dec1 = self.decoder1(torch.cat([self.upsample(size=spatial_shapes[-4])(dec2), enc1], dim=1))

#         return self.conv(self.upsample(size=spatial_shapes[-5])(dec1)).squeeze()

#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3
    

# class Conv_FC(nn.Module):
#     def __init__(self,):
#         super().__init__()

#         self.i1_head = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1,1)),
#             nn.Flatten(start_dim=1),
#         )

#         self.i2_head = deepcopy(self.i1_head) 
        
#         self.v = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Linear(16, 2),
#             nn.Tanh(),
#         )

#         self.reduce_i2 = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(16**2, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
        
#         self.combine = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 224),
#             nn.BatchNorm1d(224),
#             nn.ReLU(),
#             nn.Linear(224, 256),
#             nn.Unflatten(dim=1, unflattened_size=(16, 16)),
#         )

#     def get_v(self, i1, i2):
#         v = self.v(torch.cat((self.i1_head(i1), self.i2_head(i2)), dim=1))

#         return v

#     def get_i3(self, v, i2):
#         reduced_i2 = self.reduce_i2(i2)
#         expanded_v = self.expand_v(v)

#         return self.combine(torch.cat([reduced_i2, expanded_v], dim=1))

#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3


# class FC_Conv(nn.Module):
#     def __init__(self, seed):
#         super().__init__()

#         torch.manual_seed(seed)

#         self.v = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 2),
#             nn.Tanh(),
#         )

#         self.expand_v = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )

#         self.block = lambda in_ch, middle_ch, out_ch, k, s : nn.Sequential(
#             torch.nn.Conv2d(in_ch, middle_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, middle_ch, kernel_size=k, stride=s, padding=1, groups=middle_ch),
#             torch.nn.BatchNorm2d(middle_ch),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(middle_ch, out_ch, kernel_size=1, stride=1, padding=0),
#             torch.nn.BatchNorm2d(out_ch),
#             torch.nn.ReLU(),
#         )
        

#         '''
#         A is a UNet architecture
#         ''' 

#         self.encoders = nn.ModuleList([
#             self.block(in_ch=1, middle_ch=12, out_ch=9, k=3, s=2),
#             self.block(in_ch=9, middle_ch=36, out_ch=27, k=3, s=2),
#             self.block(in_ch=27, middle_ch=108, out_ch=81, k=3, s=2),
#             self.block(in_ch=81, middle_ch=324, out_ch=243, k=3, s=2),
#         ])

#         self.A_fc = nn.Sequential(
#             nn.Linear(243 + 64, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#         )

#         self.decoders = nn.ModuleList([
#             self.block(in_ch=64 + 243, middle_ch=256, out_ch=128, k=3, s=1),
#             self.block(in_ch=128 + 81, middle_ch=128, out_ch=64, k=3, s=1),
#             self.block(in_ch=64 + 27, middle_ch=64, out_ch=32, k=3, s=1),
#             self.block(in_ch=32 + 9, middle_ch=16, out_ch=1, k=3, s=1),
#         ])

#         self.pool = nn.AdaptiveAvgPool2d((1,1))
#         self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)


#     def get_v(self, i1, i2):
#         images = torch.cat([i1, i2], dim=1)

#         return self.v(images)

#     def get_i3(self, v, i2):
#         encs = []
#         spatial_shapes = [tuple(i2.shape[2:])]

#         enc = i2
#         for encoder in self.encoders:
#             enc = encoder(enc)

#             encs.append(enc)
#             spatial_shapes.append(tuple(enc.shape[2:]))


#         bottleneck = self.A_fc(
#             torch.cat([
#                 self.pool(encs[-1]).flatten(start_dim=1), 
#                 self.expand_v(v)
#             ], dim=1)
#         ).unflatten(dim=1, sizes=(64, 2, 2))


#         spatial_shapes = spatial_shapes[::-1]
#         encs = encs[::-1]


#         out = bottleneck
#         for i, decoder in enumerate(self.decoders):
#             out = decoder(torch.cat(
#                 [
#                     F.interpolate(input=out, size=spatial_shapes[i], mode='bilinear', align_corners=True), 
#                     encs[i]
#                 ], 
#                 dim=1
#             ))

        
#         return self.conv(F.interpolate(input=out, size=spatial_shapes[-1], mode='bilinear', align_corners=True)).squeeze()


#     def forward(self, i1, i2):
#         v = self.get_v(i1, i2)

#         pred_i3 = self.get_i3(v, i2)

#         return v, pred_i3