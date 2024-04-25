from imports import *
from utils import *


class ShrinkingBlob(Dataset):
    def __init__(
            self,
            path, train,
            patch_size, random_walk_length=20,
            std_init_range=(0.1, 0.6), max_std_shift=0.05,
            num_training=20e3, num_testing=5e3,
        ):
        super().__init__()

        assert patch_size[0] == patch_size[1]

        self.path = os.path.join(os.path.abspath(path), 'shrinking_blob_loops')
        self.config_path = os.path.join(self.path, 'config.txt')
        
        self.train = train
        self.patch_size = patch_size
        self.random_walk_length = random_walk_length
        self.std_init_range = std_init_range
        self.max_std_shift = max_std_shift
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)
        self.total_path_length = self.random_walk_length * 4 + 1

        self.create_dirs()

        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    shutil.rmtree(self.path)

                    self.create_dirs()
                    self.create_config()
        else:
            self.create_config()


        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))

    
        xs = torch.linspace(-1.0, 1.0, steps=self.patch_size[0]).double()
        ys = torch.linspace(-1.0, 1.0, steps=self.patch_size[0]).double()
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        self.X = X.repeat(self.total_path_length, 1, 1).cuda()
        self.Y = Y.repeat(self.total_path_length, 1, 1).cuda()


    def create_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'test'), exist_ok=True)

    
    def create_config(self):
        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def __repr__(self):
        config_str = ''
        config_str += '{}: {}\n'.format('patch size', self.patch_size)
        config_str += '{}: {}\n'.format('random walk length', self.random_walk_length)
        config_str += '{}: {}\n'.format('std initial range', self.std_init_range)
        config_str += '{}: {}\n'.format('max std shift', self.max_std_shift)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str


    def bigauss(self, means, stds):
        return torch.exp(
            -(self.X - means[0])**2/(2 * stds[:, 0].view(self.total_path_length, 1, 1)**2) - (self.Y - means[1])**2/(2 * stds[:, 1].view(self.total_path_length, 1, 1)**2)
        )


    def __getitem__(self, index):
        assert 0 <= index <= len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        if os.path.exists(images_path) and os.path.exists(vs_path):
            return torch.load(images_path), torch.load(vs_path)

        torch.manual_seed(self.seeds[index])


        '''
        generate velocities
        '''
        means = torch.DoubleTensor(2).cuda().uniform_(-0.8, 0.8)
        stds = torch.DoubleTensor(2).cuda().uniform_(self.std_init_range[0], self.std_init_range[1]).repeat(self.total_path_length, 1)


        vs = []
        cumsum = stds[0, :].cpu()
        while len(vs) < self.random_walk_length:
            if (len(vs) == 0) or (torch.rand(1).item() > 0.7):
                v = torch.DoubleTensor(2).uniform_(-self.max_std_shift, self.max_std_shift)

                if inclusive_range(cumsum + 2*v, self.std_init_range[0], self.std_init_range[1]):
                    vs.append(v)
                    cumsum += 2 * v
        forward = torch.stack(vs).cuda()
        backward = -forward
        final_forward_point = cumsum.cuda()

        while not inclusive_range(final_forward_point + 2 * backward.cumsum(dim=0), self.std_init_range[0], self.std_init_range[1]):
            backward = backward[torch.randperm(self.random_walk_length), :]

        gt_vs = torch.vstack((
            torch.zeros(2).cuda(),
            forward.repeat_interleave(repeats=2, dim=0),
            backward.repeat_interleave(repeats=2, dim=0)
        ))

        torch._assert(torch.isclose(gt_vs.sum(dim=0), torch.zeros(1).cuda().double(), atol=1e-8).all(), "vs must sum to zero: {}".format(gt_vs.sum(dim=0)))

        shifted_stds = (gt_vs.cumsum(dim=0) + stds).float()

        '''
        generate images
        '''
        images = self.bigauss(means, shifted_stds).unsqueeze(1).cpu()
        images = convert_range(images, (images.min().item(), images.max().item()), (0, 1))

        torch._assert(torch.isclose(images[0], images[-1], atol=1e-8).all(), "first and last image must be the same in a loop")

        gt_vs = gt_vs[1:, :].cpu()

        torch.save(images.float(), images_path)
        torch.save(gt_vs.float(), vs_path)

        return images, gt_vs

    
    def generate_sample_trajectory(self, length):
        torch.manual_seed(torch.randint(0, len(self.seeds), size=(1,)).item())
        
        total_path_length = length + 1

        means = torch.DoubleTensor(2).cuda().uniform_(-0.8, 0.8)
        stds = torch.DoubleTensor(2).cuda().uniform_(self.std_init_range[0], self.std_init_range[1]).repeat(total_path_length, 1)

        vs = []
        cumsum = stds[0, :].cpu()
        while len(vs) < length:
            v = torch.DoubleTensor(2).uniform_(-self.max_std_shift, self.max_std_shift)

            if inclusive_range(cumsum + v, self.std_init_range[0], self.std_init_range[1]):
                vs.append(v)
                cumsum += v

        gt_vs = torch.vstack((
            torch.zeros(2).cuda(),
            torch.stack(vs).cuda()
        ))

        shifted_stds = (gt_vs.cumsum(dim=0) + stds).float()

        '''
        generate images
        '''
        X = self.X[0].repeat(total_path_length, 1, 1)
        Y = self.Y[0].repeat(total_path_length, 1, 1)

        images = torch.exp(
            -(X - means[0])**2/(2 * shifted_stds[:, 0].view(total_path_length, 1, 1)**2) - (Y - means[1])**2/(2 * shifted_stds[:, 1].view(total_path_length, 1, 1)**2)
        ).unsqueeze(1).cpu()
        images = convert_range(images, (images.min().item(), images.max().item()), (0, 1))

        gt_vs = gt_vs[1:, :].cpu()

        return images.float(), gt_vs.float()


    def cmap(self, X):
        # X is a N x 2 matrix

        colormap = mp.cm.ScalarMappable(mp.colors.Normalize(vmin=0.0, vmax=1.0), cmap='hsv')

        r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        r = convert_range(r, (0, (2 * self.max_std_shift**2)**0.5), (0, 1))
        theta = torch.atan2(X[:, 1], X[:, 0])
        theta = convert_range(theta, (-torch.pi, torch.pi), (0, 1))

        colors = r.unsqueeze(1).numpy() * colormap.to_rgba(theta.numpy())
        colors = colors.clip(0, 1)

        colors[:, -1] = np.ones(colors.shape[0])

        return colors

    
    def __len__(self):
        return len(self.seeds)


class ContinuousShiftingMeansLoops(Dataset):
    def __init__(
            self,
            path, train, 
            patch_size, random_walk_length=20,
            mean_init_range=(-20, 20), std_init_range=(0.5, 2.0), max_mean_shift=2.0, 
            num_training=20e3, num_testing=5e3,
        ):
        super().__init__()

        assert patch_size[0] == patch_size[1]

        self.path = os.path.join(os.path.abspath(path), 'continuous_shifting_means_loops')
        self.config_path = os.path.join(self.path, 'config.txt')
        
        self.train = train
        self.patch_size = patch_size
        self.random_walk_length = random_walk_length
        self.mean_init_range = mean_init_range
        self.std_init_range = std_init_range
        self.max_mean_shift = max_mean_shift
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)
        self.total_path_length = self.random_walk_length * 4 + 1

        self.create_dirs()

        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    shutil.rmtree(self.path)

                    self.create_dirs()
                    self.create_config()
        else:
            self.create_config()


        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))

    
        self.num_gaussians = 50
        xs = torch.linspace(self.mean_init_range[0], self.mean_init_range[1], steps=self.patch_size[0]*4).double()
        ys = torch.linspace(self.mean_init_range[0], self.mean_init_range[1], steps=self.patch_size[0]*4).double()
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        self.X = X.repeat(self.total_path_length, self.num_gaussians, 1, 1).cuda()
        self.Y = Y.repeat(self.total_path_length, self.num_gaussians, 1, 1).cuda()


    def create_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'test'), exist_ok=True)

    
    def create_config(self):
        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def __repr__(self):
        config_str = ''
        config_str += '{}: {}\n'.format('patch size', self.patch_size)
        config_str += '{}: {}\n'.format('random walk length', self.random_walk_length)
        config_str += '{}: {}\n'.format('mean initial range', self.mean_init_range)
        config_str += '{}: {}\n'.format('std initial range', self.std_init_range)
        config_str += '{}: {}\n'.format('max shift delta', self.max_mean_shift)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str


    def bigauss(self, mean, std):
        return torch.exp(
            -((self.X - mean[:, :, 0].view(self.total_path_length, self.num_gaussians, 1, 1))**2/(2 * std[:, :, 0].view(self.total_path_length, self.num_gaussians, 1, 1)**2) + \
            (self.Y - mean[:, :, 1].view(self.total_path_length, self.num_gaussians, 1, 1))**2/(2 * std[:, :, 1].view(self.total_path_length, self.num_gaussians, 1, 1)**2))
        ).sum(dim=1)


    def get_center_patch(self, x):
        center = (x.shape[1]//2, x.shape[2]//2)

        return x[
                    :,
                    math.ceil(center[0] - self.patch_size[0]/2) : math.ceil(center[0] + self.patch_size[0]/2), 
                    math.ceil(center[1] - self.patch_size[1]/2) : math.ceil(center[1] + self.patch_size[1]/2)
                ]


    def construct_image(self, gauss_means, gauss_stds):
        return self.get_center_patch(self.bigauss(gauss_means, gauss_stds))


    def __getitem__(self, index):
        assert 0 <= index <= len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        if os.path.exists(images_path) and os.path.exists(vs_path):
            return torch.load(images_path), torch.load(vs_path)

        torch.manual_seed(self.seeds[index])

        vs = torch.DoubleTensor(self.random_walk_length, 2).cuda().uniform_(-self.max_mean_shift, self.max_mean_shift).repeat_interleave(repeats=2, dim=0)
        gt_vs = torch.vstack((torch.zeros(2).cuda(), vs, -vs))

        torch._assert(torch.isclose(gt_vs.sum(dim=0), torch.zeros(2).cuda().double(), atol=1e-8).all(), "vs must sum to zero: {}".format(gt_vs.sum(dim=0)))

        gauss_means = torch.DoubleTensor(self.num_gaussians, 2).cuda().uniform_(
            min(self.mean_init_range),
            max(self.mean_init_range)
        ).repeat(self.total_path_length, 1, 1)

        gauss_stds = torch.DoubleTensor(self.num_gaussians, 2).cuda().uniform_(
            min(self.std_init_range),
            max(self.std_init_range)
        ).repeat(self.total_path_length, 1, 1)

        shifted_means = gt_vs.cumsum(dim=0).unsqueeze(1).repeat(1, self.num_gaussians, 1) + gauss_means

        images = self.construct_image(shifted_means, gauss_stds).unsqueeze(1).cpu()
        images = convert_range(images, (images.min().item(), images.max().item()), (0, 1))

        torch._assert(torch.isclose(images[0], images[-1], atol=1e-8).all(), "first and last image must be the same in a loop")

        gt_vs = gt_vs[1:, :].cpu()

        torch.save(images.float(), images_path)
        torch.save(gt_vs.float(), vs_path)

        return images, gt_vs

    
    def generate_sample_trajectory(self, length):
        torch.manual_seed(torch.randint(0, len(self.seeds), size=(1,)).item())
        
        total_path_length = length + 1

        vs = torch.DoubleTensor(length, 2).cuda().uniform_(-self.max_mean_shift, self.max_mean_shift)
        gt_vs = torch.vstack((torch.zeros(2).cuda(), vs))

        means = torch.DoubleTensor(self.num_gaussians, 2).cuda().uniform_(
            min(self.mean_init_range),
            max(self.mean_init_range)
        ).repeat(total_path_length, 1, 1)

        std = torch.DoubleTensor(self.num_gaussians, 2).cuda().uniform_(
            min(self.std_init_range),
            max(self.std_init_range)
        ).repeat(total_path_length, 1, 1)

        mean = gt_vs.cumsum(dim=0).unsqueeze(1).repeat(1, self.num_gaussians, 1) + means

        xs = torch.linspace(self.mean_init_range[0], self.mean_init_range[1], steps=self.patch_size[0]*4).double()
        ys = torch.linspace(self.mean_init_range[0], self.mean_init_range[1], steps=self.patch_size[0]*4).double()
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        X = X.repeat(total_path_length, self.num_gaussians, 1, 1).cuda()
        Y = Y.repeat(total_path_length, self.num_gaussians, 1, 1).cuda()

        images = self.get_center_patch(torch.exp(
            -((X - mean[:, :, 0].view(total_path_length, self.num_gaussians, 1, 1))**2/(2 * std[:, :, 0].view(total_path_length, self.num_gaussians, 1, 1)**2) + \
            (Y - mean[:, :, 1].view(total_path_length, self.num_gaussians, 1, 1))**2/(2 * std[:, :, 1].view(total_path_length, self.num_gaussians, 1, 1)**2))
        ).sum(dim=1)).unsqueeze(1).cpu()
        images = convert_range(images, (images.min().item(), images.max().item()), (0, 1))

        gt_vs = gt_vs[1:, :].cpu()

        return images.float(), gt_vs.float()


    def cmap(self, X):
        # X is a N x 2 matrix

        colormap = mp.cm.ScalarMappable(mp.colors.Normalize(vmin=0.0, vmax=1.0), cmap='hsv')

        r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        r = convert_range(r, (0, (2 * self.max_mean_shift**2)**0.5), (0, 1))
        theta = torch.atan2(X[:, 1], X[:, 0])
        theta = convert_range(theta, (-torch.pi, torch.pi), (0, 1))

        colors = r.unsqueeze(1).numpy() * colormap.to_rgba(theta.numpy())
        colors = colors.clip(0, 1)

        colors[:, -1] = np.ones(colors.shape[0])

        return colors

    
    def __len__(self):
        return len(self.seeds)


class StretchyBird2D(Dataset):
    def __init__(
            self, 
            path, train, 
            random_walk_length=20, 
            max_std_shift=0.5, 
            num_training=30e3, num_testing=3e3,
        ):
        super().__init__()

        self.path = os.path.join(os.path.abspath(path), 'stretchybird2d_loops')
        self.config_path = os.path.join(self.path, 'config.txt')
        
        self.train = train
        self.random_walk_length = random_walk_length
        self.max_std_shift = max_std_shift
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)
        self.total_path_length = self.random_walk_length * 4 + 1

        self.create_dirs()

        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    shutil.rmtree(self.path)

                    self.create_dirs()
                    self.create_config()
        else:
            self.create_config()


        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))

        '''
        bird characteristics
        '''
        self.N = 32
        self.body = torch.zeros((self.total_path_length, self.N, self.N)).cuda()
        self.X, self.Y = torch.meshgrid(torch.arange(self.N), torch.arange(self.N), indexing='xy')
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()
        
        self.neck_x_range = torch.arange(2, 12)
        self.neck_y = 18
        self.body[:, 12, self.neck_y] = 1

        self.leg_x_range = torch.arange(21, 31)
        self.leg1_y = 12
        self.leg2_y = 17
        self.body[:, [19, 20], self.leg1_y] = 1
        self.body[:, [19, 20], self.leg2_y] = 1
        self.body = self.create_body(tilt_degree=150)

        self.max_std = 10 / 1.3
        self.x_grid = torch.arange(10).cuda()

    
    def create_body(self, tilt_degree):
        body = deepcopy(self.body)

        center_x = self.N // 2
        center_y = self.N // 2
        radius_x = self.N // 6
        radius_y = self.N // 10

        x_tilted = (self.X - center_x) * np.cos(np.deg2rad(tilt_degree)) + (self.Y - center_y) * np.sin(np.deg2rad(tilt_degree))
        y_tilted = -(self.X - center_x) * np.sin(np.deg2rad(tilt_degree)) + (self.Y - center_y) * np.cos(np.deg2rad(tilt_degree))

        within_body = (x_tilted / radius_x) ** 2 + (y_tilted / radius_y) ** 2 <= 1

        body[:, within_body] = 1

        return body


    def create_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'test'), exist_ok=True)

    
    def create_config(self):
        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def __repr__(self):
        config_str = ''
        config_str += '{}: {}\n'.format('random walk length', self.random_walk_length)
        config_str += '{}: {}\n'.format('max shift delta', self.max_std_shift)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str


    def gauss(self, x, mean, std):
        return torch.exp(-( (x - mean)**2/(2 * std**2) ))

    
    def bigauss(self, x, y, mean, std):
        return torch.exp(-( (x - mean[0])**2/(2 * std[0]**2) + (y - mean[1])**2/(2 * std[1]**2) ))
    

    def __getitem__(self, index):
        assert 0 <= index <= len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        if os.path.exists(images_path) and os.path.exists(vs_path):
            return torch.load(images_path), torch.load(vs_path)

        torch.manual_seed(self.seeds[index])
        
        '''
        generate velocities
        '''
        stds = torch.DoubleTensor(2).cuda().uniform_(0, self.max_std).repeat(self.total_path_length, 1)

        vs = []
        cumsum = stds[0,:].cpu()
        while len(vs) < self.random_walk_length:
            v = torch.DoubleTensor(2).uniform_(-self.max_std_shift, self.max_std_shift)

            if inclusive_range(cumsum + 2*v, 0, self.max_std):
                vs.append(v)
                cumsum += 2 * v
        forward = torch.stack(vs).cuda()
        backward = -forward
        final_forward_point = cumsum.cuda()

        while not inclusive_range(final_forward_point + 2 * backward.cumsum(dim=0), 0, self.max_std):
            backward = backward[torch.randperm(self.random_walk_length), :]
        
        gt_vs = torch.vstack((
            torch.zeros(2).cuda(),
            forward.repeat_interleave(repeats=2, dim=0),
            backward.repeat_interleave(repeats=2, dim=0)
        ))

        torch._assert(torch.isclose(gt_vs.sum(dim=0), torch.zeros(2).cuda().double(), atol=1e-8).all(), "vs must sum to zero: {}".format(gt_vs.sum(dim=0)))

        shifted_stds = (gt_vs.cumsum(dim=0) + stds).float()

        images = deepcopy(self.body)
        

        '''
        assemble neck and legs
        '''
        images[:, self.neck_x_range, self.neck_y] = self.gauss(
            self.x_grid.repeat(self.total_path_length, 1),
            self.x_grid.max().repeat(self.total_path_length, len(self.x_grid)),
            shifted_stds[:, 0].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg1_y] = self.gauss(
            self.x_grid.repeat(self.total_path_length, 1),
            self.x_grid.min().repeat(self.total_path_length, len(self.x_grid)),
            shifted_stds[:, 1].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg2_y] = self.gauss(
            self.x_grid.repeat(self.total_path_length, 1),             
            self.x_grid.min().repeat(self.total_path_length, len(self.x_grid)),
            shifted_stds[:, 1].repeat(len(self.x_grid), 1).T,
        )
        

        # images gets cropped from 32 x 32 --> 32 x 12
        images = images[:, :, 11:-9]
        images = images.unsqueeze(1).cpu()
        images.clip_(0, 1)

        torch._assert(torch.isclose(images[0], images[-1], atol=1e-8).all(), "first and last image must be the same in a loop")
        
        gt_vs = gt_vs[1:, :].cpu()

        torch.save(images.float(), images_path)
        torch.save(gt_vs.float(), vs_path)

        return images, gt_vs


    def generate_sample_trajectory(self, length):
        torch.manual_seed(torch.randint(0, len(self.seeds), size=(1,)).item())

        total_path_length = length + 1

        stds = torch.DoubleTensor(2).cuda().uniform_(0, self.max_std).repeat(total_path_length, 1)

        vs = []
        cumsum = stds[0,:].cpu()
        while len(vs) < length:
            v = torch.DoubleTensor(2).uniform_(-self.max_std_shift, self.max_std_shift)

            if inclusive_range(cumsum + v, 0, self.max_std):
                vs.append(v)
                cumsum += v
        
        gt_vs = torch.vstack((
            torch.zeros(2).cuda(),
            torch.stack(vs).cuda()
        ))

        shifted_stds = (gt_vs.cumsum(dim=0) + stds).float()

        images = deepcopy(self.body)
        images = images[0, :].repeat(total_path_length, 1, 1)


        images[:, self.neck_x_range, self.neck_y] = self.gauss(
            self.x_grid.repeat(total_path_length, 1),
            self.x_grid.max().repeat(total_path_length, len(self.x_grid)),
            shifted_stds[:, 0].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg1_y] = self.gauss(
            self.x_grid.repeat(total_path_length, 1),
            self.x_grid.min().repeat(total_path_length, len(self.x_grid)),
            shifted_stds[:, 1].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg2_y] = self.gauss(
            self.x_grid.repeat(total_path_length, 1),             
            self.x_grid.min().repeat(total_path_length, len(self.x_grid)),
            shifted_stds[:, 1].repeat(len(self.x_grid), 1).T,
        )
        

        # images gets cropped from 32 x 32 --> 32 x 12
        images = images[:, :, 11:-9]
        images = images.unsqueeze(1).cpu()
        images.clip_(0, 1)

        gt_vs = gt_vs[1:, :].cpu()

        return images.float(), gt_vs.float()
    

    def cmap(self, X):
        # X is a N x 2 matrix

        colormap = mp.cm.ScalarMappable(mp.colors.Normalize(vmin=0.0, vmax=1.0), cmap='hsv')

        r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        r = convert_range(r, (0, (2 * self.max_std_shift**2)**0.5), (0, 1))
        theta = torch.atan2(X[:, 1], X[:, 0])
        theta = convert_range(theta, (-torch.pi, torch.pi), (0, 1))

        colors = r.unsqueeze(1).numpy() * colormap.to_rgba(theta.numpy())
        colors = colors.clip(0, 1)

        colors[:, -1] = np.ones(colors.shape[0])

        return colors

    
    def __len__(self):
        return len(self.seeds)


class StretchyBird3D(Dataset):
    def __init__(
            self,
            path, train,
            random_walk_length=20, 
            max_std_shift=0.5,
            num_training=30e3, num_testing=3e3,
        ):
        super().__init__()

        self.path = os.path.join(os.path.abspath(path), 'stretchybird3d_loops')
        self.config_path = os.path.join(self.path, 'config.txt')
        
        self.train = train
        self.random_walk_length = random_walk_length
        self.max_std_shift = max_std_shift
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)
        self.total_path_length = self.random_walk_length * 4 + 1

        self.create_dirs()

        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    shutil.rmtree(self.path)

                    self.create_dirs()
                    self.create_config()
        else:
            self.create_config()


        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))

        '''
        bird characteristics
        '''
        self.N = 32
        self.body = torch.zeros((self.total_path_length, self.N, self.N)).cuda()
        self.X, self.Y = torch.meshgrid(torch.arange(self.N), torch.arange(self.N), indexing='xy')
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()
        
        self.neck_x_range = torch.arange(2, 12)
        self.neck_y = 18
        self.body[:, 12, self.neck_y] = 1

        self.leg_x_range = torch.arange(21, 31)
        self.leg1_y = 12
        self.leg2_y = 17
        self.body[:, [19, 20], self.leg1_y] = 1
        self.body[:, [19, 20], self.leg2_y] = 1
        self.body = self.create_body(tilt_degree=150)

        self.max_std = 10 / 1.3
        self.x_grid = torch.arange(10).cuda()

    
    def create_body(self, tilt_degree):
        body = deepcopy(self.body)

        center_x = self.N // 2
        center_y = self.N // 2
        radius_x = self.N // 6
        radius_y = self.N // 10

        x_tilted = (self.X - center_x) * np.cos(np.deg2rad(tilt_degree)) + (self.Y - center_y) * np.sin(np.deg2rad(tilt_degree))
        y_tilted = -(self.X - center_x) * np.sin(np.deg2rad(tilt_degree)) + (self.Y - center_y) * np.cos(np.deg2rad(tilt_degree))

        within_body = (x_tilted / radius_x) ** 2 + (y_tilted / radius_y) ** 2 <= 1

        body[:, within_body] = 1

        return body


    def create_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'test'), exist_ok=True)

    
    def create_config(self):
        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def __repr__(self):
        config_str = ''
        config_str += '{}: {}\n'.format('random walk length', self.random_walk_length)
        config_str += '{}: {}\n'.format('max shift delta', self.max_std_shift)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str


    def gauss(self, x, mean, std):
        return torch.exp(-( (x - mean)**2/(2 * std**2) ))

    
    def bigauss(self, x, y, mean, std):
        return torch.exp(-( (x - mean[0])**2/(2 * std[0]**2) + (y - mean[1])**2/(2 * std[1]**2) ))
    

    def __getitem__(self, index):
        assert 0 <= index <= len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        if os.path.exists(images_path) and os.path.exists(vs_path):
            return torch.load(images_path), torch.load(vs_path)

        torch.manual_seed(self.seeds[index])
        
        '''
        generate velocities
        '''
        stds = torch.DoubleTensor(3).cuda().uniform_(0, self.max_std).repeat(self.total_path_length, 1)

        vs = []
        cumsum = stds[0,:].cpu()
        while len(vs) < self.random_walk_length:
            v = torch.DoubleTensor(3).uniform_(-self.max_std_shift, self.max_std_shift)

            if inclusive_range(cumsum + 2*v, 0, self.max_std):
                vs.append(v)
                cumsum += 2 * v
        forward = torch.stack(vs).cuda()
        backward = -forward
        final_forward_point = cumsum.cuda()

        while not inclusive_range(final_forward_point + 2 * backward.cumsum(dim=0), 0, self.max_std):
            backward = backward[torch.randperm(self.random_walk_length), :]
        
        gt_vs = torch.vstack((
            torch.zeros(3).cuda(),
            forward.repeat_interleave(repeats=2, dim=0),
            backward.repeat_interleave(repeats=2, dim=0)
        ))

        torch._assert(torch.isclose(gt_vs.sum(dim=0), torch.zeros(3).cuda().double(), atol=1e-8).all(), "vs must sum to zero: {}".format(gt_vs.sum(dim=0)))

        shifted_stds = (gt_vs.cumsum(dim=0) + stds).float()

        images = deepcopy(self.body)


        '''
        assemble neck and legs
        '''
        images[:, self.neck_x_range, self.neck_y] = self.gauss(
            self.x_grid.repeat(self.total_path_length, 1),
            self.x_grid.max().repeat(self.total_path_length, len(self.x_grid)),
            shifted_stds[:, 0].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg1_y] = self.gauss(
            self.x_grid.repeat(self.total_path_length, 1),
            self.x_grid.min().repeat(self.total_path_length, len(self.x_grid)),
            shifted_stds[:, 1].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg2_y] = self.gauss(
            self.x_grid.repeat(self.total_path_length, 1),             
            self.x_grid.min().repeat(self.total_path_length, len(self.x_grid)),
            shifted_stds[:, 2].repeat(len(self.x_grid), 1).T,
        )


        # images gets cropped from 32 x 32 --> 32 x 12
        images = images[:, :, 11:-9]
        images = images.unsqueeze(1).cpu()
        images.clip_(0, 1)

        torch._assert(torch.isclose(images[0], images[-1], atol=1e-8).all(), "first and last image must be the same in a loop")

        gt_vs = gt_vs[1:, :].cpu()

        torch.save(images.float(), images_path)
        torch.save(gt_vs.float(), vs_path)

        return images, gt_vs

    
    def generate_sample_trajectory(self, length):
        torch.manual_seed(torch.randint(0, len(self.seeds), size=(1,)).item())

        total_path_length = length + 1

        stds = torch.DoubleTensor(3).cuda().uniform_(0, self.max_std).repeat(total_path_length, 1)

        vs = []
        cumsum = stds[0, :].cpu()
        while len(vs) < length:
            v = torch.DoubleTensor(3).uniform_(-self.max_std_shift, self.max_std_shift)

            if inclusive_range(cumsum + v, 0, self.max_std):
                vs.append(v)
                cumsum += v
        
        gt_vs = torch.vstack((
            torch.zeros(3).cuda(),
            torch.stack(vs).cuda()
        ))

        shifted_stds = (gt_vs.cumsum(dim=0) + stds).float()

        images = deepcopy(self.body)
        images = images[0, :].repeat(total_path_length, 1, 1)


        images[:, self.neck_x_range, self.neck_y] = self.gauss(
            self.x_grid.repeat(total_path_length, 1),
            self.x_grid.max().repeat(total_path_length, len(self.x_grid)),
            shifted_stds[:, 0].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg1_y] = self.gauss(
            self.x_grid.repeat(total_path_length, 1),
            self.x_grid.min().repeat(total_path_length, len(self.x_grid)),
            shifted_stds[:, 1].repeat(len(self.x_grid), 1).T,
        )
        images[:, self.leg_x_range, self.leg2_y] = self.gauss(
            self.x_grid.repeat(total_path_length, 1),             
            self.x_grid.min().repeat(total_path_length, len(self.x_grid)),
            shifted_stds[:, 2].repeat(len(self.x_grid), 1).T,
        )
        

        # images gets cropped from 32 x 32 --> 32 x 12
        images = images[:, :, 11:-9]
        images = images.unsqueeze(1).cpu()
        images.clip_(0, 1)

        gt_vs = gt_vs[1:, :].cpu()

        return images.float(), gt_vs.float()


    def cmap(self, X):
        # X is a N x 3 matrix

        colors = convert_range(X, (-self.max_std_shift, self.max_std_shift), (0, 1))
        
        return colors

    
    def __len__(self):
        return len(self.seeds)
  

class FrequencyShift1D(Dataset):
    def __init__(
            self, 
            path, train, 
            random_walk_length=20,
            init_range=(0.1, 5),
            max_shift=0.5, 
            num_training=20e3, num_testing=5e3,
        ):
        super().__init__()

        self.path = os.path.join(os.path.abspath(path), 'frequencyshift1d_loops')
        self.config_path = os.path.join(self.path, 'config.txt')
        
        self.train = train
        self.random_walk_length = random_walk_length
        self.init_range = init_range
        self.max_shift = max_shift
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)
        self.total_path_length = self.random_walk_length * 4 + 1

        self.create_dirs()

        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    shutil.rmtree(self.path)

                    self.create_dirs()
                    self.create_config()
        else:
            self.create_config()


        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))

        self.X = torch.linspace(10.0, 20.0, steps=100).cuda()


    def create_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'test'), exist_ok=True)

    
    def create_config(self):
        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def __repr__(self):
        config_str = ''
        config_str += '{}: {}\n'.format('random walk length', self.random_walk_length)
        config_str += '{}: {}\n'.format('init range', self.init_range)
        config_str += '{}: {}\n'.format('max shift delta', self.max_shift)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str
    

    def __getitem__(self, index):
        assert 0 <= index <= len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        if os.path.exists(images_path) and os.path.exists(vs_path):
            return torch.load(images_path), torch.load(vs_path)

        torch.manual_seed(self.seeds[index])
        
        '''
        generate velocities
        '''
        num_waves = 2

        values = torch.DoubleTensor(num_waves).cuda().uniform_(self.init_range[0], self.init_range[1]).repeat(self.total_path_length, 1)

        vs = []
        cumsum = values[0, :].cpu()
        while len(vs) < self.random_walk_length:
            # if len(vs) == 0 or torch.rand(1).item() > 0.5:
            v = torch.DoubleTensor(1).uniform_(-self.max_shift, self.max_shift)

            if inclusive_range(cumsum + 2*v, self.init_range[0], self.init_range[1]):
                vs.append(v)
                cumsum += 2 * v
        forward = torch.stack(vs).cuda()
        backward = -forward
        final_forward_point = cumsum.cuda()

        while not inclusive_range(final_forward_point + 2 * backward.cumsum(dim=0), self.init_range[0], self.init_range[1]):
            backward = backward[torch.randperm(self.random_walk_length), :]
        
        gt_vs = torch.vstack((
            torch.zeros(1).cuda(),
            forward.repeat_interleave(repeats=2, dim=0),
            backward.repeat_interleave(repeats=2, dim=0)
        ))

        torch._assert(torch.isclose(gt_vs.sum(dim=0), torch.zeros(1).cuda().double(), atol=1e-8).all(), "vs must sum to zero: {}".format(gt_vs.sum(dim=0)))

        shifted_values = (gt_vs.cumsum(dim=0) + values).float()


        '''
        generate images
        '''
        images = torch.sin(torch.einsum('pw,wn->pwn', shifted_values, self.X.repeat(num_waves, 1))).sum(dim=1)
        images = images.view(-1, 1, 1, self.X.shape[0])
        images = convert_range(images, (-num_waves, num_waves), (0.0, 1.0)).cpu()

        gt_vs = gt_vs[1:, :].cpu()

        torch._assert(torch.isclose(images[0], images[-1], atol=1e-8).all(), "first and last image must be the same in a loop")

        torch.save(images.float(), images_path)
        torch.save(gt_vs.float(), vs_path)

        return images, gt_vs
    

    def generate_sample_trajectory(self, length):
        torch.manual_seed(torch.randint(0, len(self.seeds), size=(1,)).item())

        total_path_length = length + 1

        '''
        generate velocities
        '''
        num_waves = 2

        values = torch.DoubleTensor(num_waves).cuda().uniform_(self.init_range[0], self.init_range[1]).repeat(total_path_length, 1)

        vs = []
        cumsum = values[0, :].cpu()
        while len(vs) < length:
            v = torch.DoubleTensor(1).uniform_(-self.max_shift, self.max_shift)

            if inclusive_range(cumsum + v, self.init_range[0], self.init_range[1]):
                vs.append(v)
                cumsum += v
        
        gt_vs = torch.vstack((
            torch.zeros(1).cuda(),
            torch.stack(vs).cuda(),
        ))

        shifted_values = (gt_vs.cumsum(dim=0) + values).float()


        images = torch.sin(torch.einsum('pw,wn->pwn', shifted_values, self.X.repeat(num_waves, 1))).sum(dim=1)
        images = images.view(-1, 1, 1, self.X.shape[0])
        images = convert_range(images, (-num_waves, num_waves), (0.0, 1.0)).cpu()

        gt_vs = gt_vs[1:, :].cpu()

        return images.float(), gt_vs.float()


    def cmap(self, X):
        # X is a N x 1 matrix

        colors = np.zeros((X.shape[0], 3))
        colors[:, 0] = convert_range(X, (-self.max_shift, self.max_shift), (0, 1)).squeeze()
        colors[:, 2] = 0.5

        return colors

    
    def __len__(self):
        return len(self.seeds)