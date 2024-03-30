class DiscreteShiftingMeansLoop(Dataset):
    def __init__(self, path, train, patch_size, random_walk_length, mean_init_range, std_init_range, max_mean_shift, num_training, num_testing):
        super().__init__()

        self.path = os.path.join(os.path.abspath(path), 'discrete_shifting_means_loop')
        self.config_path = os.path.join(self.path, 'config.txt')
        self.train = train

        assert (patch_size[0] == patch_size[1])

        self.patch_size = patch_size
        self.random_walk_length = random_walk_length
        self.mean_init_range = mean_init_range
        self.std_init_range = std_init_range
        self.max_mean_shift = max_mean_shift
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)


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
            self.seeds = torch.arange(0, self.num_training)
        else:
            self.seeds = torch.arange(self.num_training, self.num_training + self.num_testing)


        #self.velocities = torch.cartesian_prod(torch.Tensor([-self.max_mean_shift, 0, self.max_mean_shift]), torch.Tensor([-self.max_mean_shift, 0, self.max_mean_shift])).numpy()
        self.velocities = torch.cartesian_prod(torch.arange(-self.max_mean_shift, self.max_mean_shift + 1), torch.arange(-self.max_mean_shift, self.max_mean_shift + 1)).numpy()


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
    

    @staticmethod
    def bigauss(x, y, mean, std):
        return torch.exp(-( (x - mean[0])**2/(2 * std[0]**2) + (y - mean[1])**2/(2 * std[1]**2) ))


    def get_center(self, x):
        center = (x.shape[0]//2, x.shape[1]//2)

        return x[math.ceil(center[0] - self.patch_size[0]/2) : math.ceil(center[0] + self.patch_size[0]/2), math.ceil(center[1] - self.patch_size[1]/2) : math.ceil(center[1] + self.patch_size[1]/2)]


    def construct_image(self, X, Y, gauss_means, gauss_stds):
        patch = torch.zeros(*self.patch_size)

        for mean, std in zip(gauss_means, gauss_stds):
            patch += self.get_center(self.bigauss(X, Y, mean, std))

        return patch


    def __getitem__(self, index):
        assert 0 <= index < len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        if os.path.exists(images_path) and os.path.exists(vs_path):
            return torch.load(images_path), torch.load(vs_path)

        np.random.seed(self.seeds[index])

        num_gaussians = 50
        xs = torch.linspace(-20, 20, steps=64)
        ys = torch.linspace(-20, 20, steps=64)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')

        gauss_means = []
        gauss_stds = []
        for _ in range(num_gaussians):
            gauss_means.append(
                list(np.random.uniform(low=min(self.mean_init_range), high=max(self.mean_init_range), size=(2,)))
            )

            gauss_stds.append(
                list(np.random.uniform(low=min(self.std_init_range), high=max(self.std_init_range), size=(2,)))
            )

        vs = []
        for _ in range(self.random_walk_length):
            v = self.velocities[np.random.randint(0, len(self.velocities), size=(1,))].squeeze()
            vs.extend([v, v])

        for v in vs[:]:
            vs.append(-v)

        gt_vs = np.array(vs)

        # assert velocities sum to 0
        assert (gt_vs.sum(axis=0) == np.array([0.0, 0.0])).all()

        images = []
        for (vx, vy) in zip(*gt_vs.T):
            images.append(self.construct_image(X, Y, gauss_means, gauss_stds))

            for i in range(len(gauss_means)):
                gauss_means[i][0] += vx
                gauss_means[i][1] += vy

        images.append(self.construct_image(X, Y, gauss_means, gauss_stds))

        # assert that first and last image are the same (i.e. loop has been closed)
        assert (images[0] == images[-1]).all().item()

        images = torch.stack(images).unsqueeze(1)
        gt_vs = torch.Tensor(gt_vs)

        torch.save(images, images_path)
        torch.save(gt_vs, vs_path)
        
        return images, gt_vs

    def __len__(self):
        return len(self.seeds)



class FrequencyShift1D(Dataset):
    def __init__(
            self, 
            path, train, 
            random_walk_length=20,
            mean_init_range=(-5.0, 5.0), max_mean_shift=0.05,
            num_training=10e3, num_testing=2e3,
        ):
        super().__init__()

        self.path = os.path.join(os.path.abspath(path), 'frequencyshift1d_loops')
        self.config_path = os.path.join(self.path, 'config.txt')
        
        self.train = train
        self.random_walk_length = random_walk_length
        self.mean_init_range = mean_init_range
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

        self.X = torch.linspace(-5.0, 5.0, steps=100).cuda()


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
        config_str += '{}: {}\n'.format('mean init range', self.mean_init_range)
        config_str += '{}: {}\n'.format('max mean shift', self.max_mean_shift)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str


    def gauss(self, x, mean, std):
        return torch.exp(-( (x - mean)**2/(2 * std**2) ))
    

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
        means = torch.DoubleTensor(1).cuda().uniform_(self.mean_init_range[0], self.mean_init_range[1]).repeat(self.total_path_length, 1)
        std = 0.5


        vs = []
        cumsum = means[0,:].cpu()
        while len(vs) < self.random_walk_length:
            if (len(vs) == 0) or (torch.rand(1).item() > 0.8):
                v = torch.DoubleTensor(1).uniform_(-self.max_mean_shift, self.max_mean_shift)

            if inclusive_range(cumsum + 2*v, self.mean_init_range[0], self.mean_init_range[1]):
                vs.append(v)
                cumsum += 2 * v
        forward = torch.stack(vs).cuda()
        backward = -forward
        final_forward_point = cumsum.cuda()

        while not inclusive_range(final_forward_point + 2 * backward.cumsum(dim=0), self.mean_init_range[0], self.mean_init_range[1]):
            backward = backward[torch.randperm(self.random_walk_length), :]
        
        gt_vs = torch.vstack((
            torch.zeros(1).cuda(),
            forward.repeat_interleave(repeats=2, dim=0),
            backward.repeat_interleave(repeats=2, dim=0)
        ))

        torch._assert(torch.isclose(gt_vs.sum(dim=0), torch.zeros(1).cuda().double(), atol=1e-8).all(), "vs must sum to zero: {}".format(gt_vs.sum(dim=0)))

        shifted_means = (gt_vs.cumsum(dim=0) + means).float()


        '''
        generate images
        '''
        images = self.gauss(self.X, shifted_means, std).view(-1, 1, 1, len(self.X))
        images.clip_(0, 1)
        images = images.cpu()

        gt_vs = gt_vs[1:,:].cpu()

        torch._assert(torch.isclose(images[0], images[-1], atol=1e-8).all(), "first and last image must be the same in a loop")

        torch.save(images.float(), images_path)
        torch.save(gt_vs.float(), vs_path)

        return images, gt_vs


    def cmap(self, X):
        # X is a N x 1 matrix

        colors = np.zeros((X.shape[0], 3))
        colors[:, 0] = convert_range(X, (-self.max_mean_shift, self.max_mean_shift), (0, 1)).squeeze()
        colors[:, 2] = 0.5

        return colors

    
    def __len__(self):
        return len(self.seeds)