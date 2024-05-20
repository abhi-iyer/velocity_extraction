from imports import *


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # if using convs, use fixed conv backend algorithm
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    fix_randomness(seed=42)

    return


def he_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def img_tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def return_zero_if_nan(x):
    if x.isnan().item():
        return 0
    else:
        return x


def inclusive_range(x, low, high):
    return (x >= low).all().item() and (x <= high).all().item()


def convert_range(x, old_range, new_range):
    return ((x - old_range[0])/(old_range[1] - old_range[0])) * (new_range[1] - new_range[0]) + new_range[0]


def scale_to_n_labels(image, num_labels):
    '''
    image is a d1 x d2 x ... dk x H x W tensor

    returns a discrete long tensor with values in the set {0...num_labels-1} 
    '''
    assert inclusive_range(image, 0, 1)

    return (image * (num_labels - 1)).to(torch.long)


def normalize_label_tensor(image):
    '''
    if image is a d1 x d2 ... dk x num_labels x N x N tensor, return a d1 x ... x dk x N x N tensor with values in [0, 1] 
    
    if image is a d1 x d2 ... dk x num_labels=1 x N x N tensor, just return the image
    '''
    num_labels = image.shape[-3]
        
    if num_labels != 1:
        return image.argmax(dim=-3) / (float(num_labels) - 1)
    else:
        assert inclusive_range(image, 0, 1)

        return image.squeeze(dim=-3)


def get_dataset_dataloader(config):
    dataset_args = config['dataset_args']
    dataloader_args = config['dataloader_args']

    dataset_args.update(train=True)
    train_dataset = config['dataset_class'](**dataset_args)

    dataset_args.update(train=False)
    test_dataset = config['dataset_class'](**dataset_args)


    for i in tqdm(range(len(train_dataset))):
         _ = train_dataset[i]

    for i in tqdm(range(len(test_dataset))):
        _ = test_dataset[i]
        

    dataloader_args.update(
        shuffle=True,
    )
    train_loader = DataLoader(dataset=train_dataset, worker_init_fn=worker_init_fn, **dataloader_args)

    dataloader_args.update(
        shuffle=False
    )
    test_loader = DataLoader(dataset=test_dataset, worker_init_fn=worker_init_fn, **dataloader_args)


    return train_dataset, test_dataset, train_loader, test_loader


def norm_diff(x, y):
    return torch.norm(x - y, p=2, dim=-1)


def norm_diff_normalized(x, y):
    return torch.norm(x - y, p=2, dim=-1) / (torch.norm(x, p=2, dim=-1) + torch.norm(y, p=2, dim=-1))


def cosine_dissim(x, y):
    return (1 - nn.CosineSimilarity(dim=-1)(x, y)).clamp_(0.0, 1.0)


def get_i1_i2_i3(loop_path):
    if loop_path.ndim == 4:
        loop_path = loop_path.unsqueeze(0)

    return loop_path[:, :-1:2, :, :, :], loop_path[:, 1::2, :, :, :], loop_path[:, 2::2, :, :, :]


def get_i2_i3_i4(loop_path):
    if loop_path.ndim == 4:
        loop_path = loop_path.unsqueeze(0)

    return loop_path[:, 1:-2:2, :, :, :], loop_path[:, 2:-1:2, :, :, :], loop_path[:, 3:-1:2, :, :, :]


def mean_loss_dicts(loss_dicts, train):
    key = 'train_' if train else 'test_'

    mean_dict = {}

    for loss_dict in loss_dicts:
        for k, v in loss_dict.items():
            mean_dict[f'{key}{k}'] = mean_dict.get(f'{key}{k}', 0) + v
    
    return {k : v / len(loss_dicts) for k, v in mean_dict.items()}
    

def cmap2d(X, max_shift):
    # X is a N x 2 matrix

    colormap = mp.cm.ScalarMappable(mp.colors.Normalize(vmin=0.0, vmax=1.0), cmap='hsv')

    r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
    r = convert_range(r, (0, (2 * max_shift**2)**0.5), (0, 1))
    theta = torch.atan2(X[:, 1], X[:, 0])
    theta = convert_range(theta, (-torch.pi, torch.pi), (0, 1))

    colors = r.unsqueeze(1).numpy() * colormap.to_rgba(theta.numpy())
    colors = colors.clip(0, 1)

    colors[:, -1] = np.ones(colors.shape[0])

    return colors


def plotly_scatter(data, colors, title, x_range=None, y_range=None):
    if colors is None:
        hex_colors = ['#000000' for _ in range(data.shape[0])]
    else:
        colors = np.array(colors)
        
        hex_colors = ['#%02x%02x%02x' % tuple((rgba * 255).astype(int)[:3]) for rgba in colors]

    if data.shape[1] == 3:
        scatter = go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            scene = dict(
                xaxis_title='dim 1',
                yaxis_title='dim 2',
                zaxis_title='dim 3',
            )
        )
    elif data.shape[1] == 2:
        scatter = go.Scatter(
            x=data[:, 0],
            y=data[:, 1],   
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            xaxis_title='dim 1',
            yaxis_title='dim 2',

        )
    elif data.shape[1] == 1 or data.ndim == 1:
        data = data.reshape(-1, 1)
        # # plot data vs. its color (R)
        # scatter = go.Scatter(
        #     x=data[:, 0],
        #     y=(colors[:, 0] * 255).astype(int),
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color=hex_colors
        #     )
        # )
        scatter = go.Scatter(
            x=data[:, 0],
            y=np.zeros(data.shape[0]),
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            xaxis_title='dim 1',
            yaxis_title='dim 2',
        )
    else:
        raise Exception('Unsupported data dimensionality for a plotly scatter plot.')


    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title={'text' : title},
        **args,
    )
    
    if data.shape[1] == 2 or data.shape[1] == 1:
        if x_range:
            fig.update_xaxes(range=x_range)
        if y_range:
            fig.update_yaxes(range=y_range)

    return fig



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss


        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



def model_encoder(model, first_img, second_img):
    bs = first_img.shape[0]

    pred_vs = model.get_v(
        first_img.flatten(start_dim=0, end_dim=1),
        second_img.flatten(start_dim=0, end_dim=1),
    )

    pred_vs = pred_vs.unflatten(dim=0, sizes=(bs, -1))

    return pred_vs



def model_decoder(model, pred_vs, apply_img):
    bs = apply_img.shape[0]

    pred_imgs = model.get_i3(
        pred_vs.flatten(start_dim=0, end_dim=1),
        apply_img.flatten(start_dim=0, end_dim=1),
    )

    pred_imgs = pred_imgs.unflatten(dim=0, sizes=(bs, -1))

    return pred_imgs




def model_full(model, first_img, second_img, apply_img):
    pred_vs = model_encoder(model, first_img, second_img)
    pred_imgs = model_decoder(model, pred_vs, apply_img)

    return pred_vs, pred_imgs



def isometry_loss_fn(pred_vs, loss_fn_args, similarity_img1, similarity_img2, weight):
    if pred_vs.numel() == 0:
        return 0

    isometry_numerator = torch.norm(pred_vs, p=2, dim=-1).flatten()

    isometry_denominator = cosine_dissim(
        similarity_img1.flatten(start_dim=2, end_dim=-1),
        similarity_img2.flatten(start_dim=2, end_dim=-1),
    ).flatten()

    close_points = isometry_denominator < loss_fn_args['isometry_similarity_threshold']

    if close_points.sum() > 1:
        isometry_loss = weight * torch.var(isometry_numerator[close_points])
    else:
        isometry_loss = torch.Tensor([0]).cuda()

    if isometry_loss.isnan().item():
        breakpoint()

    return isometry_loss



def focal_loss_fn(input_imgs, target_imgs, num_output_labels, weight=1):    
    if (input_imgs.numel() == 0) or (target_imgs.numel() == 0):
        return 0
    else:
        target_shape = normalize_label_tensor(input_imgs).shape

        loss = weight * FocalLoss(alpha=1, gamma=2)(
            input_imgs.flatten(start_dim=0, end_dim=1),
            scale_to_n_labels(target_imgs.view(target_shape).flatten(start_dim=0, end_dim=1), num_labels=num_output_labels),
        )

        return loss
    


def mse_loss_fn(input_data, target_data, weight=1):
    mse = nn.MSELoss(reduction='mean')

    if (input_data.numel() == 0) or (target_data.numel() == 0):
        return 0
    else:
        loss = weight * mse(input_data, target_data.view_as(input_data))

        return loss



def spatial_locality_loss_fn(pred_vs, weight=1):
    loss = weight * nn.SmoothL1Loss(reduction='none')(
        pred_vs,
        torch.zeros_like(pred_vs).cuda(),
    ).flatten(start_dim=1).sum(dim=-1).mean()


    return loss



def general_loss_fn(model, images, loss_fn_args, model_args, epoch_pct):
    '''
    preliminaries
    '''
    image_loss = None 
    loop_loss = None
    shortcut_loss = None
    isometry_loss = None
    spatial_locality_loss = None


    path_len = images.shape[1]

    gt_imgs = list(get_i1_i2_i3(images))
    gt_imgs[0] = gt_imgs[0].squeeze()
    gt_imgs[1] = gt_imgs[1].squeeze()
    gt_imgs[2] = gt_imgs[2].squeeze()

    i1 = gt_imgs[0][:, :-1, :]
    i2 = gt_imgs[1][:, :-1, :]
    i4 = gt_imgs[1][:, 1:, :]
    
    # NOTE: pred_vs[:, :-1, :] describes inferred velocities from i2 -> i3 and pred_vs[:, 1:, :] describes inferred velocities from i3 -> i4
    pred_vs_i1_i2, pred_i3 = model_full(model, first_img=gt_imgs[0], second_img=gt_imgs[1], apply_img=gt_imgs[1])

    if (path_len % 2 == 0):
        pred_i3 = pred_i3[:, :-1, :, :, :]



    '''
    image prediction loss
    '''
    image_loss = focal_loss_fn(pred_i3, gt_imgs[2], num_output_labels=model_args['num_output_labels'], weight=loss_fn_args['image_loss_weight'])


    '''
    loop closure loss
    '''
    if loss_fn_args['loop_loss_weight']:
        pred_vs_i1_i2_sum = pred_vs_i1_i2.sum(dim=1)

        loop_loss = mse_loss_fn(pred_vs_i1_i2_sum, torch.zeros_like(pred_vs_i1_i2_sum).cuda(), weight=loss_fn_args['loop_loss_weight'])


    '''
    shortcut estimation loss
    '''
    if loss_fn_args['shortcut_loss_weight']:
        shortcut_loss = (         
            focal_loss_fn(
                model_decoder(model=model, pred_vs=pred_vs_i1_i2[:, :-1, :] + pred_vs_i1_i2[:, 1:, :], apply_img=i2), 
                i4, 
                num_output_labels=model_args['num_output_labels'], 
                weight=loss_fn_args['shortcut_loss_weight'],
            ) + \
            focal_loss_fn(
                model_decoder(model=model, pred_vs=-pred_vs_i1_i2[:, :-1, :] - pred_vs_i1_i2[:, 1:, :], apply_img=i4), 
                i2, 
                num_output_labels=model_args['num_output_labels'], 
                weight=loss_fn_args['shortcut_loss_weight'],
            )
            # # other terms
            # focal_loss_fn(
            #     model_decoder(model=model, pred_vs=2*pred_vs_i1_i2, apply_img=gt_imgs[0]),
            #     gt_imgs[2],
            #     num_output_labels=model_args['num_output_labels'], 
            #     weight=loss_fn_args['shortcut_loss_weight'],
            # ) + \
            # focal_loss_fn(
            #     model_decoder(model=model, pred_vs=-2*pred_vs_i1_i2, apply_img=gt_imgs[2]),
            #     gt_imgs[0],
            #     num_output_labels=model_args['num_output_labels'],
            #     weight=loss_fn_args['shortcut_loss_weight'],
            # )
            # even more other terms
            # focal_loss_fn(
            #     model_decoder(model=model, pred_vs=2*pred_vs_i1_i2[:, :-1, :] + pred_vs_i1_i2[:, 1:, :], apply_img=i1), 
            #     i4,
            #     num_output_labels=model_args['num_output_labels'], 
            #     weight=loss_fn_args['shortcut_loss_weight'],
            # ) + \
            # focal_loss_fn(
            #     model_decoder(model=model, pred_vs=-2*pred_vs_i1_i2[:, :-1, :] - pred_vs_i1_i2[:, 1:, :], apply_img=i4), 
            #     i1, 
            #     num_output_labels=model_args['num_output_labels'], 
            #     weight=loss_fn_args['shortcut_loss_weight'],
            # )
        )


    '''
    isometry loss
    '''
    if loss_fn_args['isometry_loss_weight']:
       isometry_loss = isometry_loss_fn(pred_vs_i1_i2, loss_fn_args, gt_imgs[0], gt_imgs[1], weight=loss_fn_args['isometry_loss_weight'])



    '''
    spatial locality loss
    '''
    if loss_fn_args['spatial_locality_loss']:
        if epoch_pct < 0.5:
            spatial_locality_loss = spatial_locality_loss_fn(
                pred_vs=pred_vs_i1_i2,
                weight=loss_fn_args['spatial_locality_loss']
            )



    loss_dict = {
        'image_loss' : image_loss,
        'loop_loss' : loop_loss,
        'shortcut_loss' : shortcut_loss,
        'isometry_loss' : isometry_loss,
        'spatial_locality_loss' : spatial_locality_loss,
    }

    # filter out loss terms that are not None
    return {k: v for k, v in loss_dict.items() if v is not None}



def ae_loss_fn(model, images, loss_fn_args, model_args, epoch_pct):
    bs = images.shape[0]

    image_loss = mse_loss_fn(
        input_data=model(images.flatten(start_dim=0, end_dim=1)).unflatten(dim=0, sizes=(bs, -1)),
        target_data=images,
        weight=loss_fn_args['image_loss_weight'],
    )


    loss_dict = {
        'image_loss' : image_loss,
    }

    # filter out loss terms that are not None
    return {k: v for k, v in loss_dict.items() if v is not None}


def mcnet_loss_fn(model, images, loss_fn_args, model_args, epoch_pct):
    L = images.shape[1]

    i1, i2, _ = get_i1_i2_i3(images[:, :L//2, :])

    _, pred_i2 = model(i1)

    image_loss = mse_loss_fn(
        input_data=pred_i2,
        target_data=i2,
        weight=loss_fn_args['image_loss_weight'],
    )


    loss_dict = {
        'image_loss' : image_loss,
    }

    # filter out loss terms that are not None
    return {k: v for k, v in loss_dict.items() if v is not None}



'''
utils for generating paper figures
'''


def compute_eps(X, k, show=True):
    distances = distance_matrix(X, X)
    
    # sort each row and take the k-th nearest neighbor distance (k+1 because the first is 0 distance to itself)
    # essentially find each point's distance to it's k'th closest neighbor
    sorted_distances = np.sort(distances, axis=1)
    kth_distances = sorted_distances[:, k]
    
    # sort all k'th closest neighbor distances
    sorted_kth_distances = np.sort(kth_distances)
    
    # find the knee point of the presumably increasing function
    # this epsilon value shows a significant change in the trend of distances
    knee_locator = KneeLocator(np.arange(len(sorted_kth_distances)), sorted_kth_distances, curve='convex', direction='increasing')
    eps = sorted_kth_distances[knee_locator.knee]

    # plot each point's k'th closest neighbor distance
    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(sorted_kth_distances)), sorted_kth_distances)
        plt.xlabel('Points')
        plt.ylabel('k-th Nearest Neighbor Distance')
        plt.title('k-NN Distance Plot for Determining EPS')
        plt.axhline(y=eps, color='r', linestyle='--')
        plt.text(0, eps, f'Optimal eps = {eps:.4f}', color='red')
        plt.show()
    
    return eps


def apply_dbscan(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    
    cluster_labels = db.labels_
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    return core_samples_mask, cluster_labels


def center_points(points):
    return points - np.mean(points, axis=0)


def apply_optimal_transformation(true, pred, true_filtered, pred_filtered):
    true = center_points(true)
    pred = center_points(pred)
    true_filtered = center_points(true_filtered)
    pred_filtered = center_points(pred_filtered)

    T = np.linalg.pinv(pred_filtered) @ true_filtered

    pred_transformed = pred @ T
    normalized_mse = np.mean(np.sum((true - pred_transformed)**2, axis=1)) / np.var(true)

    return normalized_mse, pred_transformed


def compute_error_metric(true, pred, num_clusters=30, show_knee_visualization=False):
    # if intrinsic/extrinsic velocities both 1D, move to 2D (make one dim just zeros)
    if true.shape[1] == 1 and pred.shape[1] == 1:
        true = np.hstack((true, np.zeros((true.shape[0], 1))))
        pred = np.hstack((pred, np.zeros((pred.shape[0], 1))))

    # eps is the maximum distance between two samples for one to be considered as 
    # in the neighborhood of the other. computed by using k-NN.
    eps = compute_eps(pred, k=num_clusters, show=show_knee_visualization)

    # apply dbscan algorithm for outlier rejection
    mask, cluster_labels = apply_dbscan(pred, eps=eps, min_samples=num_clusters)

    filtered_true = true[mask]
    filtered_pred = pred[mask]
    
    error, transformed_pred = apply_optimal_transformation(
        true=true,
        pred=pred,
        true_filtered=filtered_true,
        pred_filtered=filtered_pred,
    )

    return error, transformed_pred


def isomap_reduction(tensor, config, reduced_dim):
    fix_randomness(seed=42)

    T, H, W = tensor.shape

    isomap = Isomap(n_neighbors=5, n_components=reduced_dim)
    embedding = isomap.fit_transform(np.reshape(tensor, (T, H * W)))

    velocities = embedding[1:, :] - embedding[:-1, :]

    if config['task_dim'] == 1 and velocities.shape[1] == 1:
        velocities = np.hstack((velocities, np.zeros((velocities.shape[0], 1))))

    return velocities


def umap_reduction(tensor, config, reduced_dim):
    fix_randomness(seed=42)

    T, H, W = tensor.shape

    umap = UMAP(n_components=reduced_dim)
    embedding = umap.fit_transform(np.reshape(tensor, (T, H * W)))

    velocities = embedding[1:, :] - embedding[:-1, :]

    if config['task_dim'] == 1 and velocities.shape[1] == 1:
        velocities = np.hstack((velocities, np.zeros((velocities.shape[0], 1))))

    return velocities


def pca_reduction(tensor, config, reduced_dim):
    T, H, W = tensor.shape

    pca = PCA(n_components=reduced_dim)
    embedding = pca.fit_transform(np.reshape(tensor, (T, H * W)))

    velocities = embedding[1:, :] - embedding[:-1, :]

    if config['task_dim'] == 1 and velocities.shape[1] == 1:
        velocities = np.hstack((velocities, np.zeros((velocities.shape[0], 1))))

    return velocities


def ae_reduction(tensor, config, model_code):
    tensor = tensor.cuda()

    model = config['model_class'](seed=100, **config['model_args']).cuda()
    model.load_state_dict(torch.load('./models/model_{}.pt'.format(model_code)))
    _ = model.eval()


    with torch.no_grad():
        embedding = model.encoder(tensor).cpu().numpy()

    velocities = embedding[1:, :] - embedding[:-1, :]

    if config['task_dim'] == 1 and velocities.shape[1] == 1:
        velocities = np.hstack((velocities, np.zeros((velocities.shape[0], 1))))

    return velocities


def mcnet_reduction(tensor, config, model_code):
    tensor = tensor.cuda()

    model = config['model_class'](seed=100, **config['model_args']).cuda()
    model.load_state_dict(torch.load('./models/model_{}.pt'.format(model_code)))
    _ = model.eval()


    with torch.no_grad():
        tensor = tensor.view(1, -1, 1, *tensor.shape[1:])

        pred_vs, _ = model(tensor)
        pred_vs = pred_vs.squeeze(0).cpu().numpy()
        
    if config['task_dim'] == 1 and pred_vs.shape[1] == 1:
        pred_vs = np.hstack((pred_vs, np.zeros((pred_vs.shape[0], 1))))

    return pred_vs


def our_reduction(tensor, config, model_code):
    model = config['model_class'](seed=0, **config['model_args']).cuda()
    model.load_state_dict(torch.load('./models/model_{}.pt'.format(model_code)))
    _ = model.eval()

    i1 = tensor[:-1, :].unsqueeze(0).cuda()
    i2 = tensor[1:, :].unsqueeze(0).cuda()

    with torch.no_grad():
        pred_vs = model_encoder(model=model, first_img=i1, second_img=i2)

        pred_vs = pred_vs.squeeze(0).cpu().numpy()

    if config['task_dim'] == 1 and pred_vs.shape[1] == 1:
        pred_vs = np.hstack((pred_vs, np.zeros((pred_vs.shape[0], 1))))

    return pred_vs


def explained_variance(tensor):
    # tensor is a N x D tensor
    pca = PCA()

    pca.fit(tensor - tensor.mean(axis=0))

    return pca.explained_variance_ratio_


def firing_rates(
        config,
        gt_velocities, 
        pred_velocities,
        resample_every=30,
        T=10,
        dt=0.002,
        base_grid_period=0.35,
    ):
    gt_x = gt_velocities.cumsum(axis=0)

    pred_x = [np.zeros(2)]

    for i in range(pred_velocities.shape[0]):
        if (i % resample_every) == 0:
            pred_x.append(gt_x[i, :])
        else:
            pred_x.append(pred_x[-1] + pred_velocities[i, :])

    pred_x = np.stack(pred_x)[1:, :]

    num_grid_modules = 3
    num_neurons_per_module = 30

    max_firing_rate = 15 # hertz
    baseline_firing_rate = 0.02 * max_firing_rate # out-of-field

    time = np.arange(dt, T + dt, dt)

    base_grid_period *= 8
    grid_periods = base_grid_period + np.arange(num_grid_modules) * base_grid_period * np.sqrt(2)


    # define 3 plane waves that generate regular triangular lattice.
    # sum of cosine of dot product of position and b1, b2, b3 determines firing rate of cell at that position.
    # firing rates then form a hexagonal grid.
    # tl;dr: interference with these 3 vectors generates a hexagonal firing field/lattice.
    b1 = np.array([0, 2/np.sqrt(3)]) # oriented vertically, contributes to up/down components of hexagonal pattern
    b2 = np.array([1, -1/np.sqrt(3)]) # oriented with negative slope
    b3 = np.array([1, 1/np.sqrt(3)]) # oriented with positive slope


    # firing rates for all cells, for all modules
    # randomly distributed spatial phases for all cells, for all modules
    firing_rates = [np.zeros((num_neurons_per_module, len(time))) for _ in range(num_grid_modules)]
    preferred_phases = [np.random.rand(num_neurons_per_module, 2) for _ in range(num_grid_modules)]


    # generate grid cell rates over trajectory
    for i in range(num_neurons_per_module):
        for j in range(num_grid_modules):
            phase_offset = preferred_phases[j][i, :].reshape(-1, 1) * np.ones((2, len(time)))

            cos1 = np.cos(2 * np.pi * (b1/grid_periods[j]).reshape(1, -1) @ (pred_x.T - phase_offset))
            cos2 = np.cos(2 * np.pi * (b2/grid_periods[j]).reshape(1, -1) @ (pred_x.T - phase_offset))
            cos3 = np.cos(2 * np.pi * (b3/grid_periods[j]).reshape(1, -1) @ (pred_x.T - phase_offset))
            
            firing_rates[j][i, :] = baseline_firing_rate + (max_firing_rate - baseline_firing_rate) * \
                np.maximum(0, (1/3) * (cos1 + cos2 + cos3) - 0.2)
    

    # visualize spatial tuning of a cell at random: pick one cell from one module
    cellind = np.random.randint(0, num_neurons_per_module)  
    N = 100j

    max_vals = gt_x.max(axis=0)
    min_vals = gt_x.min(axis=0)

    fig = plt.figure(figsize=(6, 6))

    if config['task_dim'] == 1:
        extent = [min_vals[0], max_vals[0]]
        xs = np.mgrid[extent[0]:extent[1]:N]

        resampled = griddata(gt_x[:, 0], firing_rates[0][cellind, :], xs)

        plt.plot(resampled.T)
        plt.xlabel('dim 1')
        plt.ylabel('firing rate')

        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    elif config['task_dim'] == 2:
        extent = [min_vals[0], max_vals[0], min_vals[1], max_vals[1]]
        xs, ys = np.mgrid[extent[0]:extent[1]:N, extent[2]:extent[3]:N]

        resampled = griddata((gt_x[:, 0], gt_x[:, 1]), firing_rates[0][cellind, :], (xs, ys))

        plt.imshow(resampled.T, extent=extent, interpolation='gaussian', cmap='Spectral_r')
        
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        raise Exception('task dim > 2 is not supported')


    # plt.title('Spatial tuning of \nsynthetic grid cells given\n model generated velocities \nalong true trajectory')

    plt.tight_layout()    

    return fig







