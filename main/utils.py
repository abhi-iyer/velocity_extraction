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


def plotly_scatter(data, colors, title):
    if colors is None:
        hex_colors = ['#000000' for _ in range(data.shape[0])]
    else:
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
    elif data.shape[1] == 1:
        # plot data vs. its color (R)
        scatter = go.Scatter(
            x=data[:, 0],
            y=(colors[:, 0] * 255).astype(int),
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            xaxis_title='dim 1',
            yaxis_title='color value',
        )
    else:
        raise Exception('Unsupported data dimensionality for a plotly scatter plot.')


    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title={'text' : title},
        **args,
    )

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
    

'''
def general_loss_fn_old(model, images, loss_fn_args):
    # preliminaries
    image_loss = None
    loop_loss = None
    shortcut_loss = None
    isometry_loss = None


    path_len = images.shape[1]

    gt_imgs = list(get_i1_i2_i3(images))
    gt_imgs[0] = gt_imgs[0].squeeze()
    gt_imgs[1] = gt_imgs[1].squeeze()
    gt_imgs[2] = gt_imgs[2].squeeze()

    i2 = gt_imgs[1][:, :-1, :]
    # i3 = gt_imgs[0][:, 1:, :]
    i4 = gt_imgs[1][:, 1:, :]
    # i5 = gt_imgs[2][:, 1:, :]
    
    
    pred_vs_i1_i2, pred_i3_a = model_full(model, first_img=gt_imgs[0], second_img=gt_imgs[1], apply_img=gt_imgs[1])

    if (path_len % 2 == 0):
        pred_i3_a = pred_i3_a[:, :-1, :, :, :]


    # image prediction loss
    image_loss = mse_loss_fn(pred_i3_a, gt_imgs[2], loss_fn_args['image_loss_weight'])


    # loop closure loss
    if loss_fn_args['loop_loss_weight']:
        pred_vs_i1_i2_sum = pred_vs_i1_i2.sum(dim=1)

        loop_loss = mse_loss_fn(pred_vs_i1_i2_sum, torch.zeros_like(pred_vs_i1_i2_sum).cuda(), loss_fn_args['loop_loss_weight'])


    # shortcut estimation loss
    if loss_fn_args['shortcut_loss_weight']:
        pred_i1_a = model_decoder(model=model, pred_vs=-pred_vs_i1_i2, apply_img=gt_imgs[1])
        pred_i1_b = model_decoder(model=model, pred_vs=-2*pred_vs_i1_i2, apply_img=gt_imgs[2])

        pred_i2_a = model_decoder(model=model, pred_vs=pred_vs_i1_i2, apply_img=gt_imgs[0])
        pred_i2_b = model_decoder(model=model, pred_vs=-pred_vs_i1_i2, apply_img=gt_imgs[2])
        pred_i2_c = model_decoder(model=model, pred_vs=-pred_vs_i1_i2[:, :-1, :] - pred_vs_i1_i2[:, 1:, :], apply_img=i4)

        pred_i3_b = model_decoder(model=model, pred_vs=2*pred_vs_i1_i2, apply_img=gt_imgs[0])

        pred_i4 = model_decoder(model=model, pred_vs=pred_vs_i1_i2[:, :-1, :] + pred_vs_i1_i2[:, 1:, :], apply_img=i2)


        shortcut_loss = (
            mse_loss_fn(pred_i1_a, gt_imgs[0], loss_fn_args['shortcut_loss_weight']) + \
            mse_loss_fn(pred_i1_b, gt_imgs[0], loss_fn_args['shortcut_loss_weight']) + \
            
            mse_loss_fn(pred_i2_a, gt_imgs[1], loss_fn_args['shortcut_loss_weight']) + \
            mse_loss_fn(pred_i2_b, gt_imgs[1], loss_fn_args['shortcut_loss_weight']) + \
            mse_loss_fn(pred_i2_c, i2, loss_fn_args['shortcut_loss_weight']) + \

            mse_loss_fn(pred_i3_b, gt_imgs[2], loss_fn_args['shortcut_loss_weight']) + \

            mse_loss_fn(pred_i4, i4, loss_fn_args['shortcut_loss_weight'])
        )


    # isometry loss
    if loss_fn_args['isometry_loss_weight']:
        isometry_loss = isometry_loss_fn(pred_vs_i1_i2, loss_fn_args, gt_imgs[0], gt_imgs[1], loss_fn_args['isometry_loss_weight'])



    loss_dict = {
        'image_loss' : image_loss,
        'loop_loss' : loop_loss,
        'shortcut_loss' : shortcut_loss,
        'isometry_loss' : isometry_loss,
    }

    return {k: v for k, v in loss_dict.items() if v is not None}
'''



def general_loss_fn(model, images, loss_fn_args, model_args, epoch_pct):
    '''
    preliminaries
    '''
    image_loss = None 
    loop_loss = None
    shortcut_loss = None
    isometry_loss = None


    path_len = images.shape[1]

    gt_imgs = list(get_i1_i2_i3(images))
    gt_imgs[0] = gt_imgs[0].squeeze()
    gt_imgs[1] = gt_imgs[1].squeeze()
    gt_imgs[2] = gt_imgs[2].squeeze()

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
                model_decoder(model=model, pred_vs=-pred_vs_i1_i2[:, :-1, :] - pred_vs_i1_i2[:, 1:, :], apply_img=i4), 
                i2, 
                num_output_labels=model_args['num_output_labels'], 
                weight=loss_fn_args['shortcut_loss_weight'],
            ) + \
            focal_loss_fn(
                model_decoder(model=model, pred_vs=pred_vs_i1_i2[:, :-1, :] + pred_vs_i1_i2[:, 1:, :], apply_img=i2), 
                i4, 
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
        )


    '''
    isometry loss
    '''
    if loss_fn_args['isometry_loss_weight']:
       isometry_loss = isometry_loss_fn(pred_vs_i1_i2, loss_fn_args, gt_imgs[0], gt_imgs[1], weight=loss_fn_args['isometry_loss_weight'])



    # '''
    # volume minimization loss
    # '''
    # if model_args['v_dim'] > 2 and epoch_pct > 0.5:
    #     centered_pred_vs = pred_vs_i1_i2 - pred_vs_i1_i2.mean(dim=1).unsqueeze(1)
    #     cov = torch.bmm(centered_pred_vs.transpose(-1, -2), centered_pred_vs)
    #     det = torch.det(cov + 1e-6 * torch.eye(model_args['v_dim']).cuda())

    #     volume_loss = (-torch.log(det)).mean()


    # '''
    # spatial locality loss
    # '''
    # spatial_positions = pred_vs_i1_i2.cumsum(dim=1)[:, 1:, :] - pred_vs_i1_i2.cumsum(dim=1)[:, :-1, :]
    # spatial_locality_loss = 0.5 * torch.norm(spatial_positions, p=2, dim=-1).mean(dim=-1).sum()


    # '''
    # eigenvalue loss
    # '''
    # if epoch_pct >= 0.5:
    #     pred_vs = pred_vs_i1_i2.flatten(start_dim=0, end_dim=1)
    #     pred_vs = pred_vs - pred_vs.mean(dim=0)
    #     cov = torch.mm(pred_vs.t(), pred_vs) / (pred_vs.size(0) - 1)
    #     eigs = torch.linalg.eigvalsh(cov).sort(descending=True).values
    #     eig_cutoff = eigs.mean() - eigs.std() # 1 std from the mean
    #     eigenvalue_loss = 1e1 * eigs[eigs <= eig_cutoff].sum()

    
    # '''
    # entropy loss: WON'T WORK
    # '''
    # prob = pred_vs_i1_i2.softmax(dim=-1).mean(dim=-2)
    # entropy_loss = 1e1 * -torch.sum(prob * torch.log(prob + 1e-7), dim=-1).mean()


    # '''
    # variance loss
    # '''
    # variance_loss = 1e1 * torch.var(torch.norm(pred_vs_i1_i2, p=2, dim=-1).flatten())


    # '''
    # normal vector loss
    # '''
    # pred_vs = pred_vs_i1_i2 - pred_vs_i1_i2.mean(dim=1, keepdim=True)
    # _, _, Vt = torch.linalg.svd(pred_vs, full_matrices=False)
    # normal_vectors = Vt[:, -1, :]
    # normal_vector_loss = 1e1 * torch.var(normal_vectors, dim=0).sum()



    loss_dict = {
        'image_loss' : image_loss,
        'loop_loss' : loop_loss,
        'shortcut_loss' : shortcut_loss,
        'isometry_loss' : isometry_loss,
    }

    # filter out loss terms that are not None
    return {k: v for k, v in loss_dict.items() if v is not None}



def baseline_loss_fn(model, images, loss_fn_args, model_args):
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
        plt.text(0, eps, f'Optimal eps = {eps:.2f}', color='red')
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


def isomap_reduction(tensor, reduced_dim):
    T, H, W = tensor.shape

    isomap = Isomap(n_neighbors=5, n_components=reduced_dim)
    embedding = isomap.fit_transform(np.reshape(tensor, (T, H * W)))

    return embedding[1:, :] - embedding[:-1, :]


def umap_reduction(tensor, reduced_dim):
    T, H, W = tensor.shape

    umap = UMAP(n_components=reduced_dim)
    embedding = umap.fit_transform(np.reshape(tensor, (T, H * W)))

    return embedding[1:, :] - embedding[:-1, :]


def pca_reduction(tensor, reduced_dim):
    T, H, W = tensor.shape

    pca = PCA(n_components=reduced_dim)
    embedding = pca.fit_transform(np.reshape(tensor, (T, H * W)))

    return embedding[1:, :] - embedding[:-1, :]


def ae_reduction(tensor, config, model_code):
    tensor = tensor.cuda()

    model = config['model_class'](seed=100, **config['model_args']).cuda()
    model.load_state_dict(torch.load('./models/model_{}.pt'.format(model_code)))
    _ = model.eval()


    with torch.no_grad():
        embedding = model.encoder(tensor).cpu().numpy()

    return embedding[1:, :] - embedding[:-1, :]


def mcnet_reduction(tensor, config, model_code):
    tensor = tensor.cuda()

    model = config['model_class'](seed=100, **config['model_args']).cuda()
    model.load_state_dict(torch.load('./models/model_{}.pt'.format(model_code)))
    _ = model.eval()

    

    return
