from imports import *
from utils import *
from data import *
from configs import CONFIGS

def runner(config_name, seed, generate_figures=True):
    torch.cuda.empty_cache()
    
    config = deepcopy(CONFIGS[config_name])

    config['model_args'].update(
        seed=seed,
    )
    model = config['model_class'](**config['model_args']).cuda()

    fix_randomness(seed=42)

    optimizer = config['optimizer_class'](model.parameters(), lr=config['lr'])

    train_dataset, test_dataset, train_loader, test_loader = get_dataset_dataloader(config)

    
    wandb_run = wandb.init(
        name=config_name,
        project='gcpc_velocity',
        entity='iyer',
        config=config, reinit=True, mode='online'
    )


    print('Model size: {}'.format(count_parameters(model)))

    for epoch in tqdm(range(config['num_epochs'])):
        '''
        train
        '''
        model.train()

        loss_dicts = []

        for images, _ in train_loader:
            images = images.cuda()

            optimizer.zero_grad()

            loss_dict = config['loss_fn'](
                model=model,
                images=images,
                loss_fn_args=config['loss_fn_args'],
                model_args=config['model_args'],
                epoch_pct=epoch/config['num_epochs'],
            )

            sum(l for l in loss_dict.values()).backward()
            optimizer.step()

            loss_dicts.append({k : v.detach().cpu().item() for k, v in loss_dict.items()})

        wandb_run.log(dict(
            epoch=epoch,
            **mean_loss_dicts(loss_dicts, train=True)
        ))


        loss_dicts = []

        if (epoch % 10) == 0:
            '''
            eval
            '''
            model.eval()

            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.cuda()

                    loss_dict = config['loss_fn'](
                        model=model,
                        images=images,
                        loss_fn_args=config['loss_fn_args'],
                        model_args=config['model_args'],
                        epoch_pct=epoch/config['num_epochs'],
                    )

                    loss_dicts.append({k : v.detach().cpu().item() for k, v in loss_dict.items()})

            
            wandb_run.log(dict(
                epoch=epoch,
                **mean_loss_dicts(loss_dicts, train=False)
            ))
        
        if (epoch % 50) == 0 and generate_figures:
            '''
            save figures
            '''
            save_figures(
                wandb_run, config,
                model,
                train_dataset, test_dataset,
                pred_space_plot=True,
                pred_images_plot=False,
            )
    
    if generate_figures:
        save_figures(
            wandb_run, config,
            model,
            train_dataset, test_dataset,
            pred_space_plot=True,
            pred_images_plot=True,
        )
    
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/model_{}.pt'.format(wandb_run.id))
    
    wandb_run.finish()

    torch.cuda.empty_cache()


def save_figures(wandb_run, config, model, train_dataset, test_dataset, pred_space_plot, pred_images_plot):
    model.eval()

    log = dict()

    
    fig1, axes1 = plt.subplots(
        4, 2,
        figsize=(13, 10),
        constrained_layout=True,
    )
    fig1.suptitle('training set images')
    axes1 = axes1.flatten()

    train_gt_velocities = []
    train_predicted_velocities = []
    train_colors = []
    for i in range(50):
        with torch.no_grad():
            index = np.random.randint(0, len(train_dataset), size=(1,))[0]
            images, gt_vs = train_dataset[index]

            images = images.cuda().unsqueeze(0)

            i1, i2, i3 = get_i1_i2_i3(images)

            pred_vs, pred_imgs = model(
                i1.flatten(start_dim=0, end_dim=1),
                i2.flatten(start_dim=0, end_dim=1)
            )
            pred_imgs = pred_imgs.unsqueeze(0)

            train_gt_velocities.append(gt_vs[::2, :].cpu().numpy())
            train_predicted_velocities.append(pred_vs.cpu().numpy())
            train_colors.append(train_dataset.cmap(gt_vs[::2, :].cpu()))

            if i < int(len(axes1)//2):
                rand_ind = np.random.randint(0, pred_imgs.shape[1], size=(1,))[0]

                axes1[2*i].imshow(i3.flatten(0, 2)[rand_ind, :, :].detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
                axes1[2*i].set_title("true")

                axes1[2*i + 1].imshow(normalize_label_tensor(pred_imgs)[:, rand_ind, :, :].squeeze(0).detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
                axes1[2*i + 1].set_title("pred")


    fig2, axes2 = plt.subplots(
        4, 2,
        figsize=(13, 10),
        constrained_layout=True,
    )
    fig2.suptitle('testing set images')
    axes2 = axes2.flatten()

    test_gt_velocities = []
    test_predicted_velocities = []
    test_colors = []
    for i in range(50):
        with torch.no_grad():
            index = np.random.randint(0, len(test_dataset), size=(1,))[0]
            images, gt_vs = test_dataset[index]

            images = images.cuda().unsqueeze(0)

            i1, i2, i3 = get_i1_i2_i3(images)

            pred_vs, pred_imgs = model(
                i1.flatten(start_dim=0, end_dim=1),
                i2.flatten(start_dim=0, end_dim=1)
            )
            pred_imgs = pred_imgs.unsqueeze(0)

            test_gt_velocities.append(gt_vs[::2, :].cpu().numpy())
            test_predicted_velocities.append(pred_vs.cpu().numpy())
            test_colors.append(test_dataset.cmap(gt_vs[::2, :].cpu()))

            if i < int(len(axes2)//2):
                rand_ind = np.random.randint(0, pred_imgs.shape[1], size=(1,))[0]

                axes2[2*i].imshow(i3.flatten(0, 2)[rand_ind, :, :].detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
                axes2[2*i].set_title("true")

                axes2[2*i + 1].imshow(normalize_label_tensor(pred_imgs)[:, rand_ind, :, :].squeeze(0).detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
                axes2[2*i + 1].set_title("pred")
                


    train_gt_velocities = np.concatenate(train_gt_velocities, axis=0)
    test_gt_velocities = np.concatenate(test_gt_velocities, axis=0)

    train_predicted_velocities = np.concatenate(train_predicted_velocities, axis=0)
    test_predicted_velocities = np.concatenate(test_predicted_velocities, axis=0)


    fig3 = plotly_scatter(train_gt_velocities, np.concatenate(train_colors, axis=0), 'training set: true velocity space')
    fig4 = plotly_scatter(test_gt_velocities, np.concatenate(test_colors, axis=0), 'testing set: true velocity space')
    fig5 = plotly_scatter(train_predicted_velocities, np.concatenate(train_colors, axis=0), 'training set: predicted velocity space')
    fig6 = plotly_scatter(test_predicted_velocities, np.concatenate(test_colors, axis=0), 'testing set: predicted velocity space')


    if pred_space_plot:
        log['images/train_true_velocity_space'] = wandb.Html(plotly.io.to_html(fig3))
        log['images/test_true_velocity_space'] = wandb.Html(plotly.io.to_html(fig4))
        log['images/train_pred_velocity_space'] = wandb.Html(plotly.io.to_html(fig5))
        log['images/test_pred_velocity_space'] = wandb.Html(plotly.io.to_html(fig6))
    
    if pred_images_plot:
        log['images/train_images_plot'] = wandb.Image(fig1)
        log['images/test_images_plot'] = wandb.Image(fig2)


    plt.close('all')

    wandb_run.log(log)


if __name__ == '__main__':
    runner('bird3d', seed=103)
    runner('bird3d', seed=104)