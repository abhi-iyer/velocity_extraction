from imports import *
from models import *
from data import *
from utils import *


base = dict()

shift1_1 = deepcopy(base)
shift1_1.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


shift10 = deepcopy(base)
shift10.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=10,
    ),
    model_class=PINV_GRAY_1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


shift20 = deepcopy(base)
shift20.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=20,
    ),
    model_class=PINV_GRAY_1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


shift1_2 = deepcopy(base)
shift1_2.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_2,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)




shift1_vreduce_combine_tripletdiv = deepcopy(base)
shift1_vreduce_combine_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vreduce_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)

shift1_vreduce_combine_tripletsub = deepcopy(base)
shift1_vreduce_combine_tripletsub.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vreduce_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletSub(data_range=1.0, channel=1),
)

shift1_vreduce_LR_tripletdiv = deepcopy(base)
shift1_vreduce_LR_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vreduce_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)

shift1_vreduce_LR_tripletsub = deepcopy(base)
shift1_vreduce_LR_tripletsub.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vreduce_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletSub(data_range=1.0, channel=1),
)








shift1_vflat_combine_tripletdiv = deepcopy(base)
shift1_vflat_combine_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)

shift1_vflat_combine_tripletsub = deepcopy(base)
shift1_vflat_combine_tripletsub.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletSub(data_range=1.0, channel=1),
)

shift1_vflat_LR_tripletdiv = deepcopy(base)
shift1_vflat_LR_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)

shift1_vflat_LR_tripletsub = deepcopy(base)
shift1_vflat_LR_tripletsub.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(64, 64),
        eps_shift=1,
    ),
    model_class=PINV_GRAY_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletSub(data_range=1.0, channel=1),
)





shift1_rand_vflat_combine_regloss = deepcopy(base)
shift1_rand_vflat_combine_regloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)

shift1_rand_vflat_combine_tripletdiv = deepcopy(base)
shift1_rand_vflat_combine_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)

shift1_rand_vflat_combine_tripletsub = deepcopy(base)
shift1_rand_vflat_combine_tripletsub.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletSub(data_range=1.0, channel=1),
)

shift1_rand_vflat_LR_regloss = deepcopy(base)
shift1_rand_vflat_LR_regloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)

shift1_rand_vflat_LR_tripletdiv = deepcopy(base)
shift1_rand_vflat_LR_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)


shift1_rand_vflat_LR_tripletsub = deepcopy(base)
shift1_rand_vflat_LR_tripletsub.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletSub(data_range=1.0, channel=1),
)






shift5_combine_cropregloss = deepcopy(base)
shift5_combine_cropregloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_Combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_Loss(eps_shift=5, data_range=1.0, channel=1),
)


shift5_combine_croptripletdiv = deepcopy(base)
shift5_combine_croptripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_Combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_TripletDiv(eps_shift=5, data_range=1.0, channel=1),
)


shift5_combine_tripletdiv = deepcopy(base)
shift5_combine_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_Combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)


shift5_combine_regloss = deepcopy(base)
shift5_combine_regloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_Combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


shift5_LR_cropregloss = deepcopy(base)
shift5_LR_cropregloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_Loss(eps_shift=5, data_range=1.0, channel=1),
)


shift5_LR_croptripletdiv = deepcopy(base)
shift5_LR_croptripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_TripletDiv(eps_shift=5, data_range=1.0, channel=1),
)


shift5_LR_tripletdiv = deepcopy(base)
shift5_LR_tripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_TripletDiv(data_range=1.0, channel=1),
)


shift5_LR_regloss = deepcopy(base)
shift5_LR_regloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=NewCaltech256Pairs,
    dataset_args=dict(
        root="/home/abhiram/Documents/Fiete Lab/gcpc_velocity",
        image_size=(320, 320),
        image_type="L",
        patch_size=(32, 32),
        eps_shift=5,
    ),
    model_class=PINV_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)






shift1_rand_combine_cropregloss = deepcopy(base)
shift1_rand_combine_cropregloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_Loss(eps_shift=1, data_range=1.0, channel=1),
)

shift1_rand_combine_croptripletdiv = deepcopy(base)
shift1_rand_combine_croptripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_combine,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_TripletDiv(eps_shift=1, data_range=1.0, channel=1),
)

shift1_rand_LR_cropregloss = deepcopy(base)
shift1_rand_LR_cropregloss.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_Loss(eps_shift=1, data_range=1.0, channel=1),
)

shift1_rand_LR_croptripletdiv = deepcopy(base)
shift1_rand_LR_croptripletdiv.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=RandomPairs,
    dataset_args=dict(
        image_size=(320, 320),
        patch_size=(16, 16),
        eps_shift=1,
    ),
    model_class=PINV_RAND_vflat_LR,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Cropped_TripletDiv(eps_shift=1, data_range=1.0, channel=1),
)



PINV_CONFIGS = dict(
    shift1_1=shift1_1,
    shift10=shift10,
    shift20=shift20,
    shift1_2=shift1_2,

    shift1_vreduce_combine_tripletdiv=shift1_vreduce_combine_tripletdiv,
    shift1_vreduce_combine_tripletsub=shift1_vreduce_combine_tripletsub,
    shift1_vreduce_LR_tripletdiv=shift1_vreduce_LR_tripletdiv,
    shift1_vreduce_LR_tripletsub=shift1_vreduce_LR_tripletsub,

    shift1_vflat_combine_tripletdiv=shift1_vflat_combine_tripletdiv,
    shift1_vflat_combine_tripletsub=shift1_vflat_combine_tripletsub,
    shift1_vflat_LR_tripletdiv=shift1_vflat_LR_tripletdiv,
    shift1_vflat_LR_tripletsub=shift1_vflat_LR_tripletsub,


    shift1_rand_vflat_combine_regloss=shift1_rand_vflat_combine_regloss,
    shift1_rand_vflat_combine_tripletdiv=shift1_rand_vflat_combine_tripletdiv,
    shift1_rand_vflat_combine_tripletsub=shift1_rand_vflat_combine_tripletsub,
    shift1_rand_vflat_LR_regloss=shift1_rand_vflat_LR_regloss,
    shift1_rand_vflat_LR_tripletdiv=shift1_rand_vflat_LR_tripletdiv,
    shift1_rand_vflat_LR_tripletsub=shift1_rand_vflat_LR_tripletsub,


    shift5_combine_cropregloss=shift5_combine_cropregloss,
    shift5_combine_croptripletdiv=shift5_combine_croptripletdiv,
    shift5_combine_tripletdiv=shift5_combine_tripletdiv,
    shift5_combine_regloss=shift5_combine_regloss,
    shift5_LR_cropregloss=shift5_LR_cropregloss,
    shift5_LR_croptripletdiv=shift5_LR_croptripletdiv,
    shift5_LR_tripletdiv=shift5_LR_tripletdiv,
    shift5_LR_regloss=shift5_LR_regloss,



    shift1_rand_combine_cropregloss=shift1_rand_combine_cropregloss,
    shift1_rand_combine_croptripletdiv=shift1_rand_combine_croptripletdiv,
    shift1_rand_LR_cropregloss=shift1_rand_LR_cropregloss,
    shift1_rand_LR_croptripletdiv=shift1_rand_LR_croptripletdiv,
)