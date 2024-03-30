
from imports import *
from models import *
from data import *
from utils import *

base = dict()

# shift1_rand_LR1_mse = deepcopy(base)
# shift1_rand_LR1_mse.update(
#     lr=1e-3,
#     batch_size=768,
#     dataset_class=RandomPairs,
#     dataset_args=dict(
#         image_size=(100, 100),
#         patch_size=(8, 8),
#         eps_shift=1,
#         num_training=30e3,
#         num_testing=10e3,
#     ),
#     model_class=Rand8x8_LR1,
#     model_args=dict(
#     ),
#     optimizer_class=optim.Adam,
#     loss_fn=nn.MSELoss(),
# )

# shift1_rand_LR1_croppedmse = deepcopy(base)
# shift1_rand_LR1_croppedmse.update(
#     lr=1e-3,
#     batch_size=768,
#     dataset_class=RandomPairs,
#     dataset_args=dict(
#         image_size=(100, 100),
#         patch_size=(8, 8),
#         eps_shift=1,
#         num_training=30e3,
#         num_testing=10e3,
#     ),
#     model_class=Rand8x8_LR1,
#     model_args=dict(
#     ),
#     optimizer_class=optim.Adam,
#     loss_fn=MSE_Cropped_Loss(eps_shift=1),
# )




shift1_rand_LR1_MSE_ContrastV_Base = deepcopy(base)
shift1_rand_LR1_MSE_ContrastV_Base.update(
    lr=1e-3,
    batch_size=768,
    dataset_class=ThreeRandomOneShift,
    dataset_args=dict(
        patch_size=(8, 8),
        eps_shift=1,
        num_training=30e3,
        num_testing=10e3,
    ),
    model_class=Rand8x8_LR1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=MSE_ContrastV_Base(sigma=0.5),
)

shift1_rand_LR1_Noise1_MSE_ContrastV_Base = deepcopy(base)
shift1_rand_LR1_Noise1_MSE_ContrastV_Base.update(
    lr=1e-3,
    batch_size=768,
    dataset_class=ThreeRandomOneShift,
    dataset_args=dict(
        patch_size=(8, 8),
        eps_shift=1,
        num_training=30e3,
        num_testing=10e3,
    ),
    model_class=Rand8x8_LR1,
    model_args=dict(
        std=0.5,
    ),
    optimizer_class=optim.Adam,
    loss_fn=MSE_ContrastV_Base(sigma=0.5),
)

shift1_rand_LR1_Noise2_MSE_ContrastV_Base = deepcopy(base)
shift1_rand_LR1_Noise2_MSE_ContrastV_Base.update(
    lr=1e-3,
    batch_size=768,
    dataset_class=ThreeRandomOneShift,
    dataset_args=dict(
        patch_size=(8, 8),
        eps_shift=1,
        num_training=30e3,
        num_testing=10e3,
    ),
    model_class=Rand8x8_LR1,
    model_args=dict(
        std=1.0,
    ),
    optimizer_class=optim.Adam,
    loss_fn=MSE_ContrastV_Base(sigma=0.5),
)




shift1_rand_LR1_MSE_ContrastV1 = deepcopy(base)
shift1_rand_LR1_MSE_ContrastV1.update(
    lr=1e-3,
    batch_size=768,
    dataset_class=ThreeRandomOneShift,
    dataset_args=dict(
        patch_size=(8, 8),
        eps_shift=1,
        num_training=30e3,
        num_testing=10e3,
    ),
    model_class=Rand8x8_LR1,
    model_args=dict(
        std=0.0,
    ),
    optimizer_class=optim.Adam,
    loss_fn=MSE_ContrastV1(sigma=0.5),
)

shift1_rand_LR1_Noise1_MSE_ContrastV1 = deepcopy(base)
shift1_rand_LR1_Noise1_MSE_ContrastV1.update(
    lr=1e-3,
    batch_size=768,
    dataset_class=ThreeRandomOneShift,
    dataset_args=dict(
        patch_size=(8, 8),
        eps_shift=1,
        num_training=30e3,
        num_testing=10e3,
    ),
    model_class=Rand8x8_LR1,
    model_args=dict(
        std=0.5,
    ),
    optimizer_class=optim.Adam,
    loss_fn=MSE_ContrastV1(sigma=0.5),
)



shift1_rand_LR1_NoiseSweep1 = deepcopy(base)
shift1_rand_LR1_NoiseSweep1.update(
    lr=1e-3,
    batch_size=768,
    dataset_class=ThreeRandomOneShift,
    dataset_args=dict(
        patch_size=(8, 8),
        eps_shift=1,
        num_training=30e3,
        num_testing=10e3,
    ),
    model_class=Rand8x8_LR1,
    model_args=dict(
        std=0.1,
    ),
    optimizer_class=optim.Adam,
    loss_fn=MSE_ContrastV_Base(sigma=0.5),
)

shift1_rand_LR1_NoiseSweep2 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep2.update(model_args=dict(std=0.2))

shift1_rand_LR1_NoiseSweep3 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep3.update(model_args=dict(std=0.3))

shift1_rand_LR1_NoiseSweep4 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep4.update(model_args=dict(std=0.4))

shift1_rand_LR1_NoiseSweep5 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep5.update(model_args=dict(std=0.5))

shift1_rand_LR1_NoiseSweep6 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep6.update(model_args=dict(std=0.6))

shift1_rand_LR1_NoiseSweep7 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep7.update(model_args=dict(std=0.7))

shift1_rand_LR1_NoiseSweep8 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep8.update(model_args=dict(std=0.8))

shift1_rand_LR1_NoiseSweep9 = deepcopy(shift1_rand_LR1_NoiseSweep1)
shift1_rand_LR1_NoiseSweep9.update(model_args=dict(std=0.9))





# shift1_rand_LR1_MSE_ContrastV_OppV = deepcopy(base)
# shift1_rand_LR1_MSE_ContrastV_OppV.update(
#     lr=1e-3,
#     batch_size=768,
#     dataset_class=ThreeRandomPairs,
#     dataset_args=dict(
#         patch_size=(8, 8),
#         eps_shift=1,
#         num_training=30e3,
#         num_testing=10e3,
#     ),
#     model_class=Rand8x8_LR1,
#     model_args=dict(
#     ),
#     optimizer_class=optim.Adam,
#     loss_fn=MSE_ContrastV_OppV(sigma=0.5),
# )

# shift1_rand_LR1_MSE_ContrastV_OppZeroV = deepcopy(base)
# shift1_rand_LR1_MSE_ContrastV_OppZeroV.update(
#     lr=1e-3,
#     batch_size=768,
#     dataset_class=ThreeRandomPairs,
#     dataset_args=dict(
#         patch_size=(8, 8),
#         eps_shift=1,
#         num_training=30e3,
#         num_testing=10e3,
#     ),
#     model_class=Rand8x8_LR1,
#     model_args=dict(
#     ),
#     optimizer_class=optim.Adam,
#     loss_fn=MSE_ContrastV_OppZeroV(sigma=0.5),
# )

# shift1_rand_LR1_MSE_ContrastV_ZeroV = deepcopy(base)
# shift1_rand_LR1_MSE_ContrastV_ZeroV.update(
#     lr=1e-3,
#     batch_size=768,
#     dataset_class=ThreeRandomPairs,
#     dataset_args=dict(
#         patch_size=(8, 8),
#         eps_shift=1,
#         num_training=30e3,
#         num_testing=10e3,
#     ),
#     model_class=Rand8x8_LR1,
#     model_args=dict(
#     ),
#     optimizer_class=optim.Adam,
#     loss_fn=MSE_ContrastV_ZeroV(sigma=0.5),
# )

# shift1_rand_LR1_MSE_ContrastV = deepcopy(base)
# shift1_rand_LR1_MSE_ContrastV.update(
#     lr=1e-3,
#     batch_size=768,
#     dataset_class=ThreeRandomPairs,
#     dataset_args=dict(
#         patch_size=(8, 8),
#         eps_shift=1,
#         num_training=30e3,
#         num_testing=10e3,
#     ),
#     model_class=Rand8x8_LR1,
#     model_args=dict(
#     ),
#     optimizer_class=optim.Adam,
#     loss_fn=MSE_ContrastV(sigma=0.5),
# )







RAND_CONFIGS = dict(
    # shift1_rand_LR1_mse=shift1_rand_LR1_mse,
    # shift1_rand_LR1_croppedmse=shift1_rand_LR1_croppedmse,

    shift1_rand_LR1_MSE_ContrastV_Base=shift1_rand_LR1_MSE_ContrastV_Base,
    shift1_rand_LR1_Noise1_MSE_ContrastV_Base=shift1_rand_LR1_Noise1_MSE_ContrastV_Base,
    shift1_rand_LR1_Noise2_MSE_ContrastV_Base=shift1_rand_LR1_Noise2_MSE_ContrastV_Base,
    

    shift1_rand_LR1_MSE_ContrastV1=shift1_rand_LR1_MSE_ContrastV1,
    shift1_rand_LR1_Noise1_MSE_ContrastV1=shift1_rand_LR1_Noise1_MSE_ContrastV1,
    

    shift1_rand_LR1_NoiseSweep1=shift1_rand_LR1_NoiseSweep1,
    shift1_rand_LR1_NoiseSweep2=shift1_rand_LR1_NoiseSweep2,
    shift1_rand_LR1_NoiseSweep3=shift1_rand_LR1_NoiseSweep3,
    shift1_rand_LR1_NoiseSweep4=shift1_rand_LR1_NoiseSweep4,
    shift1_rand_LR1_NoiseSweep5=shift1_rand_LR1_NoiseSweep5,
    shift1_rand_LR1_NoiseSweep6=shift1_rand_LR1_NoiseSweep6,
    shift1_rand_LR1_NoiseSweep7=shift1_rand_LR1_NoiseSweep7,
    shift1_rand_LR1_NoiseSweep8=shift1_rand_LR1_NoiseSweep8,
    shift1_rand_LR1_NoiseSweep9=shift1_rand_LR1_NoiseSweep9,


    # shift1_rand_LR1_MSE_ContrastV_OppV=shift1_rand_LR1_MSE_ContrastV_OppV,
    # shift1_rand_LR1_MSE_ContrastV_OppZeroV=shift1_rand_LR1_MSE_ContrastV_OppZeroV,
    # shift1_rand_LR1_MSE_ContrastV_ZeroV=shift1_rand_LR1_MSE_ContrastV_ZeroV,
    # shift1_rand_LR1_MSE_ContrastV=shift1_rand_LR1_MSE_ContrastV,
)