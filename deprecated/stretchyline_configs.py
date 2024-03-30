from imports import *
from models import *
from data import *
from utils import *


base = dict()


shift5_unfixedshift_LR1_mse = deepcopy(base)
shift5_unfixedshift_LR1_mse.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=False,
    ),
    model_class=StretchyLine_LR1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)

shift5_fixedshift_LR1_mse = deepcopy(base)
shift5_fixedshift_LR1_mse.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=True,
    ),
    model_class=StretchyLine_LR1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)

shift5_unfixedshift_A1_mse = deepcopy(base)
shift5_unfixedshift_A1_mse.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=False,
    ),
    model_class=StretchyLine_A1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)

shift5_fixedshift_A1_mse = deepcopy(base)
shift5_fixedshift_A1_mse.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=True,
    ),
    model_class=StretchyLine_A1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)



shift5_unfixedshift_LR1_ssim = deepcopy(base)
shift5_unfixedshift_LR1_ssim.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=False,
    ),
    model_class=StretchyLine_LR1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)

shift5_fixedshift_LR1_ssim = deepcopy(base)
shift5_fixedshift_LR1_ssim.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=True,
    ),
    model_class=StretchyLine_LR1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)

shift5_unfixedshift_A1_ssim = deepcopy(base)
shift5_unfixedshift_A1_ssim.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=False,
    ),
    model_class=StretchyLine_A1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)

shift5_fixedshift_A1_ssim = deepcopy(base)
shift5_fixedshift_A1_ssim.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=True,
    ),
    model_class=StretchyLine_A1,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


##### with A2 ######

shift5_unfixedshift_A2_mse = deepcopy(base)
shift5_unfixedshift_A2_mse.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=False,
    ),
    model_class=StretchyLine_A2,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)


shift5_fixedshift_A2_mse = deepcopy(base)
shift5_fixedshift_A2_mse.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=True,
    ),
    model_class=StretchyLine_A2,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=nn.MSELoss(),
)


shift5_unfixedshift_A2_ssim = deepcopy(base)
shift5_unfixedshift_A2_ssim.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=False,
    ),
    model_class=StretchyLine_A2,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


shift5_fixedshift_A2_ssim = deepcopy(base)
shift5_fixedshift_A2_ssim.update(
    lr=1e-3,
    batch_size=256,
    dataset_class=StretchyVertical,
    dataset_args=dict(
        image_size=(64, 64),
        eps_shift=5,
        fixed_shifts=True,
    ),
    model_class=StretchyLine_A2,
    model_args=dict(
    ),
    optimizer_class=optim.Adam,
    loss_fn=SSIM_Loss(data_range=1.0, channel=1),
)


STRETCHYLINE_CONFIGS = dict(
    shift5_unfixedshift_LR1_mse=shift5_unfixedshift_LR1_mse,
    shift5_fixedshift_LR1_mse=shift5_fixedshift_LR1_mse,
    shift5_unfixedshift_A1_mse=shift5_unfixedshift_A1_mse,
    shift5_fixedshift_A1_mse=shift5_fixedshift_A1_mse,
    shift5_unfixedshift_LR1_ssim=shift5_unfixedshift_LR1_ssim,
    shift5_fixedshift_LR1_ssim=shift5_fixedshift_LR1_ssim,
    shift5_unfixedshift_A1_ssim=shift5_unfixedshift_A1_ssim,
    shift5_fixedshift_A1_ssim=shift5_fixedshift_A1_ssim,
    
    
    
    shift5_unfixedshift_A2_mse=shift5_unfixedshift_A2_mse,
    shift5_fixedshift_A2_mse=shift5_fixedshift_A2_mse,
    shift5_unfixedshift_A2_ssim=shift5_unfixedshift_A2_ssim,
    shift5_fixedshift_A2_ssim=shift5_fixedshift_A2_ssim,
)