from imports import *
from data import *
from models import *
from utils import *


#---------------------------------------


shrinkingblob = dict(
    task_dim=2,
    lr=5e-4,
    num_epochs=800,
    model_class=None,
    model_args=dict(),
    optimizer_class=optim.Adam,
    loss_fn=None,
    loss_fn_args=dict(),
    dataset_class=ShrinkingBlob,
    dataset_args=dict(
        path='./',
        patch_size=(16, 16),
        random_walk_length=20,
        std_init_range=(0.05, 0.6), max_std_shift=0.08,
        num_training=20e3, num_testing=5e3,
    ),
    dataloader_args=dict(
        batch_size=256,
        pin_memory=True,
        num_workers=12,
    )
)


#---------------------------------------


contshiftingmeans = dict(
    task_dim=2,
    lr=5e-4,
    num_epochs=800,
    model_class=None,
    model_args=dict(),
    optimizer_class=optim.Adam,
    loss_fn=None,
    loss_fn_args=dict(),
    dataset_class=ContinuousShiftingMeansLoops,
    dataset_args=dict(
        path='./',
        patch_size=(16, 16),
        random_walk_length=20,
        mean_init_range=(-20.0, 20.0), std_init_range=(0.5, 2.0), max_mean_shift=1.0,
        num_training=20e3, num_testing=5e3,
    ),
    dataloader_args=dict(
        batch_size=256,
        pin_memory=True,
        num_workers=12,
    )
)


#---------------------------------------


stretchybird2d = dict(
    task_dim=2,
    lr=5e-4,
    num_epochs=800,
    model_class=None,
    model_args=dict(),
    optimizer_class=optim.Adam,
    loss_fn=None,
    loss_fn_args=dict(),
    dataset_class=StretchyBird2D,
    dataset_args=dict(
        path='./',
        random_walk_length=20,
        max_std_shift=1.5,
        num_training=20e3, num_testing=5e3,
    ),
    dataloader_args=dict(
        batch_size=192,
        pin_memory=True,
        num_workers=12,
    )
)


#---------------------------------------


stretchybird3d = dict(
    task_dim=3,
    lr=5e-4,
    num_epochs=800,
    model_class=None,
    model_args=dict(),
    optimizer_class=optim.Adam,
    loss_fn=None,
    loss_fn_args=dict(),
    dataset_class=StretchyBird3D,
    dataset_args=dict(
        path='./',
        random_walk_length=20,
        max_std_shift=1.5,
        num_training=20e3, num_testing=5e3,
    ),
    dataloader_args=dict(
        batch_size=192,
        pin_memory=True,
        num_workers=12,
    )
)

    
#---------------------------------------


frequencyshift1d = dict(
    task_dim=1,
    lr=1e-4,
    num_epochs=1200,
    model_class=None,
    model_args=dict(),
    optimizer_class=optim.Adam,
    loss_fn=None,
    loss_fn_args=dict(),
    dataset_class=FrequencyShift1D,
    dataset_args=dict(
        path='./',
        random_walk_length=20,
        init_range=(1e-1, 1e1),
        max_shift=0.05,
        num_training=20e3, num_testing=5e3,
    ),
    dataloader_args=dict(
        batch_size=256,
        pin_memory=True,
        num_workers=12,
    )
)


#------------------------------------------------------------------------------------------------------------


blob = deepcopy(shrinkingblob)
blob.update(
    model_class=Blob_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
blob['model_args'].update(
    v_dim=2,
    num_output_labels=200,
)
blob['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-4,
    spatial_locality_loss=0.0,
)


blob_3dim = deepcopy(shrinkingblob)
blob_3dim.update(
    model_class=Blob_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
blob_3dim['model_args'].update(
    v_dim=3,
    num_output_labels=200,
) 
blob_3dim['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-4,
    spatial_locality_loss=1,
)


blob_ae = deepcopy(shrinkingblob)
blob_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=ae_loss_fn,
)
blob_ae['model_args'].update(
    input_shape=(16, 16),
    layer_sizes=(256, 128, 64, 32, 16, 8, 4, 2),
    v_dim=2,
)
blob_ae['loss_fn_args'].update(
    image_loss_weight=1e3,
)


blob_mcnet = deepcopy(shrinkingblob)
blob_mcnet.update(
    model_class=MCNet_16x16,
    lr=1e-3,
    num_epochs=800,
    loss_fn=mcnet_loss_fn,
)
blob_mcnet['model_args'].update(
    v_dim=2,
)
blob_mcnet['loss_fn_args'].update(
    image_loss_weight=1e3,
)


#---------------------------------------


gauss_blobs = deepcopy(contshiftingmeans)
gauss_blobs.update(
    model_class=Gauss_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
gauss_blobs['model_args'].update(
    v_dim=2,
    num_output_labels=200,
)
gauss_blobs['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-2,
    spatial_locality_loss=0.0,
)


gauss_blobs_3dim = deepcopy(contshiftingmeans)
gauss_blobs_3dim.update(
    model_class=Gauss_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
gauss_blobs_3dim['model_args'].update(
    v_dim=3,
    num_output_labels=200,
)
gauss_blobs_3dim['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1,
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-2,
    spatial_locality_loss=0.0,
)


gauss_blobs_ae = deepcopy(contshiftingmeans)
gauss_blobs_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=ae_loss_fn,
)
gauss_blobs_ae['model_args'].update(
    input_shape=(16, 16),
    layer_sizes=(256, 128, 64, 32, 16, 8, 4, 2),
    v_dim=2,
)
gauss_blobs_ae['loss_fn_args'].update(
    image_loss_weight=1e3,
)


gauss_blobs_mcnet = deepcopy(contshiftingmeans)
gauss_blobs_mcnet.update(
    model_class=MCNet_16x16,
    lr=1e-3,
    num_epochs=800,
    loss_fn=mcnet_loss_fn,
)
gauss_blobs_mcnet['model_args'].update(
    v_dim=2,
)
gauss_blobs_mcnet['loss_fn_args'].update(
    image_loss_weight=1e3,
)


#---------------------------------------


bird2d = deepcopy(stretchybird2d)
bird2d.update(
    model_class=Bird_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
bird2d['model_args'].update(
    v_dim=2,
    num_output_labels=200,
)
bird2d['loss_fn_args'].update(
    image_loss_weight=1e1, loop_loss_weight=1e2, shortcut_loss_weight=1e1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=6e-3,
    spatial_locality_loss=0.0,
)


bird2d_3dim = deepcopy(stretchybird2d)
bird2d_3dim.update(
    model_class=Bird_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
bird2d_3dim['model_args'].update(
    v_dim=3,
    num_output_labels=200,
)
bird2d_3dim['loss_fn_args'].update(
    image_loss_weight=1e1, loop_loss_weight=1e2, shortcut_loss_weight=1e1,
    isometry_loss_weight=1e2, isometry_similarity_threshold=6e-3,
    spatial_locality_loss=1e1,
)


bird2d_ae = deepcopy(stretchybird2d)
bird2d_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=ae_loss_fn,
)
bird2d_ae['model_args'].update(
    input_shape=(32, 12),
    layer_sizes=(384, 192, 96, 48, 24, 12, 6, 2),
    v_dim=2,
)
bird2d_ae['loss_fn_args'].update(
    image_loss_weight=1e3,
)


bird2d_mcnet = deepcopy(stretchybird2d)
bird2d_mcnet.update(
    model_class=MCNet_32x12,
    lr=1e-3,
    num_epochs=800,
    loss_fn=mcnet_loss_fn,
)
bird2d_mcnet['model_args'].update(
    v_dim=2,
)
bird2d_mcnet['loss_fn_args'].update(
    image_loss_weight=1e3,
)


#---------------------------------------


bird3d = deepcopy(stretchybird3d)
bird3d.update(
    model_class=Bird_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
bird3d['model_args'].update(
    v_dim=3,
    num_output_labels=200,
)
bird3d['loss_fn_args'].update(
    image_loss_weight=1e1, loop_loss_weight=1e2, shortcut_loss_weight=1e1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=6e-3,
    spatial_locality_loss=0.0,
)


bird3d_ae = deepcopy(stretchybird3d)
bird3d_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=ae_loss_fn,
)
bird3d_ae['model_args'].update(
    input_shape=(32, 12),
    layer_sizes=(384, 192, 96, 48, 24, 12, 6, 3),
    v_dim=3,
)
bird3d_ae['loss_fn_args'].update(
    image_loss_weight=1e3,
)


bird3d_mcnet = deepcopy(stretchybird3d)
bird3d_mcnet.update(
    model_class=MCNet_32x12,
    lr=1e-3,
    num_epochs=800,
    loss_fn=mcnet_loss_fn,
)
bird3d_mcnet['model_args'].update(
    v_dim=3,
)
bird3d_mcnet['loss_fn_args'].update(
    image_loss_weight=1e3,
)


#---------------------------------------


freq = deepcopy(frequencyshift1d)
freq.update(
    model_class=Frequency_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
freq['model_args'].update(
    v_dim=1,
    num_output_labels=200,
)
freq['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-4,
    spatial_locality_loss=0.0,
)


freq_2dim = deepcopy(frequencyshift1d)
freq_2dim.update(
    model_class=Frequency_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
freq_2dim['model_args'].update(
    v_dim=2,
    num_output_labels=200,
)
freq_2dim['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-4,
    spatial_locality_loss=0.0,
)


freq_3dim = deepcopy(frequencyshift1d)
freq_3dim.update(
    model_class=Frequency_FC_FC,
    lr=5e-4,
    num_epochs=800,
    loss_fn=general_loss_fn,
)
freq_3dim['model_args'].update(
    v_dim=3,
    num_output_labels=200,
)
freq_3dim['loss_fn_args'].update(
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, 
    isometry_loss_weight=1e2, isometry_similarity_threshold=1e-4,
    spatial_locality_loss=0.0,
)


freq_ae = deepcopy(frequencyshift1d)
freq_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=ae_loss_fn,
)
freq_ae['model_args'].update(
    input_shape=(1, 100),
    layer_sizes=(100, 80, 60, 40, 20, 10, 5, 1),
    v_dim=1,
)
freq_ae['loss_fn_args'].update(
    image_loss_weight=1e3,
)


freq_mcnet = deepcopy(frequencyshift1d)
freq_mcnet.update(
    model_class=MCNet_1x100,
    lr=1e-3,
    num_epochs=800,
    loss_fn=mcnet_loss_fn,
)
freq_mcnet['model_args'].update(
    v_dim=1,
)
freq_mcnet['loss_fn_args'].update(
    image_loss_weight=1e3,
)


#------------------------------------------------------------------------------------------------------------


CONFIGS = dict(
    blob=blob,
    blob_3dim=blob_3dim,
    blob_ae=blob_ae,
    blob_mcnet=blob_mcnet,

    gauss_blobs=gauss_blobs,
    gauss_blobs_3dim=gauss_blobs_3dim,
    gauss_blobs_ae=gauss_blobs_ae,
    gauss_blobs_mcnet=gauss_blobs_mcnet,

    bird2d=bird2d,
    bird2d_3dim=bird2d_3dim,
    bird2d_ae=bird2d_ae,
    bird2d_mcnet=bird2d_mcnet,

    bird3d=bird3d,
    bird3d_ae=bird3d_ae,
    bird3d_mcnet=bird3d_mcnet,

    freq=freq,
    freq_2dim=freq_2dim,
    freq_3dim=freq_3dim,
    freq_ae=freq_ae,
    freq_mcnet=freq_mcnet,
)


#------------------------------------------------------------------------------------------------------------


EXP_CODES = dict(
    freq=['z528sp0x', 'dh5950bh', '422dsi5h', 'cj8x2eoo', '8z118ppg', 'tnhupber'],
    freq_2dim=['70yf9dxl', 'yajgq80l', '5m5okdv0', '261ihs6j', 'kigt2pvn', 'vlvcjjlh'],
    freq_3dim=['w0z5rowu', 'nrak4phj', 'z296devh', 'u0tp2unu', 't7cmkkru', 'cecuup8u'],
    freq_ae=['jrkp1opx', 'b187kcxw', '9l5tzvgx', 'r6znjmpv', '2wpo2rgr', 'j5dt5bdm'],
    freq_mcnet=['1lwd02te', 'wk3mafka', '10soeyja', 'v4nckzde', 'fsyqscga', '8hzhx9xg'],

    blob=['lrf29umi', 'iy4ekrmv', '3nhdswr0', 'b434tgs6', 'ev40t32f', 'bzjch81e'],
    blob_3dim=['785o0lfi', '8d3yb4ty', 'x8kauyhn', 'nczimes7', '5p3wuqja', 'y1rf1osf'],
    blob_ae=['qf6jy4p4', 'phvw3mma', '9i6957v9', 'syrwzfwy', 'i4pu66v4', 'xdlloemp'],
    blob_mcnet=['rjtkvfjz', 'wxawvpn9', 'iohysj37', '05xbakxc', 'ktmpsew7', '3h3hihcw'],

    gauss_blobs=['vg2es21d', 'slj7afhk', '2jc1ffd7', 'q81c8b20', 'pd60xufu', 'aq61yki5'],
    gauss_blobs_3dim=['qxjymcmk', 'tkhky706', 'gvb7vgly', 'eq4vnb86', 'qxpvetjm', 'npfcju5f'],
    gauss_blobs_ae=['wbg6v2on', 'lodhhjsy', '4dr82kee', 'tugrv3xs', 'xgt6zjrz', 'yg2dyy2l'],
    gauss_blobs_mcnet=['0uubc3bq', 'ruimrjmf', '7erj3ehm', '5vw9o21p', 'h1x11wtr', '7cprunfj'],

    bird2d=['zrg1n81t', '2u17r582', '48tfp8ic', '5c3pyiry', 'hwsqxr2q', '68fu6zq5'],
    bird2d_3dim=['wkxw31o2', 'ibom3py6', '08ewagh1', '2m5edxjl', 'f08any5m', 'a584kajr'],
    bird2d_ae=['0n5qkwzw', 'l3an4sxg', 'pzl7qxh2', 'dl9uty2x', 'vfxa66hm', '9balclqg'],
    bird2d_mcnet=['krq3ykjw', '8aqcqw13', 'tdqnvca8', 's9xsnq5y', 'vm7arkdt', 'fpasw3ig'],

    bird3d=['vfi9uv12', 'uchpxrsi', 'osvsspoa', 'vk9imiwu', 'wdp0krcb', '360d94lh'],
    bird3d_ae=['x0j5bj52', 'sj7ir8kc', 'wiq4n405', 'siadb8h1', 'q2b3crp2', 'bd3kowxg'],
    bird3d_mcnet=['f4ff068v', '362ogfdp', 'cqgv7m2i', 'zc2c3mzw', 'w3g78iw5', 't31omm8g'],
)
