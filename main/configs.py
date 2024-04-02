from imports import *
from data import *
from models import *
from utils import *


#---------------------------------------


shrinkingblob = dict(
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
        std_init_range=(0.05, 0.5), max_std_shift=0.08,
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
        num_training=20e3, num_testing=2e3,
    ),
    dataloader_args=dict(
        batch_size=256,
        pin_memory=True,
        num_workers=12,
    )
)


#---------------------------------------


stretchybird2d = dict(
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
        num_training=15e3, num_testing=3e3,
    ),
    dataloader_args=dict(
        batch_size=192,
        pin_memory=True,
        num_workers=12,
    )
)


#---------------------------------------


stretchybird3d = dict(
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
        num_training=15e3, num_testing=3e3,
    ),
    dataloader_args=dict(
        batch_size=192,
        pin_memory=True,
        num_workers=12,
    )
)

    
#---------------------------------------


frequencyshift1d = dict(
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-4,
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-4,
)


blob_ae = deepcopy(shrinkingblob)
blob_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=baseline_loss_fn,
)
blob_ae['model_args'].update(
    input_shape=(16, 16),
    layer_sizes=(256, 128, 64, 32, 16, 8, 4, 2),
)
blob_ae['loss_fn_args'].update(
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-2,
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-2,
)


gauss_blobs_ae = deepcopy(contshiftingmeans)
gauss_blobs_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=baseline_loss_fn,
)
gauss_blobs_ae['model_args'].update(
    input_shape=(16, 16),
    layer_sizes=(256, 128, 64, 32, 16, 8, 4, 2),
)
gauss_blobs_ae['loss_fn_args'].update(
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
    image_loss_weight=1e1, loop_loss_weight=1e2, shortcut_loss_weight=1e1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=6e-3,
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
    image_loss_weight=1e1, loop_loss_weight=1e2, shortcut_loss_weight=1e1, isometry_loss_weight=1e2, 
    isometry_similarity_threshold=6e-3,
)


bird2d_ae = deepcopy(stretchybird2d)
bird2d_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=baseline_loss_fn,
)
bird2d_ae['model_args'].update(
    input_shape=(32, 12),
    layer_sizes=(384, 192, 96, 48, 24, 12, 6, 2),
)
bird2d_ae['loss_fn_args'].update(
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
bird3d['dataloader_args'].update(
    batch_size=192,
)
bird3d['model_args'].update(
    v_dim=3,
    num_output_labels=200,
)
bird3d['dataset_args'].update(
    path='./',
    random_walk_length=20,
    max_std_shift=1.5,
    num_training=15e3, num_testing=3e3,
)
bird3d['loss_fn_args'].update(
    image_loss_weight=1e1, loop_loss_weight=1e2, shortcut_loss_weight=1e1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=6e-3,
)


bird3d_ae = deepcopy(stretchybird3d)
bird3d_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=baseline_loss_fn,
)
bird3d_ae['model_args'].update(
    input_shape=(32, 12),
    layer_sizes=(384, 192, 96, 48, 24, 12, 6, 3),
)
bird3d_ae['loss_fn_args'].update(
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-4,
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-4,
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
    image_loss_weight=1, loop_loss_weight=1e1, shortcut_loss_weight=1, isometry_loss_weight=1e2,
    isometry_similarity_threshold=1e-4,
)


freq_ae = deepcopy(frequencyshift1d)
freq_ae.update(
    model_class=Autoencoder,
    lr=1e-3,
    num_epochs=800,
    loss_fn=baseline_loss_fn,
)
freq_ae['model_args'].update(
    input_shape=(1, 100),
    layer_sizes=(100, 80, 60, 40, 20, 10, 5, 2),
)
freq_ae['loss_fn_args'].update(
    image_loss_weight=1e3,
)


#------------------------------------------------------------------------------------------------------------


CONFIGS = dict(
    blob=blob,
    blob_3dim=blob_3dim,
    blob_ae=blob_ae,

    gauss_blobs=gauss_blobs,
    gauss_blobs_3dim=gauss_blobs_3dim,
    gauss_blobs_ae=gauss_blobs_ae,

    bird2d=bird2d,
    bird2d_3dim=bird2d_3dim,
    bird2d_ae=bird2d_ae,

    bird3d=bird3d,
    bird3d_ae=bird3d_ae,

    freq=freq,
    freq_2dim=freq_2dim,
    freq_3dim=freq_3dim,
    freq_ae=freq_ae,
)


#------------------------------------------------------------------------------------------------------------


EXP_CODES = dict(
    freq=['z528sp0x', 'dh5950bh', '422dsi5h', 'cj8x2eoo', '8z118ppg', 'tnhupber'],
    freq_2dim=['70yf9dxl', 'yajgq80l', '5m5okdv0', '261ihs6j', 'kigt2pvn', 'vlvcjjlh'],
    freq_3dim=[],
    freq_ae=['jjotij2g'],
    freq_mcnet=[],

    blob=['x1axbbz9', 'lhdeijjs', 'u97nrn1r', '6cbtkrec', 'xmrrutxk', '8d0k2dwo'],
    blob_3dim=[],
    blob_ae=['mwn6e19n'],
    blob_mcnet=[],

    gauss_blobs=['vg2es21d', 'slj7afhk', '2jc1ffd7', 'q81c8b20', 'pd60xufu', 'aq61yki5'],
    gauss_blobs_3dim=['qxjymcmk', 'tkhky706', 'gvb7vgly', 'eq4vnb86', 'qxpvetjm', 'npfcju5f'],
    gauss_blobs_ae=['wbg6v2on'],
    gauss_blobs_mcnet=[],

    bird2d=['kecqzhj6', '033kma5k', '4dab3zb9', '7eqag39e', 'bi9emre0', '289qlq2f'],
    bird2d_3dim=[],
    bird2d_ae=['fwid48ml'],
    bird2d_mcnet=[],

    bird3d=['7tpqq4n6', 'b942twxh', '307rqww5', 'l9354lc8', 's9rnn0aw', 'ecwyvubc'],
    bird3d_ae=['mlwxi3u4'],
    bird3d_mcnet=[],
)