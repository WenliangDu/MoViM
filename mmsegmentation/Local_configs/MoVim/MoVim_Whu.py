# model settings
_base_ = [
    '../../configs/_base_/datasets/whu_cat.py',
    '../../configs/_base_/default_runtime.py', 
    '../../configs/_base_/schedules/schedule_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size=(512,512)
data_preprocessor = dict(size=crop_size,
                         type='SegDataPreProcessor',
                         mean=[104.74, 96.77, 79.21, 100.92, 54.12],  
                         std=[171.04, 217.80, 305.58, 697.17, 2169.97],  
                         bgr_to_rgb=False,
                        )


model_cfgs = dict(
    channels=[32, 64, 128, 160],
    out_channels=[None, 256,256,256],
    decode_out_indices=[1, 2, 3],
)


model = dict(
    type='EncoderDecoder',
    data_preprocessor = data_preprocessor,
    backbone=dict(
        type='MoVim',
        channels=model_cfgs['channels'],
        out_channels=model_cfgs['out_channels'], 
        decode_out_indices=model_cfgs['decode_out_indices'],
        in_channels = 5,
        depths=4,
        key_dim=16,
        num_heads=8,
        attn_ratios=2,
        mlp_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256,256,256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

find_unused_parameters=True
randomness = dict(seed=42)