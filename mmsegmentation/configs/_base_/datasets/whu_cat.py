# dataset settings
dataset_type = 'WhuDataset'
data_root = '../../../../datasets/whu-512'
crop_size = (512,512)
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations', reduce_zero_label=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),  # 自定义加载Tiff图像
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadTiffAnnotations', reduce_zero_label=True),

    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True)
             for r in img_ratios
             ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], 
            [dict(type='LoadTiffAnnotations')], 
            [dict(type='PackSegInputs')]
        ])
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
        data_prefix=dict(
            img_path='merge/train', seg_map_path='mask/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='merge/val',
            seg_map_path='mask/val'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='merge/val',
            seg_map_path='mask/val'),
        pipeline=test_pipeline)

)

val_evaluator = [dict(type='IoUMetric', iou_metrics=['mIoU']),
                 dict(type='KappaMetric')  # 添加Kappa指标
                ]
test_evaluator = val_evaluator
