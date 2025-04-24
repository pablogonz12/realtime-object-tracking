
_base_ = './rtmdet_l_8xb32-300e_coco.py' # Inherit from a base RTMDet config

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=1.33,
        widen_factor=1.25,
        init_cfg=dict(type='Load', checkpoint=checkpoint, prefix='backbone.')),
    neck=dict(
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        in_channels=320,
        feat_channels=320,
        # Add mask head configuration for instance segmentation
        mask_head=dict(
            type='RTMDetInsHead',
            num_classes=80,
            in_channels=320,
            norm_cfg=dict(type='SyncBN'),
            loss_mask=dict(
                type='DiceLoss', loss_weight=2.0, eps=1e-6, reduction='mean'))
            ),
    # Enable mask output in test_cfg
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5 # Threshold for mask binarization
        ))

# dataset settings - needed for config structure but not critical for inference
train_dataloader = dict(batch_size=32, num_workers=10)
val_dataloader = dict(batch_size=5, num_workers=10)

# Modify learning rate schedule (less relevant for inference)
max_epochs = 300
base_lr = 0.004
interval = 10
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(optimizer=dict(lr=base_lr))
