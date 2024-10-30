_base_ = [
    '../../../../configs/_base_/datasets/pv_daseg.py', '../../../../configs/_base_/default_runtime.py',
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_forMDAAF_211_try_11_mask4_gan_fd_both2_ms_dr_nmk_is_nw2',
    pretrained=None,
    dsk_neck=dict(
        type='Rec_try_1',
        in_channels=64),
	dsk_neck1=dict(
        type='Rec_try_1',
        in_channels=64),
    backbone_s=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='/workspace/gpu8/my/work/ST-DASegNet/pretrained/mit_b5.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head_s=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])),
    decode_head_st=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])),
    decode_head_ts=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])),
    discriminator_s=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='BN'),
        in_channels=4),
    discriminator_t=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='BN'),
        in_channels=64),
	cross_EMA = dict(
        ## two types: 'single_t', 'decoder_only_t'
        type='single_t',
        #type='single_t',
        training_ratio=0.25,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 1.01, 1.51, 1.51, 2.01, 2.01],
        backbone_EMA=dict(
            type='MixVisionTransformer',
            init_cfg=dict(type='Pretrained', checkpoint='/pretrained/mit_b5.pth'),
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        decode_head_EMA=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5]))
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

data = dict(samples_per_gpu=8, workers_per_gpu=1)

work_dir = ''

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

total_iters = 20000
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True,save_best='mIoU')

# optimizer setting
optimizer = dict(
    backbone_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    dsk_neck=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01),
    dsk_neck1=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01),
    decode_head_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),    
    decode_head_st=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head_ts=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    discriminator_s=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)),
    discriminator_t=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99))	)

runner = None
#use_ddp_wrapper = True
find_unused_parameters = True
