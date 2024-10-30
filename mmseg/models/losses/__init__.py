# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

## added by LYU: 2022/04/06
from .gan_loss import GANLoss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'GANLoss'
]
