# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import kornia
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
#from copy import deepcopy
from PIL import Image
from skimage import io
#from mmcv.parallel import MMDistributedDataParallel
import numpy as np
import copy
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import  MS_SSIM
ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
kl_loss = nn.KLDivLoss(reduction="batchmean")
sm = torch.nn.Softmax(dim = 1)
log_sm = torch.nn.LogSoftmax(dim = 1)
def dice_coefficient(predicted, target, smooth=1e-5):
    intersection = torch.sum(predicted * target, dim=(2, 3))
    union = torch.sum(predicted, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()
palette = np.asarray([[255, 255, 255], [255, 0, 0], [255, 255, 0],
                              [0, 0, 255], [159, 129, 183], [0, 255, 0],
                              [255, 195, 128]]).reshape((-1,)).tolist()
def dice_loss(predicted, target):
    return 1 - dice_coefficient(predicted, target)
def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
    
@torch.no_grad()
def generate_mask( imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / 64), round(
            W / 64)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > 0.6).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

@torch.no_grad()
def mask_image( imgs):
        input_mask = generate_mask(imgs)
        return imgs * input_mask,input_mask
def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)     # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
    amp_src[:, :, 0:b, w - b:w] = amp_trg[:, :, 0:b, w - b:w]    # top right
    amp_src[:, :, h - b:h, 0:b] = amp_trg[:, :, h - b:h, 0:b]    # bottom left
    amp_src[:, :, h - b:h, w - b:w] = amp_trg[:, :, h - b:h, w - b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.view_as_real(torch.fft.fft(src_img.clone(), dim=2))
    fft_trg = torch.view_as_real(torch.fft.fft(trg_img.clone(), dim=2))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft(torch.view_as_complex(fft_src_), n=fft_src_.shape[2], dim=2)

    return src_in_trg


@torch.no_grad()
def fourier_mix(src_images, tgt_images, L=0.1):
    """ Transfers style of style images to content images. Assumes input 
        is a PyTorch tensor with a batch dimension."""
    B, sC, sH, sW = src_images.shape
    B, tC, tH, tW = tgt_images.shape
    if (sH > tH) or (sW > tW):
        tgt_images = F.interpolate(tgt_images, size=(sH, sW), mode='bicubic')
    mixed_images = FDA_source_to_target(src_images, tgt_images, L=L)
    return mixed_images.to(src_images.device)
def get_rand_bbox(size, lam):

    # Get cutout size
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Sample location uniformly at random
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Clip
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, feature_batch1, feature_batch2):
        # Flatten the feature batches along BCHW dimensions
        B, C, H, W = feature_batch1.size()
        feature1 = feature_batch1.view(B, -1)  # Flatten BCHW to B x (C*H*W)
        feature2 = feature_batch2.view(B, -1)  # Flatten BCHW to B x (C*H*W)

        # Calculate cosine similarity between feature1 and feature2
        similarity = F.cosine_similarity(feature1, feature2, dim=1)

        # Minimize the negative of cosine similarity (maximize cosine similarity)
        loss = -torch.mean(similarity)

        return loss


@SEGMENTORS.register_module()
class EncoderDecoder_forMDAAF_211_try_11_mask4_gan_fd_both2_ms_dr_nmk_is_nw2(BaseSegmentor):
    """Encoder Decoder segmentors for ST-DASegNet.

    EncoderDecoder_forDSFN typically consists of two backbone, two decode_head. Here, we do not
    apply auxiliary_head, neck to simplify the implementation.

    Args:
        backbone_s: backbone for source.
        backbone_t: backbone for target.
        decode_head_s: decode_head for source
        decode_head_t: decode_head for target
        discriminator_s: discriminator for source and fake_source
        discriminator_t: discriminator for target and fake_target
    """

    def __init__(self,
                 backbone_s,
                 decode_head_s,
				 decode_head_st,
				 decode_head_ts,
                 discriminator_s=None,
                 discriminator_t=None,
                 dsk_neck=None,
                 dsk_neck1=None,
                 cross_EMA= None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder_forMDAAF_211_try_11_mask4_gan_fd_both2_ms_dr_nmk_is_nw2, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone_s.get('pretrained') is None, \
                'both backbone_s and segmentor set pretrained weight'
            #assert backbone_t.get('pretrained') is None, \
            #    'both backbone_t and segmentor set pretrained weight'
            backbone_s.pretrained = pretrained
            #backbone_t.pretrained = pretrained
        self.backbone_s = builder.build_backbone(backbone_s)
        #self.backbone_t = builder.build_backbone(backbone_t)
        self.visualize= 0      
        self.decode_head_s = self._init_decode_head(decode_head_s)
        self.decode_head_st = self._init_decode_head(decode_head_st)
        self.decode_head_ts = self._init_decode_head(decode_head_ts)
        #self.decode_head_t = self._init_decode_head(decode_head_t)
        self.num_classes = self.decode_head_s.num_classes
        self.align_corners = self.decode_head_s.align_corners
        #assert self.decode_head_s.num_classes == self.decode_head_t.num_classes, \
        #        'both decode_head_s and decode_head_t must have same num_classes'

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        ## added by 
        #self.discriminator_f = builder.build_discriminator(discriminator_f)
        #self.discriminator_c = builder.build_discriminator(discriminator_c)
        #self.discriminator_r = builder.build_discriminator(discriminator_r)
        ## added by 
        self.discriminator_s = builder.build_discriminator(discriminator_s)
        self.discriminator_t = builder.build_discriminator(discriminator_t)
        self.dsk_neck = builder.build_neck(dsk_neck)
        self.dsk_neck1 = builder.build_neck(dsk_neck1)

        ## added by 
        if cross_EMA is not None:
            self.cross_EMA = cross_EMA
            self._init_cross_EMA(self.cross_EMA)
        self._parse_train_cfg()
    
    ##############################
    ## added by 
    ## added for cross_EMA
    def get_ema_model(self):
        return get_module(self.ema_model)
    def _init_cross_EMA(self, cfg):
        self.cross_EMA_type = cfg['type']
        self.cross_EMA_alpha = cfg['decay']
        self.cross_EMA_training_ratio = cfg['training_ratio']
        self.cross_EMA_pseu_cls_weight = cfg['pseudo_class_weight']
        self.cross_EMA_pseu_thre = cfg['pseudo_threshold']
        self.cross_EMA_rare_pseu_thre = cfg['pseudo_rare_threshold']
        if self.cross_EMA_type == 'single_t':
            self.cross_EMA_backbone = builder.build_backbone(cfg['backbone_EMA'])
            self.cross_EMA_decoder = self._init_decode_head(cfg['decode_head_EMA'])

        #elif self.cross_EMA_type == 'decoder_only_t':
        #    self.cross_EMA_decoder_s = self._init_decode_head(cfg['decode_head_EMA'])
        #    self.cross_EMA_decoder_t = self._init_decode_head(cfg['decode_head_EMA'])
        #elif self.cross_EMA_type == 'whole':
        #    ## DEPRECATED, too much memory cost
        #    pass
        else:
            ## No cross_EMA
            pass
        
    def _update_cross_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.cross_EMA_alpha)
        if self.cross_EMA_type == 'single_t':
            ## 1. update target_backbone
            for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone_s.parameters()):
                ## For scalar params
                if not target_b.data.shape:
                    ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
                ## For tensor params
                else:
                    ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]

            ## 2. updata target_decoder
            for ema_d, target_d in zip(self.cross_EMA_decoder.parameters(), self.decode_head_s.parameters()):
                ## For scalar params
                if not target_d.data.shape:
                    ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
                ## For tensor params
                else:
                    ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]
        if self.cross_EMA_type == 'decoder_only_t':
            ## 1. updata EMA_source_decoder
            for ema_d_s, source_d in zip(self.cross_EMA_decoder_s.parameters(), self.decode_head_s.parameters()):
                ## For scalar params
                if not source_d.data.shape:
                    ema_d_s.data = alpha_t * ema_d_s.data + (1 - alpha_t) * source_d.data
                ## For tensor params
                else:
                    ema_d_s.data[:] = alpha_t * ema_d_s.data[:] + (1 - alpha_t) * source_d.data[:]
            ## 2. updata EMA_source_decoder
            for ema_d_t, target_d in zip(self.cross_EMA_decoder_t.parameters(), self.decode_head_t.parameters()):
                ## For scalar params
                if not target_d.data.shape:
                    ema_d_t.data = alpha_t * ema_d_t.data + (1 - alpha_t) * target_d.data
                ## For tensor params
                else:
                    ema_d_t.data[:] = alpha_t * ema_d_t.data[:] + (1 - alpha_t) * target_d.data[:]
    
    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        ##############################
        #### 1. vanilla pseudo label generation
        pred_softmax = torch.softmax(pred, dim=1)
        pseudo_prob, pseudo_label = torch.max(pred_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=dev)
        ##############################
        ##############################
        #### 2. class balancing strategy
        #### 2.1 change pseudo_weight and further set a threshold for rare class. E.g. For threshold over 0.8: 10x for car and clutter; 5x for 'low_vegetation' and 'tree'
        if self.cross_EMA_pseu_cls_weight is not None and self.cross_EMA_rare_pseu_thre is not None:
            ps_large_p_rare = pseudo_prob.ge(self.cross_EMA_rare_pseu_thre).long() == 1
            pseudo_weight = pseudo_weight * ps_large_p_rare
            pseudo_class_weight = copy.deepcopy(pseudo_label.float())
            for i in range(len(self.cross_EMA_pseu_cls_weight)):
                pseudo_class_weight[pseudo_class_weight == i] = self.cross_EMA_pseu_cls_weight[i]
            pseudo_weight = pseudo_class_weight * pseudo_weight
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5
        ##############################
        pseudo_label = pseudo_label[:, None, :, :]
        return pseudo_label, pseudo_weight

    def encode_decode_crossEMA(self, input=None, dev=None):
        ## option1: 'single_t': inference all cross_EMA_teacher including cross_EMA_backbone and cross_EMA_decoder

        ema_feature = self.forward_backbone(self.cross_EMA_backbone, input)  
        
        #print("inside ema", input.shape, ema_feature[0].shape,ema_feature[1].shape,ema_feature[2].shape,ema_feature[3].shape)


        ema_logit = self.forward_decode_head(self.cross_EMA_decoder, ema_feature)
            
        
        ## option2: 'decoder_only_t': inference including cross_EMA_decoder_s and cross_EMA_decoder_t

            
        return ema_logit
    
    ## CODE for cross_EMA
    ##############################

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

    ## modified by Avinash
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = builder.build_head(decode_head)
        return decode_head

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone_s(img)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ## 1. forward backbone
        #if isinstance(self.dsk_neck.in_channels, int):
        #    F_t2s = self.forward_backbone(self.backbone_s, img)[-1]
        #    F_t2t = self.forward_backbone(self.backbone_t, img)[-1]
        #else:
        F_t2s = self.forward_backbone(self.backbone_s, img)
        #    F_t2t = self.forward_backbone(self.backbone_t, img)
        ## 2. forward neck
        #F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t)
        
        ## 3. forward decode_head
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s)
        P_t2s1 = self.forward_decode_head(self.decode_head_ts, F_t2s)
        P_t2s2 = self.forward_decode_head(self.decode_head_st, F_t2s)
        #P_t2t = self.forward_decode_head(self.decode_head_t, F_t2t_dsk)
        out = P_t2s#(P_t2s +P_t2s1 +P_t2s2 )/ 3
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_s.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    ## added by 
    def forward_backbone(self, backbone, img):
        F_b = backbone(img)
        return F_b
    
    def forward_decode_head(self, decode_head, feature):
        Pred = decode_head(feature)
        return Pred
    
    def forward_discriminator(self, discriminator, seg_pred):
        dis_pred = discriminator(seg_pred)
        return dis_pred

    def forward_train(self, img, B_img):
        pass
        """Forward function for training."""

    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg
        

    
    ## added by 
    def _get_gan_loss(self, discriminator, pred, domain, target_is_real):
        losses = dict()
        losses[f'loss_gan_{domain}'] = discriminator.gan_loss(pred, target_is_real)
        loss_dis, log_vars_dis = self._parse_losses(losses)
        ## added by 
        ## auxiliary_ganloss: TBD
        return loss_dis, log_vars_dis
    
    ## added by 
    def _get_KD_loss(self, teacher, student, pred_name, T=3):
        losses = dict()
        losses[f'loss_KD_{pred_name}'] = self.KL_loss(teacher, student, T)
        loss_KD, log_vars_KD = self._parse_losses(losses)
        return loss_KD, log_vars_KD

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        The whole process including back propagation and 
        optimizer updating is also defined in this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        ## added by 
        # dirty walkround for not providing running status
        means, stds = get_mean_std(data_batch['img_metas'], dev=data_batch['img'].device)
        alpha = 0.7
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': 0.2,#self.color_jitter_s,
            'color_jitter_p': 0.2,#self.color_jitter_p,
            'blur': random.uniform(0, 1) ,#if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration 
        #visualize = curr_iter
        ## added by 
        ## CODE for cross_EMA
        if curr_iter > 0:
            self._update_cross_EMA(curr_iter)
        visualize = curr_iter#self.visualize
        ## 1. towards all optimizers, clear gradients
        optimizer['backbone_s'].zero_grad()
        #optimizer['backbone_t'].zero_grad()
        optimizer['decode_head_s'].zero_grad()
        optimizer['decode_head_st'].zero_grad()
        optimizer['decode_head_ts'].zero_grad()
        optimizer['discriminator_s'].zero_grad()
        optimizer['discriminator_t'].zero_grad()
        #optimizer['decode_head_t'].zero_grad()
        #optimizer['discriminator_f'].zero_grad()
        #optimizer['discriminator_c'].zero_grad()
        #optimizer['discriminator_r'].zero_grad()
        #optimizer['discriminator_t'].zero_grad()
        optimizer['dsk_neck'].zero_grad()
        optimizer['dsk_neck1'].zero_grad()

        self.set_requires_grad(self.backbone_s, False)
        #self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_st, False)
        self.set_requires_grad(self.decode_head_ts, False)
        #self.set_requires_grad(self.decode_head_t, False)
        #self.set_requires_grad(self.discriminator_f, False)
        #self.set_requires_grad(self.discriminator_c, False)
        #self.set_requires_grad(self.discriminator_r, False)
        #self.set_requires_grad(self.discriminator_t, False)
        self.set_requires_grad(self.dsk_neck, False)
        self.set_requires_grad(self.dsk_neck1, False)
        log_vars = dict()

        ## 1.1 forward backbone
        self.set_requires_grad(self.backbone_s, True)
        self.set_requires_grad(self.decode_head_s, True)
        self.set_requires_grad(self.decode_head_st, False)
        self.set_requires_grad(self.decode_head_ts, False)
        self.set_requires_grad(self.discriminator_s, False)
        self.set_requires_grad(self.discriminator_t, False)
        #self.set_requires_grad(self.backbone_t, True)
        #self.set_requires_grad(self.decode_head_t, True)
        F_s2s_all = self.forward_backbone(self.backbone_s, data_batch['img'])
        #F_s2t_all = self.forward_backbone(self.backbone_s, data_batch['B_img'])
        #F_s2s = F_s2s_all[-1]
        #F_s2t = F_s2t_all[-1]
        #print(F_s2s_all[0].shape,F_s2s_all[1].shape,F_s2s_all[2].shape,F_s2s_all[3].shape)
        #print("**************************************************************")
        #F_s2s_dsk, F_s2t_dsk = self.dsk_neck(F_s2s_all, F_s2t_all)
        #P_s2s = self.forward_decode_head(self.decode_head_s, F_s2s_all)
        #a, b,c,d = self.dsk_neck(F_s2s_all)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print(F_s2s_dsk[0].shape, F_s2s_dsk[1].shape,F_s2s_dsk[2].shape,F_s2s_dsk[3].shape )
        #print("%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^%%%%%%%%%%%%%%%%%%%%")
        #rec_s = self.dsk_neck(F_s2s_all[0])
        #rec_t = self.dsk_neck(F_s2t_all[0])
        P_s2s = self.forward_decode_head(self.decode_head_s, F_s2s_all)
        #P_s2t = self.forward_decode_head(self.decode_head_s, F_s2t_all)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%GGGGG%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print("==================",P_s2s.shape)
        #print("***********",F_s2s_all[0].shape, rec_s.shape, data_batch['img'].shape )
        #loss_rec_s = torch.nn.L1Loss()(rec_s, data_batch['img'])
        #loss_rec_t = torch.nn.L1Loss()(rec_t, data_batch['B_img'])
        
        loss_seg_s2s, log_vars_seg_s2s = self._get_segmentor_loss(self.decode_head_s, P_s2s, data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s2s)
        loss_seg = loss_seg_s2s #+ (loss_rec_s )*alpha
        
        loss_seg.backward()
        
        ema_logits = self.encode_decode_crossEMA(input=data_batch['B_img'], dev=data_batch['img'].device)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        
        #print("&&&&",ema_logits.shape,ema_softmax.shape)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=data_batch['img'].device)
        if self.cross_EMA_pseu_cls_weight is not None and self.cross_EMA_rare_pseu_thre is not None:
            ps_large_p_rare = pseudo_prob.ge(self.cross_EMA_rare_pseu_thre).long() == 1
            pseudo_weight = pseudo_weight * ps_large_p_rare
            pseudo_class_weight = copy.deepcopy(pseudo_label.float())
            for i in range(len(self.cross_EMA_pseu_cls_weight)):
                pseudo_class_weight[pseudo_class_weight == i] = self.cross_EMA_pseu_cls_weight[i]
            pseudo_weight = pseudo_class_weight * pseudo_weight
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5
        pseudo_label = pseudo_label[:, None, :, :]
        pseudo_weight =  resize(
            input=torch.unsqueeze(pseudo_weight,1),
            size=data_batch['gt_semantic_seg'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        pseudo_weight = torch.squeeze(pseudo_weight,1)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=data_batch['img'].device)
        '''loss_seg_t2s, log_vars_seg_t2s = self._get_segmentor_loss(self.decode_head_s, P_s2t, pseudo_label, gt_weight=pseudo_weight)
        log_vars.update(log_vars_seg_t2s)
        loss_seg = loss_seg + self.cross_EMA_training_ratio * loss_seg_t2s'''
        
        
        '''batch_size,_,_,_ = P_s2s.shape
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = data_batch['gt_semantic_seg']

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((data_batch['gt_semantic_seg'][i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)'''
        pseudo_label1 =  resize(
            input=pseudo_label.float(),
            size=data_batch['gt_semantic_seg'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
            
        #batch_size = 4
        batch_size,_,_,_ = P_s2s.shape
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
		
        mix_masks = get_class_masks(data_batch['gt_semantic_seg'])
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((data_batch['img'][i], data_batch['B_img'][i])),
                target=torch.stack((data_batch['gt_semantic_seg'][i], pseudo_label1[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        
        


        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl).long()
		
		
		
        for_mic_full = mixed_img#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic_full = self.forward_backbone(self.backbone_s, for_mic_full)
        mic_feat_full = self.forward_decode_head(self.decode_head_s, Fo_mic_full)
        #print(torch.unsqueeze(mask1[:,0,:,:],dim=1).shape,mic_feat.shape)#print(mask1[:,0,:,:].shape)
        #mask1_label= mask1#torch.cat((mask1,mask1,torch.unsqueeze(mask1[:,0,:,:],dim=1)),dim=1)

        #print(mask1.shape,mask1_label.shape,mask1_label_resize.shape)
        mic_loss_full, mic_log_vars_full = self._get_segmentor_loss(self.decode_head_s, mic_feat_full, mixed_lbl,pseudo_weight)
        #print(torch.unique(mic_feat_full))
		#print("###########", Fo_mic[0].type(),y_cutmix.type())
        #recf_mic_full = self.dsk_neck(Fo_mic_full[0])#,y_cutmix.float())
        #lossmic_rec_cm_full = torch.nn.L1Loss()(recf_mic_full, mixed_img)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mic_log_vars_full)
        
        
        
        '''for_mic1 = mask_image(for_mix)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic1 = self.forward_backbone(self.backbone_s, for_mic1)
        mic_feat1 = self.forward_decode_head(self.decode_head_s, Fo_mic1)
        mic_loss1, mic_log_vars1 = self._get_segmentor_loss(self.decode_head_s, mic_feat1, data_batch['gt_semantic_seg'])
        recf_mic1 = self.dsk_neck(Fo_mic1[0])#,data_batch['gt_semantic_seg'].float())
        lossmic_rec_cm1 = torch.nn.L1Loss()(recf_mic1, for_mix)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mic_log_vars1)'''
        
        
        #loss_seg_m =  (mix_loss + mixf_loss)/2  +  (mic_loss + mic_loss1)/2 + (loss_rec_cm*0.05+lossf_rec_cm*0.05)/2  + (lossmic_rec_cm1*0.05 +lossmic_rec_cm*0.05)/2
        loss_seg_m1_full =   mic_loss_full   #+ lossmic_rec_cm_full*alpha #+lossmic_rec_cm*alpha)/2
		
        loss_seg_m1_full.backward()
		##################################
		
		
        self.set_requires_grad(self.dsk_neck, True)
        self.set_requires_grad(self.dsk_neck1, True)
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_st, True)
        self.set_requires_grad(self.decode_head_ts, True)
		
        mixed_img_nw, mixed_lbl_nw = [None] * batch_size, [None] * batch_size
        mix_masks_nw = get_class_masks(data_batch['gt_semantic_seg'])
		
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks_nw[i]
            mixed_img_nw[i], mixed_lbl_nw[i] = strong_transform(
                strong_parameters,
                data=torch.stack((data_batch['img'][i], data_batch['B_img'][i])),
                target=torch.stack((data_batch['gt_semantic_seg'][i], pseudo_label1[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        
        


        mixed_img_nw = torch.cat(mixed_img_nw)
        mixed_lbl_nw = torch.cat(mixed_lbl_nw).long()
		
        mixed_1, mixed_2,mask_save = [None] * batch_size, [None] * batch_size, [None] * batch_size
        for i in range(batch_size):
            mixed_1[i] =  mixed_img_nw[i]*mix_masks_nw[i]
            mixed_2[i] =  mixed_img_nw[i]*(1 - mix_masks_nw[i])
        mixed_1 = torch.cat(mixed_1)
        mixed_2 = torch.cat(mixed_2) #.long()
        #for i in range(batch_size):
		
        mask_save = torch.cat(mix_masks_nw)
        #print(mask_save.shape)#, mix_masks_nw.shape)
		
        ###################################################
        pseudo_weight_both = pseudo_weight
        mixed_img_nw_both, mixed_lbl_nw_both = [None] * batch_size, [None] * batch_size
        #mix_masks_nw_both = get_class_masks(pseudo_label1)
        for i in range(batch_size):
            strong_parameters['mix'] = 1- mix_masks_nw[i]
            mixed_img_nw_both[i], mixed_lbl_nw_both[i] = strong_transform(
                strong_parameters,
                data=torch.stack(( data_batch['img'][i], data_batch['B_img'][i])),
                target=torch.stack(( data_batch['gt_semantic_seg'][i], pseudo_label1[i])))
            _,pseudo_weight_both[i] = strong_transform(
                strong_parameters,
                target=torch.stack(( gt_pixel_weight[i], pseudo_weight[i])))
        
        


        mixed_img_nw_both = torch.cat(mixed_img_nw_both)
        mixed_lbl_nw_both = torch.cat(mixed_lbl_nw_both).long()
		
		
        mixed_1_both, mixed_2_both = [None] * batch_size, [None] * batch_size
        for i in range(batch_size):
            mixed_1_both[i] =  mixed_img_nw_both[i]*(1-mix_masks_nw[i])
            mixed_2_both[i] =  mixed_img_nw_both[i]* mix_masks_nw[i]
        mixed_1_both = torch.cat(mixed_1_both)
        mixed_2_both = torch.cat(mixed_2_both) #.long()
        
        
        ######################################
        #mixed_2 mixed_1_both
		#mixed_1 = torch.cat(mixed_1 mixed_2_both,dim=1)
        #mixed_2 = torch.cat(mixed_2 mixed_1_both,dim=1)
		#mixed_lbl_nw = torch.cat(mixed_lbl_nw,mixed_lbl_nw_both,dim=1)
        
        
        #for_mic,mask1 = mask_image(mixed_1)#fourier_mix(data_batch['img'],data_batch['B_img'])
        #for_mic,mask1 = mask_image(mixed_1)#fourier_mix(data_batch['img'],data_batch['B_img'])
        for_mic=mixed_1#,mask1 = mask_image(mixed_1)
        Fo_mic = self.forward_backbone(self.backbone_s, for_mic)
        mic_feat = self.forward_decode_head(self.decode_head_st, Fo_mic)
        #print(torch.unsqueeze(mask1[:,0,:,:],dim=1).shape,mic_feat.shape)#print(mask1[:,0,:,:].shape)
        mic_feat_o = torch.unsqueeze(mic_feat.argmax(dim=1),1)
        recf_mic = self.dsk_neck(Fo_mic[0],mic_feat_o)
        mic_feat = resize(input=mic_feat,size=pseudo_label1.shape[2:],mode='bilinear',align_corners=self.align_corners)
        
        #mic_feat_l,mixed_lbl_nw_l,recf_mic_l = [None] * batch_size, [None] * batch_size, [None] * batch_size
        #print(mic_feat.shape, mix_masks_nw.shape)
        #mix_masks_nw_l = mix_masks_nw
        '''for i in range(batch_size):
            
            mx=torch.unsqueeze(mix_masks_nw[i],0)
            #print(mic_feat[i].shape,mic_feat.shape[2:],mx[i].shape,mx.shape)
            mix_masks_nw_l[i] = resize(input=mx[i].float(),size=mic_feat.shape[2:],mode='bilinear',align_corners=self.align_corners)
        '''

        #for i in range(batch_size):
        #    mic_feat_l[i] =  mic_feat[i]*(1 - mix_masks_nw_l[i]) 
        #    mixed_lbl_nw_l[i] =  mixed_lbl_nw[i]*(1 - mix_masks_nw_l[i]) 
        #    recf_mic_l[i] =  recf_mic[i]*(1 - mix_masks_nw[i])
        #mic_feat_l = torch.cat(mic_feat_l)
        #mixed_lbl_nw_l = torch.cat(mixed_lbl_nw_l) 
        #recf_mic_l = torch.cat(recf_mic_l) 
        mic_loss, mic_log_vars = self._get_segmentor_loss(self.decode_head_st, mic_feat, mixed_lbl_nw,pseudo_weight)
        #print("###########", Fo_mic[0].type(),y_cutmix.type())
        #,y_cutmix.float())
        lossmic_rec_cm =  nn.L1Loss(reduction='none')(recf_mic, mixed_2)
        lossmic_rec_cm =(lossmic_rec_cm *(1-mask_save).float()).sum()
        non_zero_elements = (1-mask_save).sum()
        lossmic_rec_cm = lossmic_rec_cm / non_zero_elements
        #print(lossmic_rec_cm.shape)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        #log_vars.update(mic_log_vars) 
        
        
        
        '''for_mic1 = mask_image(for_mix)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic1 = self.forward_backbone(self.backbone_s, for_mic1)
        mic_feat1 = self.forward_decode_head(self.decode_head_s, Fo_mic1)
        mic_loss1, mic_log_vars1 = self._get_segmentor_loss(self.decode_head_s, mic_feat1, data_batch['gt_semantic_seg'])
        recf_mic1 = self.dsk_neck(Fo_mic1[0])#,data_batch['gt_semantic_seg'].float())
        lossmic_rec_cm1 = torch.nn.L1Loss()(recf_mic1, for_mix)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mic_log_vars1)'''
        
        
        #loss_seg_m =  (mix_loss + mixf_loss)/2  +  (mic_loss + mic_loss1)/2 + (loss_rec_cm*0.05+lossf_rec_cm*0.05)/2  + (lossmic_rec_cm1*0.05 +lossmic_rec_cm*0.05)/2
        loss_seg_m1 =    mic_loss + lossmic_rec_cm*.3#*alpha #+lossmic_rec_cm*alpha)/2

        
        ##################Cycle####
        #rec_mx
        #recf_mx
        
        #for_cyc1,ma = mask_image(mixed_2)#*(1-mask1)#mask_image(recf_mic)#fourier_mix(data_batch['img'],data_batch['B_img'])
        for_cyc1=mixed_2
        Fo_cyc1 = self.forward_backbone(self.backbone_s, for_cyc1)
        cyc1_feat1 = self.forward_decode_head(self.decode_head_ts, Fo_cyc1)
        cyc1_feat1_o = torch.unsqueeze(cyc1_feat1.argmax(dim=1),1)
        recf_cyc1 = self.dsk_neck1(Fo_cyc1[0],cyc1_feat1_o)
        cyc1_feat1 = resize(input=cyc1_feat1,size=pseudo_label1.shape[2:],mode='bilinear',align_corners=self.align_corners)
        #cyc1_feat1_l,mixed_lbl_nw_l,recf_cyc1_l = [None] * batch_size, [None] * batch_size, [None] * batch_size

        #for i in range(batch_size):
        #    cyc1_feat1_l[i] =  cyc1_feat1[i]*mix_masks_nw_l[i]
        #    mixed_lbl_nw_l[i] =  mixed_lbl_nw[i]*mix_masks_nw_l[i] 
        #    recf_cyc1_l[i] =  recf_cyc1[i]*mix_masks_nw[i]
        #cyc1_feat1_l = torch.cat(cyc1_feat1_l)
        #mixed_lbl_nw_l = torch.cat(mixed_lbl_nw_l) 
        #recf_cyc1_l = torch.cat(recf_cyc1_l) 
        cyc1_loss1, cyc1_log_vars1 = self._get_segmentor_loss(self.decode_head_ts, cyc1_feat1, mixed_lbl_nw,pseudo_weight)
        #, y_cutmix.float())
        losscyc1_rec_cm1 =  nn.L1Loss(reduction='none')(recf_cyc1, mixed_1)
        losscyc1_rec_cm1 =  (losscyc1_rec_cm1*mask_save.float()).sum()
        non_zero_elements = mask_save.sum()
        losscyc1_rec_cm1 = losscyc1_rec_cm1 / non_zero_elements

        '''for_cyc2 = mask_image(recf_mic1)
        Fo_cyc2 = self.forward_backbone(self.backbone_s, for_cyc2)
        cyc2_feat1 = self.forward_decode_head(self.decode_head_s, Fo_cyc2)
        cyc2_loss1, cyc2_log_vars1 = self._get_segmentor_loss(self.decode_head_s, cyc2_feat1, data_batch['gt_semantic_seg'])
        recf_cyc2 = self.dsk_neck(Fo_cyc2[0])#,data_batch['gt_semantic_seg'].float())
        losscyc2_rec_cm1 = torch.nn.L1Loss()(recf_cyc2, for_mix)'''
        
        loss_seg_m2 =    cyc1_loss1 + losscyc1_rec_cm1*.3#*alpha #+losscyc2_rec_cm1*alpha)/2
        
		
		###%%%%%%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for_mic_both=mixed_1_both#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic_both = self.forward_backbone(self.backbone_s, for_mic_both)
        mic_feat_both = self.forward_decode_head(self.decode_head_st, Fo_mic_both)
        #print(torch.unsqueeze(mask1[:,0,:,:],dim=1).shape,mic_feat.shape)#print(mask1[:,0,:,:].shape)
 
        #print(mask1.shape,mask1_label.shape,mask1_label_resize.shape)
        
        #print("###########", Fo_mic[0].type(),y_cutmix.type())
        mic_feat_both_o = torch.unsqueeze(mic_feat_both.argmax(dim=1),1)
        recf_mic_both = self.dsk_neck(Fo_mic_both[0],mic_feat_both_o)#,y_cutmix.float())
        mic_feat_both = resize(input=mic_feat_both,size=pseudo_label1.shape[2:],mode='bilinear',align_corners=self.align_corners)        
        #mic_feat_both_l,mixed_lbl_nw_both_l,recf_mic_both_l = [None] * batch_size, [None] * batch_size, [None] * batch_size

        #for i in range(batch_size):
        #    mic_feat_both_l[i] =  mic_feat_both[i]*mix_masks_nw_l[i]
        #    mixed_lbl_nw_both_l[i] =  mixed_lbl_nw_both[i]*mix_masks_nw_l[i] 
        #    recf_mic_both_l[i] =  recf_mic_both[i]*mix_masks_nw[i]
        #mic_feat_both_l = torch.cat(mic_feat_both_l)
        #mixed_lbl_nw_both_l = torch.cat(mixed_lbl_nw_both_l) 
        #recf_mic_both_l = torch.cat(recf_mic_both_l) 
        mic_loss_both, mic_log_vars_both = self._get_segmentor_loss(self.decode_head_st, mic_feat_both, mixed_lbl_nw_both,pseudo_weight_both)
        lossmic_rec_cm_both =  nn.L1Loss(reduction='none')(recf_mic_both, mixed_2_both)
        lossmic_rec_cm_both =  (lossmic_rec_cm_both*mask_save.float()).sum()
        non_zero_elements = mask_save.sum()
        lossmic_rec_cm_both = lossmic_rec_cm_both / non_zero_elements
		#rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        #log_vars.update(mic_log_vars) 
        
        
        
        '''for_mic1 = mask_image(for_mix)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic1 = self.forward_backbone(self.backbone_s, for_mic1)
        mic_feat1 = self.forward_decode_head(self.decode_head_s, Fo_mic1)
        mic_loss1, mic_log_vars1 = self._get_segmentor_loss(self.decode_head_s, mic_feat1, data_batch['gt_semantic_seg'])
        recf_mic1 = self.dsk_neck(Fo_mic1[0])#,data_batch['gt_semantic_seg'].float())
        lossmic_rec_cm1 = torch.nn.L1Loss()(recf_mic1, for_mix)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mic_log_vars1)'''
        
        
        #loss_seg_m =  (mix_loss + mixf_loss)/2  +  (mic_loss + mic_loss1)/2 + (loss_rec_cm*0.05+lossf_rec_cm*0.05)/2  + (lossmic_rec_cm1*0.05 +lossmic_rec_cm*0.05)/2
        loss_seg_m1_both =    mic_loss_both + lossmic_rec_cm_both*.3#*alpha #+lossmic_rec_cm*alpha)/2

        
        ##################Cycle####
        #rec_mx
        #recf_mx
        
        for_cyc1_both = mixed_2_both#*(1-mask1)#mask_image(recf_mic)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_cyc1_both = self.forward_backbone(self.backbone_s, for_cyc1_both)
        cyc1_feat1_both = self.forward_decode_head(self.decode_head_ts, Fo_cyc1_both)
        cyc1_feat1_both_o = torch.unsqueeze(cyc1_feat1_both.argmax(dim=1),1)
        recf_cyc1_both = self.dsk_neck1(Fo_cyc1_both[0],cyc1_feat1_both_o)#, y_cutmix.float())
        cyc1_feat1_both = resize(input=cyc1_feat1_both,size=pseudo_label1.shape[2:],mode='bilinear',align_corners=self.align_corners)
        #cyc1_feat1_both_l,mixed_lbl_nw_both_l,recf_cyc1_both_l = [None] * batch_size, [None] * batch_size, [None] * batch_size

        #for i in range(batch_size):
        #    cyc1_feat1_both_l[i] =  cyc1_feat1_both[i]*(1 - mix_masks_nw_l[i])
        #    mixed_lbl_nw_both_l[i] =  mixed_lbl_nw_both[i]*(1 - mix_masks_nw_l[i])
        #    recf_cyc1_both_l[i] =  recf_cyc1_both[i]*(1 - mix_masks_nw[i])
        #cyc1_feat1_both_l = torch.cat(cyc1_feat1_both_l)
        #mixed_lbl_nw_both_l = torch.cat(mixed_lbl_nw_both_l) 
        #recf_cyc1_both_l = torch.cat(recf_cyc1_both_l) 
        cyc1_loss1_both, cyc1_log_vars1 = self._get_segmentor_loss(self.decode_head_ts, cyc1_feat1_both, mixed_lbl_nw_both,pseudo_weight_both)
        
        losscyc1_rec_cm1_both =  nn.L1Loss(reduction='none')(recf_cyc1_both, mixed_1_both)
        losscyc1_rec_cm1_both =(losscyc1_rec_cm1_both*(1-mask_save).float()).sum()
        non_zero_elements = (1-mask_save).sum()
        losscyc1_rec_cm1_both = losscyc1_rec_cm1_both / non_zero_elements

        '''for_cyc2 = mask_image(recf_mic1)
        Fo_cyc2 = self.forward_backbone(self.backbone_s, for_cyc2)
        cyc2_feat1 = self.forward_decode_head(self.decode_head_s, Fo_cyc2)
        cyc2_loss1, cyc2_log_vars1 = self._get_segmentor_loss(self.decode_head_s, cyc2_feat1, data_batch['gt_semantic_seg'])
        recf_cyc2 = self.dsk_neck(Fo_cyc2[0])#,data_batch['gt_semantic_seg'].float())
        losscyc2_rec_cm1 = torch.nn.L1Loss()(recf_cyc2, for_mix)'''
        
        loss_seg_m2_both =    cyc1_loss1_both + losscyc1_rec_cm1_both*.3#*alpha #+losscyc2_rec_cm1*alpha)/2
		###$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		
		
        ########################################
        ############%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''for_cyc3 = mask_image(recf_cyc1)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_cyc3 = self.forward_backbone(self.backbone_s, for_cyc3)
        cyc3_feat1 = self.forward_decode_head(self.decode_head_s, Fo_cyc3)
        cyc3_loss1, cyc3_log_vars1 = self._get_segmentor_loss(self.decode_head_s, cyc3_feat1, mixed_lbl)
        recf_cyc3 = self.dsk_neck(Fo_cyc3[0])#,y_cutmix.float())
        losscyc3_rec_cm1 = torch.nn.L1Loss()(recf_cyc3, mixed_img)'''
        
        
        '''for_cyc4 = mask_image(recf_cyc2)
        Fo_cyc4 = self.forward_backbone(self.backbone_s, for_cyc4)
        cyc4_feat1 = self.forward_decode_head(self.decode_head_s, Fo_cyc4)
        cyc4_loss1, cyc2_log_vars1 = self._get_segmentor_loss(self.decode_head_s, cyc4_feat1, data_batch['gt_semantic_seg'])
        recf_cyc4 = self.dsk_neck(Fo_cyc4[0])#,data_batch['gt_semantic_seg'].float())
        losscyc4_rec_cm1 = torch.nn.L1Loss()(recf_cyc4, for_mix)'''
        
        #loss_seg_m3 =   cyc3_loss1   + losscyc3_rec_cm1*alpha #+losscyc4_rec_cm1*alpha)/2
        ###############################################
        mixed_1_s, mixed_1_t,l_s,l_t = [None] * batch_size, [None] * batch_size,[None] * batch_size,[None] * batch_size
        #print(mic_feat.shape,mic_feat_both.shape)
        mic_feat = resize(
            input=mic_feat,
            size=pseudo_label1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        mic_feat_both = resize(
            input=mic_feat_both,
            size=pseudo_label1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cyc1_feat1 = resize(
            input=cyc1_feat1,
            size=pseudo_label1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cyc1_feat1_both = resize(
            input=cyc1_feat1_both,
            size=pseudo_label1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        for i in range(batch_size):
            mixed_1_s[i] =  recf_mic[i]*(mix_masks_nw[i]) +recf_mic_both[i]*(1-mix_masks_nw[i])
            mixed_1_t[i] =  recf_cyc1[i]* (1-mix_masks_nw[i]) +recf_cyc1_both[i]* mix_masks_nw[i]
            l_s[i] =  mic_feat[i]*(mix_masks_nw[i]) +mic_feat_both[i]*(1-mix_masks_nw[i])
            l_t[i] =  cyc1_feat1[i]* (1-mix_masks_nw[i]) +cyc1_feat1_both[i]* mix_masks_nw[i]
        mixed_1_s = torch.cat(mixed_1_s)
        mixed_1_t = torch.cat(mixed_1_t) #.long()
        l_s = torch.cat(l_s).long()
        l_t = torch.cat(l_t).long() #.long()

		##data_batch['img'] data_batch['B_img'] data_batch['gt_semantic_seg'] pseudo_label1
        cross_org1 =  nn.L1Loss()(mixed_1_s, data_batch['img'])
        cross_org2 =  nn.L1Loss()(mixed_1_t, data_batch['B_img'])
		
        '''print(torch.unique(mixed_lbl_nw_both))
        print(torch.unique(mixed_lbl_nw))
        print(torch.unique(recf_mic))
        print(torch.unique(recf_cyc1))

        print(torch.unique(l_s))
        print(torch.unique(l_t))
        print('xxx',torch.unique(torch.argmax(l_s,dim=1)))#.max(dim=1, keepdim=True)[0]))
        print(torch.unique(mixed_lbl_nw_both))'''
		
        #print(l_s.dtype)
		
        #print(torch.unique(data_batch['gt_semantic_seg']))
        cross_org_seg1 = dice_loss(l_s, data_batch['gt_semantic_seg'])
        cross_org_seg2 = dice_loss(l_t, pseudo_label1)
		#n, c, h, w = mic_feat.shape
        #cross_seg_org = kl_loss(log_sm(mic_feat) , sm(cyc1_feat1) )/(n*h*w)
        '''print(l_s.dtype,data_batch['gt_semantic_seg'].dtype)
        loss_seg1 = F.cross_entropy(
        l_s,
        data_batch['gt_semantic_seg'])
        loss_seg2 = F.cross_entropy(
        l_t,
        pseudo_label1,
        weight='none',
        reduction='none')'''
        loss_org = (cross_org1 + cross_org2) #+ (cross_org_seg1 + cross_org_seg2)
		#########################################
        ###########################
        #previous save value 0.05
        #loss_seg_m = loss_seg_m1 + loss_seg_m2*0.5 + loss_seg_m3*0.5
        cross =  nn.L1Loss()(recf_cyc1, recf_mic)
        n, c, h, w = mic_feat.shape
        cross_seg = dice_loss(mic_feat,cyc1_feat1)#kl_loss(log_sm(mic_feat) , sm(cyc1_feat1) )/(n*h*w)
		
        #feat_cross = CosineSimilarityLoss()(Fo_mic[0], Fo_cyc1[0])
        feature_batch1=Fo_mic[0]
        feature_batch2=Fo_mic_both[0]
        feature_batch1_flat = feature_batch1.reshape(feature_batch1.size(0), -1)  # Flatten BCHW to B x (C*H*W)
        feature_batch2_flat = feature_batch2.reshape(feature_batch2.size(0), -1)  # Flatten BCHW to B x (C*H*W)


        cross_both =  nn.L1Loss()(recf_cyc1_both, recf_mic_both)
        n, c, h, w = mic_feat_both.shape
        cross_seg_both = dice_loss(mic_feat_both,cyc1_feat1_both)#kl_loss(log_sm(mic_feat_both) , sm(cyc1_feat1_both) )/(n*h*w)
		

        similarity = F.cosine_similarity(feature_batch1_flat, feature_batch2_flat, dim=1)
        feat_cross = -torch.mean(similarity)
		
        #similarity_both = F.cosine_similarity(feature_batch1_flat_both, feature_batch2_flat_both, dim=1)
        #feat_cross_both = torch.mean(similarity_both)
        loss_seg_m = loss_seg_m1*0.9 + loss_seg_m2*0.9+ (cross + cross_seg)*0.1 #*.5 + loss_seg_m3*.25
        loss_seg_m_both = loss_seg_m1_both*0.9 + loss_seg_m2_both*0.9+ (cross_both + cross_seg_both)*0.1 #+ feat_cross_both*0.1
        #print(loss_seg_m1.item(),loss_seg_m2.item(), cross.item(), cross_seg.item())
        #cyc1_feat1
        #loss_seg_m = loss_seg_m*.35+loss_seg_m_both*.35+ feat_cross*0.15 +loss_org*.15
        loss_seg_m = loss_seg_m+loss_seg_m_both+ feat_cross*0.1 +loss_org*.1
        one_channel_output = torch.unsqueeze(mic_feat.argmax(dim=1),1)#max(dim=1, keepdim=True)[0]
        #print(mic_feat.shape, one_channel_output.shape,"dddddddddd")
        one_channel_output1 = torch.unsqueeze(cyc1_feat1.argmax(dim=1),1)#.max(dim=1, keepdim=True)[0]
        one_channel_output_both = torch.unsqueeze(mic_feat_both.argmax(dim=1),1)#.max(dim=1, keepdim=True)[0]
        #print(torch.unique(cyc1_feat1))
        #print(torch.unique(one_channel_output1))
        one_channel_output1_both = torch.unsqueeze(cyc1_feat1_both.argmax(dim=1),1)#.max(dim=1, keepdim=True)[0]
        one_channel_s = torch.unsqueeze(l_s.argmax(dim=1),1)#.max(dim=1, keepdim=True)[0]
        one_channel_t = torch.unsqueeze(l_t.argmax(dim=1),1)#.max(dim=1, keepdim=True)[0]
        #real_gan=torch.cat((mixed_1,mixed_lbl_nw,mixed_1_both,mixed_lbl_nw_both),1)
        real_gan_org1=torch.cat((data_batch['img'] , data_batch['gt_semantic_seg']),1)
        real_gan_org2=torch.cat((data_batch['B_img'], pseudo_label1),1)
        '''one_channel_output = resize(
            input=one_channel_output,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        one_channel_output1 = resize(
            input=one_channel_output1,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        one_channel_output_both = resize(
            input=one_channel_output_both,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        one_channel_output1_both = resize(
            input=one_channel_output1_both,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        #fake_gan1=torch.cat((one_channel_output, recf_mic,one_channel_output_both , recf_mic_both),1)
        #fake_gan2=torch.cat((one_channel_output1,recf_cyc1,one_channel_output1_both,recf_cyc1_both),1)
        fake_gan1=torch.cat((one_channel_output, mixed_img_nw),1)
        real_gan1=torch.cat((one_channel_output1,mixed_img_nw),1)

        fake_gan2=torch.cat((one_channel_output_both , mixed_img_nw_both),1)
        real_gan2=torch.cat((one_channel_output1_both,mixed_img_nw_both),1)
		
        #fake_gan2=torch.cat((mixed_lbl_nw_both , recf_mic_both),1)
        #real_gan2=torch.cat((mixed_lbl_nw_both,recf_cyc1_both),1)
		
        fake_gan_org1=torch.cat((mixed_1_s,one_channel_s),1)
        fake_gan_org2=torch.cat((mixed_1_t,one_channel_t),1)
        F_s2s = real_gan1#F_s2s_all[-1]
        F_t2s = fake_gan1#F_t2s_all[-1]
        F_s2s_dis_sm = F_s2s
        F_t2s_dis_sm = F_t2s
        #print(F_s2s_dis_sm.shape,F_t2s_dis_sm.shape,mixed_lbl_nw.shape)
        F_s2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_sm)
        F_t2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_sm)
        F_t2s_dis_oup = resize(
            input=F_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_oup = resize(
            input=F_s2s_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_s, log_vars_dis_s = self._get_gan_loss(self.discriminator_s, F_t2s_dis_oup, 'F_t2s_ds_seg', 1)
        log_vars.update(log_vars_dis_s)
		
		
		#fake_gan2=torch.cat((recf_cyc1,cyc1_feat1),1)
        F_s2t = real_gan2#F_s2t_all[-1]
        F_t2t = fake_gan2#F_t2t_all[-1]
        F_s2t_dis_sm = F_s2t
        F_t2t_dis_sm = F_t2t
        F_s2t_dis_oup = self.forward_discriminator(self.discriminator_s, F_s2t_dis_sm)
        F_t2t_dis_oup = self.forward_discriminator(self.discriminator_s, F_t2t_dis_sm)
        F_t2t_dis_oup = resize(
            input=F_t2t_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_oup = resize(
            input=F_s2t_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_t, log_vars_dis_t = self._get_gan_loss(self.discriminator_s, F_s2t_dis_oup, 'F_s2t_dt_seg', 1)
		
        #fake_gan2=torch.cat((recf_cyc1,cyc1_feat1),1)
        F_s2t_org1 = real_gan_org1#F_s2t_all[-1]
        F_t2t_org1 = fake_gan_org1#F_t2t_all[-1]
        #F_s2t_dis_sm_org1 = F_s2t_org1
        #F_t2t_dis_sm_org1 = F_t2t_org1

        F_s2t_dis_sm_org1 = Fo_mic_both[0]
        F_t2t_dis_sm_org1 = Fo_cyc1_both[0]
        F_s2t_dis_oup_org1 = self.forward_discriminator(self.discriminator_t, F_s2t_dis_sm_org1)
        F_t2t_dis_oup_org1 = self.forward_discriminator(self.discriminator_t, F_t2t_dis_sm_org1)
        F_t2t_dis_oup_org1 = resize(
            input=F_t2t_dis_oup_org1,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_oup_org1 = resize(
            input=F_s2t_dis_oup_org1,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_t_org1, log_vars_dis_t_org1 = self._get_gan_loss(self.discriminator_t, F_s2t_dis_oup_org1, 'F_s2t_dt_seg_org1', 1)
		
		#######################
		
        F_s2t_org2 = Fo_mic[0]#F_s2t_all[-1]
        F_t2t_org2 = Fo_cyc1[0]#F_t2t_all[-1]
        F_s2t_dis_sm_org2 = F_s2t_org2
        F_t2t_dis_sm_org2 = F_t2t_org2
        F_s2t_dis_oup_org2 = self.forward_discriminator(self.discriminator_t, F_s2t_dis_sm_org2)
        F_t2t_dis_oup_org2 = self.forward_discriminator(self.discriminator_t, F_t2t_dis_sm_org2)
        '''F_t2t_dis_oup_org2 = resize(
            input=F_t2t_dis_oup_org2,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''F_s2t_dis_oup_org2 = resize(
            input=F_s2t_dis_oup_org2,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        loss_dis_t_org2, log_vars_dis_t_org2 = self._get_gan_loss(self.discriminator_t, F_s2t_dis_oup_org2, 'F_s2t_dt_seg_org2', 1)
        loss_adv = (loss_dis_t + loss_dis_s)+ (loss_dis_t_org1 + loss_dis_t_org2)
		
        loss_seg_m = loss_seg_m + loss_adv*0.05
		
		
		
        loss_seg_m.backward()
		
		
		
		
        '''for_mic_cmx,mask1_cmx = mask_image(x_cutmix)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic_cmx = self.forward_backbone(self.backbone_s, for_mic_cmx)
        mic_feat_cmx = self.forward_decode_head(self.decode_head_s, Fo_mic_cmx)
        #print(torch.unsqueeze(mask1[:,0,:,:],dim=1).shape,mic_feat.shape)#print(mask1[:,0,:,:].shape)
        mask1_label= mask1#torch.cat((mask1,mask1,torch.unsqueeze(mask1[:,0,:,:],dim=1)),dim=1)
        mask1_label_resize= resize(
            input=mask1,
            size=mic_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #print(mask1.shape,mask1_label.shape,mask1_label_resize.shape)
        mic_loss_cmx, mic_log_vars_cmx = self._get_segmentor_loss(self.decode_head_s, mic_feat_cmx, y_cutmix.long())
        #print("###########", Fo_mic[0].type(),y_cutmix.type())
        recf_mic_cmx = self.dsk_neck(Fo_mic_cmx[0])#,y_cutmix.float())
        lossmic_rec_cm_cmx = torch.nn.L1Loss()(recf_mic_cmx, x_cutmix)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mic_log_vars_cmx)
        

        
        
        #loss_seg_m =  (mix_loss + mixf_loss)/2  +  (mic_loss + mic_loss1)/2 + (loss_rec_cm*0.05+lossf_rec_cm*0.05)/2  + (lossmic_rec_cm1*0.05 +lossmic_rec_cm*0.05)/2
        loss_seg_m1_cmx =   mic_loss_cmx   + lossmic_rec_cm_cmx*alpha #+lossmic_rec_cm*alpha)/2

        
        ##################Cycle####
        #rec_mx
        #recf_mx
        
        for_cyc1_cmx = recf_mic_cmx*(1-mask1_cmx)#mask_image(recf_mic)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_cyc1_cmx = self.forward_backbone(self.backbone_s, for_cyc1_cmx)
        cyc1_feat1_cmx = self.forward_decode_head(self.decode_head_s, Fo_cyc1_cmx)
        cyc1_loss1_cmx, cyc1_log_vars1_cmx = self._get_segmentor_loss(self.decode_head_s, cyc1_feat1_cmx, y_cutmix.long())
        recf_cyc1_cmx = self.dsk_neck(Fo_cyc1_cmx[0])#, y_cutmix.float())
        losscyc1_rec_cm1_cmx = torch.nn.L1Loss()(recf_cyc1_cmx, x_cutmix)
        
        
        
        loss_seg_m2_cmx =   cyc1_loss1_cmx   + losscyc1_rec_cm1_cmx*alpha #+losscyc2_rec_cm1*alpha)/2
        
        loss_seg_m_cmx = loss_seg_m1_cmx + loss_seg_m2_cmx*alpha#*.5 + loss_seg_m3*.25
        
        loss_seg_m_cmx.backward()
		
		
		
		
        for_mic_fr,mask1_fr = mask_image(for_mix)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_mic_fr = self.forward_backbone(self.backbone_s, for_mic_fr)
        mic_feat_fr = self.forward_decode_head(self.decode_head_s, Fo_mic_fr)
        #print(torch.unsqueeze(mask1[:,0,:,:],dim=1).shape,mic_feat.shape)#print(mask1[:,0,:,:].shape)
        mask1_label= mask1#torch.cat((mask1,mask1,torch.unsqueeze(mask1[:,0,:,:],dim=1)),dim=1)
        mask1_label_resize= resize(
            input=mask1,
            size=mic_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #print(mask1.shape,mask1_label.shape,mask1_label_resize.shape)
        mic_loss_fr, mic_log_vars_fr = self._get_segmentor_loss(self.decode_head_s, mic_feat_fr, data_batch['gt_semantic_seg'])
        #print("###########", Fo_mic[0].type(),y_cutmix.type())
        recf_mic_fr = self.dsk_neck(Fo_mic_fr[0])#,y_cutmix.float())
        lossmic_rec_cm_fr = torch.nn.L1Loss()(recf_mic_fr, for_mix)
        #rec_t = self.dsk_neck(F_s2t_all)
        #get_model().forward_train(
        #    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        #mix_losses.pop('features')
        #mix_losses = add_prefix(mix_losses, 'mix')
        #mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mic_log_vars_fr)
        

        
        
        #loss_seg_m =  (mix_loss + mixf_loss)/2  +  (mic_loss + mic_loss1)/2 + (loss_rec_cm*0.05+lossf_rec_cm*0.05)/2  + (lossmic_rec_cm1*0.05 +lossmic_rec_cm*0.05)/2
        loss_seg_m1_fr =   mic_loss_fr   + lossmic_rec_cm_fr*alpha #+lossmic_rec_cm*alpha)/2

        
        ##################Cycle####
        #rec_mx
        #recf_mx
        
        for_cyc1_fr = recf_mic_fr*(1-mask1_fr)#mask_image(recf_mic)#fourier_mix(data_batch['img'],data_batch['B_img'])
        Fo_cyc1_fr = self.forward_backbone(self.backbone_s, for_cyc1_fr)
        cyc1_feat1_fr = self.forward_decode_head(self.decode_head_s, Fo_cyc1_fr)
        cyc1_loss1_fr, cyc1_log_vars1_fr = self._get_segmentor_loss(self.decode_head_s, cyc1_feat1_fr, data_batch['gt_semantic_seg'])
        recf_cyc1_fr = self.dsk_neck(Fo_cyc1_fr[0])#, y_cutmix.float())
        losscyc1_rec_cm1_fr = torch.nn.L1Loss()(recf_cyc1_fr, for_mix)
        
        
        
        loss_seg_m2_fr =   cyc1_loss1_fr   + losscyc1_rec_cm1_fr*alpha #+losscyc2_rec_cm1*alpha)/2
        
        loss_seg_m_fr = loss_seg_m1_fr + loss_seg_m2_fr*alpha#*.5 + loss_seg_m3*.25
        
        loss_seg_m_fr.backward()'''

        #self.visualize = self.visualize + 1
        optimizer['backbone_s'].step()
        #optimizer['backbone_t'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_st'].step()
        optimizer['decode_head_ts'].step()
        #optimizer['decode_head_t'].step()
        optimizer['dsk_neck'].step()
        optimizer['dsk_neck1'].step()
		
        self.set_requires_grad(self.dsk_neck, False)
        self.set_requires_grad(self.dsk_neck1, False)
        #log_vars = dict()

        ## 1.1 forward backbone
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_st, False)
        self.set_requires_grad(self.decode_head_ts, False)
		
        self.set_requires_grad(self.discriminator_s, True)
        F_s2s_dis_detach = F_s2s_dis_sm.detach()
        F_t2s_dis_detach = F_t2s_dis_sm.detach()
        F_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_detach)
        F_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_detach)
        F_t2s_dis_detach_oup = resize(
            input=F_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_detach_oup = resize(
            input=F_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_s2s_ds, log_vars_adv_s2s_ds = self._get_gan_loss(self.discriminator_s, F_s2s_dis_detach_oup, 'F_s2s_ds', 1)
        loss_adv_s2s_ds.backward()
        log_vars.update(log_vars_adv_s2s_ds)
        loss_adv_t2s_ds, log_vars_adv_t2s_ds = self._get_gan_loss(self.discriminator_s, F_t2s_dis_detach_oup, 'F_t2s_ds', 0)
        loss_adv_t2s_ds.backward()
        log_vars.update(log_vars_adv_t2s_ds)
        #optimizer['discriminator_s'].step()
        #self.set_requires_grad(self.discriminator_s, False)

        ## 2.2.5 discriminator_t
        #self.set_requires_grad(self.discriminator_s, True)
        F_s2t_dis_detach = F_s2t_dis_sm.detach()
        F_t2t_dis_detach = F_t2t_dis_sm.detach()
        F_s2t_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_s2t_dis_detach)
        F_t2t_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_t2t_dis_detach)
        F_t2t_dis_detach_oup = resize(
            input=F_t2t_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup = resize(
            input=F_s2t_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_t2t_dt, log_vars_adv_t2t_dt = self._get_gan_loss(self.discriminator_s, F_t2t_dis_detach_oup, 'F_t2t_dt', 1)
        loss_adv_t2t_dt.backward()
        log_vars.update(log_vars_adv_t2t_dt)
        loss_adv_s2t_dt, log_vars_adv_s2t_dt = self._get_gan_loss(self.discriminator_s, F_s2t_dis_detach_oup, 'F_s2t_dt', 0)
        loss_adv_s2t_dt.backward()
        log_vars.update(log_vars_adv_s2t_dt)
        optimizer['discriminator_s'].step()
        self.set_requires_grad(self.discriminator_s, False)



        self.set_requires_grad(self.discriminator_t, True)
        mc, cy1, mc_1, cy1_1 = 	Fo_mic_both[0],Fo_cyc1_both[0],Fo_mic[0],Fo_cyc1[0]	
        #print("#########################",mc.shape, cy1.shape, mc_1.shape, cy1_1.shape)			
        F_s2t_dis_detach_org1 = mc.detach().contiguous() 
        F_t2t_dis_detach_org1 = cy1.detach().contiguous() 
        F_s2t_dis_detach_oup_org1 = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach_org1)
        F_t2t_dis_detach_oup_org1 = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach_org1)
        '''F_t2t_dis_detach_oup_org1 = resize(
            input=F_t2t_dis_detach_oup_org1,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup_org1 = resize(
            input=F_s2t_dis_detach_oup_org1,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        loss_adv_t2t_dt_org1, log_vars_adv_t2t_dt_org1 = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup_org1, 'F_t2t_dt1', 1)
        loss_adv_t2t_dt_org1.backward()
        log_vars.update(log_vars_adv_t2t_dt_org1)
        loss_adv_s2t_dt_org1, log_vars_adv_s2t_dt_org1 = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup_org1, 'F_s2t_dt1', 0)
        loss_adv_s2t_dt_org1.backward()
        log_vars.update(log_vars_adv_s2t_dt_org1)
		
		
        F_s2t_dis_detach_org2 = mc_1.detach().contiguous() 
        F_t2t_dis_detach_org2 = cy1_1.detach().contiguous() 
        F_s2t_dis_detach_oup_org2 = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach_org2)
        F_t2t_dis_detach_oup_org2 = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach_org2)
        '''F_t2t_dis_detach_oup_org2 = resize(
            input=F_t2t_dis_detach_oup_org2,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup_org2 = resize(
            input=F_s2t_dis_detach_oup_org2,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        loss_adv_t2t_dt_org2, log_vars_adv_t2t_dt_org2 = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup_org2, 'F_t2t_dt2', 1)
        loss_adv_t2t_dt_org2.backward()
        log_vars.update(log_vars_adv_t2t_dt_org2)
        loss_adv_s2t_dt_org2, log_vars_adv_s2t_dt_org2 = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup_org2, 'F_s2t_dt2', 0)
        loss_adv_s2t_dt_org2.backward()
        log_vars.update(log_vars_adv_s2t_dt_org2)
		
		
		
        optimizer['discriminator_t'].step()
        self.set_requires_grad(self.discriminator_t, False)		
		## 2.2.5 discriminator_t ORG
        '''self.set_requires_grad(self.discriminator_t, True)
        F_s2t_dis_detach_org1 = F_s2t_dis_sm_org1.detach()
        F_t2t_dis_detach_org1 = F_t2t_dis_sm_org1.detach()
        F_s2t_dis_detach_oup_org1 = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach_org1)
        F_t2t_dis_detach_oup_org1 = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach_org1)
        F_t2t_dis_detach_oup_org1 = resize(
            input=F_t2t_dis_detach_oup_org1,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup_org1 = resize(
            input=F_s2t_dis_detach_oup_org1,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_t2t_dt_org1, log_vars_adv_t2t_dt_org1 = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup_org1, 'F_t2t_dt1', 1)
        loss_adv_t2t_dt_org1.backward()
        log_vars.update(log_vars_adv_t2t_dt_org1)
        loss_adv_s2t_dt_org1, log_vars_adv_s2t_dt_org1 = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup_org1, 'F_s2t_dt1', 0)
        loss_adv_s2t_dt_org1.backward()
        log_vars.update(log_vars_adv_s2t_dt_org1)
		
		
        F_s2t_dis_detach_org2 = F_s2t_dis_sm_org2.detach()
        F_t2t_dis_detach_org2 = F_t2t_dis_sm_org2.detach()
        F_s2t_dis_detach_oup_org2 = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach_org2)
        F_t2t_dis_detach_oup_org2 = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach_org2)
        F_t2t_dis_detach_oup_org2 = resize(
            input=F_t2t_dis_detach_oup_org2,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup_org2 = resize(
            input=F_s2t_dis_detach_oup_org2,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_t2t_dt_org2, log_vars_adv_t2t_dt_org2 = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup_org2, 'F_t2t_dt2', 1)
        loss_adv_t2t_dt_org2.backward()
        log_vars.update(log_vars_adv_t2t_dt_org2)
        loss_adv_s2t_dt_org2, log_vars_adv_s2t_dt_org2 = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup_org2, 'F_s2t_dt2', 0)
        loss_adv_s2t_dt_org2.backward()
        log_vars.update(log_vars_adv_s2t_dt_org2)
		
		
		
        optimizer['discriminator_t'].step()
        self.set_requires_grad(self.discriminator_t, False)'''

        '''F_s2s = F_s2s_all[-1]
        F_t2s = F_s2t_all[-1]
        Fmix = F_mix[-1]
        F_s2s_dis_sm = self.sw_softmax(F_s2s)
        F_t2s_dis_sm = self.sw_softmax(F_t2s)
        Fmix_sm = self.sw_softmax(Fmix)
        F_s2s_dis_oup = self.forward_discriminator(self.discriminator_f, F_s2s_dis_sm)
        F_t2s_dis_oup = self.forward_discriminator(self.discriminator_f, F_t2s_dis_sm)
        Fmix_oup = self.forward_discriminator(self.discriminator_f, Fmix_sm)
        F_t2s_dis_oup = resize(
            input=F_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_oup = resize(
            input=F_s2s_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
            
        Fmix_oup = resize(
            input=Fmix_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_f, log_vars_dis_f = self._get_gan_loss(self.discriminator_f, Fmix_oup, 'Fmix_oup', 1)
        log_vars.update(log_vars_dis_f)   
        
        
        ###########################################
        
        C_s2s_dis_sm = self.sw_softmax(P_s2s)
        C_t2s_dis_sm = self.sw_softmax(P_s2t)
        Cmix_sm = self.sw_softmax(mix_feat)
        C_s2s_dis_oup = self.forward_discriminator(self.discriminator_c, C_s2s_dis_sm)
        C_t2s_dis_oup = self.forward_discriminator(self.discriminator_c, C_t2s_dis_sm)
        Cmix_oup = self.forward_discriminator(self.discriminator_c, Cmix_sm)
        C_t2s_dis_oup = resize(
            input=C_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        C_s2s_dis_oup = resize(
            input=C_s2s_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
            
        Cmix_oup = resize(
            input=Cmix_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_c, log_vars_dis_c = self._get_gan_loss(self.discriminator_c, Cmix_oup, 'Cmix_oup', 1)
        log_vars.update(log_vars_dis_c)  
        
        
        ###########################################
        
        R_s2s_dis_sm = self.sw_softmax(rec_s)
        R_t2s_dis_sm = self.sw_softmax(rec_t)
        Rmix_sm = self.sw_softmax(rec_mx)
        R_s2s_dis_oup = self.forward_discriminator(self.discriminator_r, R_s2s_dis_sm)
        R_t2s_dis_oup = self.forward_discriminator(self.discriminator_r, R_t2s_dis_sm)
        Rmix_oup = self.forward_discriminator(self.discriminator_r, Rmix_sm)
        R_t2s_dis_oup = resize(
            input=R_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        R_s2s_dis_oup = resize(
            input=R_s2s_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
            
        Rmix_oup = resize(
            input=Rmix_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_r, log_vars_dis_r = self._get_gan_loss(self.discriminator_r, Rmix_oup, 'Rmix_oup', 1)
        log_vars.update(log_vars_dis_r) 
        

        loss_adv =  loss_dis_f + loss_dis_c +loss_dis_r
        loss_stage1 = loss_seg + loss_adv
        loss_stage1.backward()
        optimizer['backbone_s'].step()
        #optimizer['backbone_t'].step()
        optimizer['decode_head_s'].step()
        #optimizer['decode_head_t'].step()
        optimizer['dsk_neck'].step()
        self.set_requires_grad(self.backbone_s, False)
        #self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        #self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.dsk_neck, False)
        
        
        
        
        
        self.set_requires_grad(self.discriminator_f, True)
        self.set_requires_grad(self.discriminator_c, False)
        self.set_requires_grad(self.discriminator_r, False)
        F_s2s_dis_detach = F_s2s_dis_sm.detach()
        F_t2s_dis_detach = F_t2s_dis_sm.detach()
        #Fmix_detach = Fmix_sm.detach()
        #F_t2s_dis_detach = F_t2s_dis_sm.detach()
        F_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_f, F_s2s_dis_detach)
        F_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_f, F_t2s_dis_detach)
        #Fmix_detach_oup = self.forward_discriminator(self.discriminator_f, Fmix_detach)
        F_t2s_dis_detach_oup = resize(
            input=F_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_detach_oup = resize(
            input=F_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''Fmix_detach_oup = resize(
            input=Fmix_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''loss_adv_s2s_ds, log_vars_adv_s2s_ds = self._get_gan_loss(self.discriminator_f, F_s2s_dis_detach_oup, 'F_s2s_ds', 1)
        loss_adv_s2s_ds.backward()
        log_vars.update(log_vars_adv_s2s_ds)
        loss_adv_t2s_ds, log_vars_adv_t2s_ds = self._get_gan_loss(self.discriminator_f, F_t2s_dis_detach_oup, 'F_t2s_ds', 0)
        loss_adv_t2s_ds.backward()
        log_vars.update(log_vars_adv_t2s_ds)
        #loss_adv_mix_ds, log_vars_adv_mix_ds = self._get_gan_loss(self.discriminator_f, Fmix_detach_oup, 'F_mix_ds', 0)
        #loss_adv_mix_ds.backward()
        #log_vars.update(log_vars_adv_mix_ds)
        optimizer['discriminator_f'].step()
            
            
        ############################
        self.set_requires_grad(self.discriminator_f, False)
        self.set_requires_grad(self.discriminator_c, True)
        self.set_requires_grad(self.discriminator_r, False)
        C_s2s_dis_detach = C_s2s_dis_sm.detach()
        C_t2s_dis_detach = C_t2s_dis_sm.detach()
        #Cmix_detach = Cmix_sm.detach()
        #F_t2s_dis_detach = F_t2s_dis_sm.detach()
        C_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_c, C_s2s_dis_detach)
        C_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_c, C_t2s_dis_detach)
        #Cmix_detach_oup = self.forward_discriminator(self.discriminator_c, Cmix_detach)
        C_t2s_dis_detach_oup = resize(
            input=C_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        C_s2s_dis_detach_oup = resize(
            input=C_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''Cmix_detach_oup = resize(
            input=Cmix_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''loss_adv_s2s_c, log_vars_adv_s2s_c = self._get_gan_loss(self.discriminator_c, C_s2s_dis_detach_oup, 'F_s2s_c', 1)
        loss_adv_s2s_c.backward()
        log_vars.update(log_vars_adv_s2s_c)
        loss_adv_t2s_c, log_vars_adv_t2s_c = self._get_gan_loss(self.discriminator_c, C_t2s_dis_detach_oup, 'F_t2s_c', 0)
        loss_adv_t2s_c.backward()
        log_vars.update(log_vars_adv_t2s_c)
        #loss_adv_mix_c, log_vars_adv_mix_c = self._get_gan_loss(self.discriminator_c, C_t2s_dis_detach_oup, 'F_mix_c', 0)
        #loss_adv_mix_c.backward()
        #log_vars.update(log_vars_adv_mix_c)
        optimizer['discriminator_c'].step()
        
        ##################################3
        ############################
        self.set_requires_grad(self.discriminator_f, False)
        self.set_requires_grad(self.discriminator_c, False)
        self.set_requires_grad(self.discriminator_r, True)
        R_s2s_dis_detach = R_s2s_dis_sm.detach()
        R_t2s_dis_detach = R_t2s_dis_sm.detach()
        #Rmix_detach = Rmix_sm.detach()
        #F_t2s_dis_detach = F_t2s_dis_sm.detach()
        R_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_r, R_s2s_dis_detach)
        R_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_r, R_t2s_dis_detach)
        #Rmix_detach_oup = self.forward_discriminator(self.discriminator_r, Rmix_detach)
        R_t2s_dis_detach_oup = resize(
            input=R_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        R_s2s_dis_detach_oup = resize(
            input=R_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''Rmix_detach_oup = resize(
            input=Rmix_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)'''
        '''loss_adv_s2s_r, log_vars_adv_s2s_r = self._get_gan_loss(self.discriminator_r, R_s2s_dis_detach_oup, 'F_s2s_r', 1)
        loss_adv_s2s_r.backward()
        log_vars.update(log_vars_adv_s2s_r)
        loss_adv_t2s_r, log_vars_adv_t2s_r = self._get_gan_loss(self.discriminator_r, R_t2s_dis_detach_oup, 'F_t2s_r', 0)
        loss_adv_t2s_r.backward()
        log_vars.update(log_vars_adv_t2s_r)
        #loss_adv_mix_r, log_vars_adv_mix_r = self._get_gan_loss(self.discriminator_r, Rmix_detach_oup, 'F_mix_r', 0)
        #loss_adv_mix_r.backward()
        #log_vars.update(log_vars_adv_mix_r)
        optimizer['discriminator_r'].step()
        
        
        self.set_requires_grad(self.discriminator_f, False)
        self.set_requires_grad(self.discriminator_c, False)
        self.set_requires_grad(self.discriminator_r, False)'''
        
        
        ###################################
        '''F_t2s_all = self.forward_backbone(self.backbone_s, data_batch['B_img'])
        F_s2t_all = self.forward_backbone(self.backbone_t, data_batch['img'])
        F_t2t_all = self.forward_backbone(self.backbone_t, data_batch['B_img'])
        
        if isinstance(self.dsk_neck.in_channels, int):
            F_s2s = F_s2s_all[-1]
            F_t2s = F_t2s_all[-1]
            F_s2t = F_s2t_all[-1]
            F_t2t = F_t2t_all[-1]
        else:
            F_s2s = F_s2s_all
            F_t2s = F_t2s_all
            F_s2t = F_s2t_all
            F_t2t = F_t2t_all

        ## 1.2 forward dsk_neck
        self.set_requires_grad(self.dsk_neck, True)
        F_s2s_dsk, F_s2t_dsk = self.dsk_neck(F_s2s, F_s2t)
        F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t)

        ## 1.3 forward head
        P_s2s = self.forward_decode_head(self.decode_head_s, F_s2s_dsk)
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s_dsk)
        P_s2t = self.forward_decode_head(self.decode_head_t, F_s2t_dsk)
        P_t2t = self.forward_decode_head(self.decode_head_t, F_t2t_dsk)
        loss_seg_s2s, log_vars_seg_s2s = self._get_segmentor_loss(self.decode_head_s, P_s2s, data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s2s)
        loss_seg_s2t, log_vars_seg_s2t = self._get_segmentor_loss(self.decode_head_t, P_s2t, data_batch['gt_semantic_seg'])
        log_vars_seg_s2t['loss_ce_seg_s2t'] = log_vars_seg_s2t.pop('loss_ce')
        log_vars_seg_s2t['acc_seg_s2t'] = log_vars_seg_s2t.pop('acc_seg')
        log_vars_seg_s2t['loss_ce_seg_s2t'] = log_vars_seg_s2t.pop('loss')
        log_vars.update(log_vars_seg_s2t)
        loss_seg = loss_seg_s2s + loss_seg_s2t

        ##############################
        ## 1.4 forward EMA for pseudo_label
        ## CODE for cross_EMA
        # FOR decode_only_t and single_t
        pseudo_label, pseudo_weight = self.encode_decode_crossEMA(input=data_batch['B_img'], F_t2s=F_t2s_dsk, F_t2t=F_t2t_dsk, dev=data_batch['img'].device)
        loss_seg_t2s, log_vars_seg_t2s = self._get_segmentor_loss(self.decode_head_s, P_t2s, pseudo_label, gt_weight=pseudo_weight)
        log_vars_seg_t2s['loss_ce_seg_t2s'] = log_vars_seg_t2s.pop('loss_ce')
        log_vars_seg_t2s['acc_seg_t2s'] = log_vars_seg_t2s.pop('acc_seg')
        log_vars_seg_t2s['loss_ce_seg_t2s'] = log_vars_seg_t2s.pop('loss')
        log_vars.update(log_vars_seg_t2s)
        loss_seg_t2t, log_vars_seg_t2t = self._get_segmentor_loss(self.decode_head_t, P_t2t, pseudo_label, gt_weight=pseudo_weight)
        log_vars_seg_t2t['loss_ce_seg_t2t'] = log_vars_seg_t2t.pop('loss_ce')
        log_vars_seg_t2t['acc_seg_t2t'] = log_vars_seg_t2t.pop('acc_seg')
        log_vars_seg_t2t['loss_ce_seg_t2t'] = log_vars_seg_t2t.pop('loss')
        log_vars.update(log_vars_seg_t2t)
        loss_seg = loss_seg + self.cross_EMA_training_ratio * (loss_seg_t2s + loss_seg_t2t)
        ## CODE for cross_EMA
        ##############################

        ## 2.2 forward&backward Ds/Dt
        ## 2.2.1 generator/prediction alignment Ds
        F_s2s = F_s2s_all[-1]
        F_t2s = F_t2s_all[-1]
        F_s2s_dis_sm = self.sw_softmax(F_s2s)
        F_t2s_dis_sm = self.sw_softmax(F_t2s)
        F_s2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_sm)
        F_t2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_sm)
        F_t2s_dis_oup = resize(
            input=F_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_oup = resize(
            input=F_s2s_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_s, log_vars_dis_s = self._get_gan_loss(self.discriminator_s, F_t2s_dis_oup, 'F_t2s_ds_seg', 1)
        log_vars.update(log_vars_dis_s)

        ## 2.2.2 generator/prediction alignment Dt
        F_s2t = F_s2t_all[-1]
        F_t2t = F_t2t_all[-1]
        F_s2t_dis_sm = self.sw_softmax(F_s2t)
        F_t2t_dis_sm = self.sw_softmax(F_t2t)
        F_s2t_dis_oup = self.forward_discriminator(self.discriminator_t, F_s2t_dis_sm)
        F_t2t_dis_oup = self.forward_discriminator(self.discriminator_t, F_t2t_dis_sm)
        F_t2t_dis_oup = resize(
            input=F_t2t_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_oup = resize(
            input=F_s2t_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_t, log_vars_dis_t = self._get_gan_loss(self.discriminator_t, F_s2t_dis_oup, 'F_s2t_dt_seg', 1)
        log_vars.update(log_vars_dis_t)
        loss_adv = loss_dis_t + loss_dis_s
        loss_stage1 = loss_seg + loss_adv
        loss_stage1.backward()
        optimizer['backbone_s'].step()
        optimizer['backbone_t'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        optimizer['dsk_neck'].step()
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.dsk_neck, False)

        ## 2.2.4 discriminator_s
        self.set_requires_grad(self.discriminator_s, True)
        F_s2s_dis_detach = F_s2s_dis_sm.detach()
        F_t2s_dis_detach = F_t2s_dis_sm.detach()
        F_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_detach)
        F_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_detach)
        F_t2s_dis_detach_oup = resize(
            input=F_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_detach_oup = resize(
            input=F_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_s2s_ds, log_vars_adv_s2s_ds = self._get_gan_loss(self.discriminator_s, F_s2s_dis_detach_oup, 'F_s2s_ds', 1)
        loss_adv_s2s_ds.backward()
        log_vars.update(log_vars_adv_s2s_ds)
        loss_adv_t2s_ds, log_vars_adv_t2s_ds = self._get_gan_loss(self.discriminator_s, F_t2s_dis_detach_oup, 'F_t2s_ds', 0)
        loss_adv_t2s_ds.backward()
        log_vars.update(log_vars_adv_t2s_ds)
        optimizer['discriminator_s'].step()
        self.set_requires_grad(self.discriminator_s, False)

        ## 2.2.5 discriminator_t
        self.set_requires_grad(self.discriminator_t, True)
        F_s2t_dis_detach = F_s2t_dis_sm.detach()
        F_t2t_dis_detach = F_t2t_dis_sm.detach()
        F_s2t_dis_detach_oup = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach)
        F_t2t_dis_detach_oup = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach)
        F_t2t_dis_detach_oup = resize(
            input=F_t2t_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup = resize(
            input=F_s2t_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_t2t_dt, log_vars_adv_t2t_dt = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup, 'F_t2t_dt', 1)
        loss_adv_t2t_dt.backward()
        log_vars.update(log_vars_adv_t2t_dt)
        loss_adv_s2t_dt, log_vars_adv_s2t_dt = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup, 'F_s2t_dt', 0)
        loss_adv_s2t_dt.backward()
        log_vars.update(log_vars_adv_s2t_dt)
        optimizer['discriminator_t'].step()
        self.set_requires_grad(self.discriminator_t, False)
       
        loss = loss_seg
        if hasattr(self, 'iteration'):
            self.iteration += 1'''
        loss = loss_seg_m
        if hasattr(self, 'iteration'):
            self.iteration += 1        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    ## added by 
    def MSE_loss(self, teacher, student):
        MSE_loss = nn.MSELoss()
        t = self.sw_softmax(teacher)
        s = self.sw_softmax(student)
        KD_loss = MSE_loss(s, t)
        return KD_loss

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requires_grad for all the networks.

        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    @staticmethod
    def sw_softmax(pred):
        N, C, H, W = pred.shape
        pred_sh = torch.reshape(pred, (N, C, H*W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out
    
    ## added by 
    @staticmethod
    def KL_loss(teacher, student, T=5):
        KL_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(student/T, dim=1),
                             F.softmax(teacher/T, dim=1)) * (T * T)
        return KL_loss