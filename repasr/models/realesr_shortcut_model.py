import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger

import torch.nn as nn
from torch.nn import functional as F

from basicsr.losses import gradient_penalty_loss

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms import Normalize

import math
    

class DinoV2VE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        x = self.model.forward_features(x)
        x = x['x_norm_patchtokens']
        return x.detach().clone()


@MODEL_REGISTRY.register()
class RealESRShortcutModel(SRModel):
    """GAN-based Real_HAT Model.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRShortcutModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        self.sample_step = opt.get('sample_step', 128)  # must be 2**n
        self.test_sample_step = opt.get('test_sample_step', 4) 
        self.consistency_pct = opt.get('consistency_pct', 0.25)  # 3:1 in the paper.
        # self.usm_pct = opt.get('usm_pct', 1.0)

        if self.is_train:
            self.dino = DinoV2VE().cuda()
    
    def _get_discrete_sample_step(self, n_samples, n_steps):
        max_multiplier = math.log2(n_steps)  # since max step is converted to d=0
        multipliers = np.random.choice(
            [i for i in range(1, int(max_multiplier) + 1)],
            size=n_samples,
            replace=True,
        )
        t = np.array([self._discrete_t(m) for m in multipliers])
        d = np.array([1 / 2 ** m for m in multipliers])
        t = torch.from_numpy(t).float()[:, 0]
        d = torch.from_numpy(d).float()
        return d, t
    
    def _discrete_t(self, n: int):
        t = np.random.choice([i for i in range(1, n+1)], size=1)  # ignore last n
        d = 1 / 2 ** n
        t = t / 2 ** n
        t = 1 - t - d  # [0, 1-2d]
        return t

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.net_g.train()

        # define losses
        self.cri_flow = build_loss(train_opt['flow_opt']).to(self.device)
        self.cri_align = build_loss(train_opt['align_opt']).to(self.device)
        self.cri_consistency = build_loss(train_opt['consistency_opt']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRShortcutModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        lq = self.lq
        gt = self.gt
        # gt_usm = self.gt_usm

        n_cons = int(lq.shape[0] * self.consistency_pct)
        n_flow = lq.shape[0] - n_cons

        x0 = torch.randn_like(gt[:n_flow])
        t = torch.rand(x0.shape[0], device=x0.device)
        d = torch.zeros_like(t)

        x1 = gt[:n_flow]

        xt = x0 * (1 - t.view(-1, 1, 1, 1)) + x1 * t.view(-1, 1, 1, 1)
        xt = xt.detach().clone()

        target = (x1 - x0).detach().clone()

        # generate semantic feature
        with torch.no_grad():
            align_feat = self.dino(gt[:n_flow])

        loss_dict = OrderedDict()

        # Flow matching and Representation alignment
        self.net_g.train()
        pred, to_align_feat = self.net_g(xt, t, d, lq[:n_flow])

        l_g_flow = self.cri_flow(pred, target)
        l_g_align = self.cri_align(to_align_feat, align_feat)

        loss_dict['l_flow'] = l_g_flow
        loss_dict['l_align'] = l_g_align

        # Self-consistency for shortcut model
        d, t = self._get_discrete_sample_step(n_cons, self.sample_step)
        d = d.to(lq.device)
        t = t.to(lq.device)
        x0 = torch.randn_like(gt[n_flow:])
        x1 = gt[n_flow:]

        xt = x0 * (1 - t.view(-1, 1, 1, 1)) + x1 * t.view(-1, 1, 1, 1)
        xt = xt.detach().clone()

        with torch.no_grad():
            self.net_g_ema.eval()
            st = self.net_g_ema(xt, t, d, lq[n_flow:])
            xtd = xt + st * d.view(-1, 1, 1, 1)
            std = self.net_g_ema(xtd, t + d, d, lq[n_flow:])
            target = (st + std) / 2
            target = target.detach().clone()

        pred, _ = self.net_g(xt, t, 2 * d, lq[n_flow:])
        l_g_consistency = self.cri_consistency(pred, target)
        loss_dict['l_consistency'] = l_g_consistency

        l_g_total = l_g_flow + l_g_align + l_g_consistency

        loss_dict['l_total'] = l_g_total

        self.optimizer_g.zero_grad()
        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def test(self):
        ref = self.lq

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            ut = self.net_g_ema
        else:
            self.net_g.eval()
            ut = self.net_g

        scale = self.opt['scale']
        steps = self.test_sample_step
        xt = torch.randn((1, 3, ref.size(2) * scale, ref.size(3) * scale)).to(ref.device)
        t = torch.zeros(xt.size(0)).to(ref.device)
        d = torch.ones_like(t).to(ref.device) / steps
        for _ in range(steps):
            xt = xt + ut(xt, t, d, ref) * d.view(-1, 1, 1, 1)
            t = t + d
        
        xt = torch.clamp(xt, 0, 1)
        self.output = xt
