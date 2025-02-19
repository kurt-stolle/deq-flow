from __future__ import annotations

import deq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deq.layer_utils import DEQWrapper
from deq.norm import apply_weight_norm, reset_weight_norm

from .corr import CorrBlock
from .extractor import Encoder
from .gma import Attention
from .metrics import process_metrics
from .update import UpdateBlock
from .utils.utils import coords_grid
from .variant import Variant

__all__ = ["DEQFlow"]

autocast = torch.cuda.amp.autocast


class DEQFlow(nn.Module):
    def __init__(
        self,
        variant: Variant | str,
        deq: deq.DEQBase,
        *,
        dropout: float = 0.0,
        use_gma: bool = False,
        use_legacy: bool = False,
        use_wnorm: bool = True,
        use_all_grad: bool = False,
        use_mixed_precision: bool = True,
    ):
        super(DEQFlow, self).__init__()

        if isinstance(variant, str):
            variant = Variant(variant)

        self.use_mixed_precision = use_mixed_precision
        self.use_all_grad = use_all_grad
        self.use_wnorm = use_wnorm

        match variant:
            case Variant.TINY:
                self.output_dim = odim = 64
                self.hidden_dim = hdim = 32
                self.context_dim = cdim = 32
            case Variant.MEDIUM:
                self.output_dim = odim = 256
                self.hidden_dim = hdim = 128
                self.context_dim = cdim = 128
            case Variant.LARGE:
                self.output_dim = odim = 256
                self.hidden_dim = hdim = 192
                self.context_dim = cdim = 192
            case Variant.HUGE:
                self.output_dim = odim = 256
                self.hidden_dim = hdim = 256
                self.context_dim = cdim = 256
            case Variant.GIGANTIC:
                self.output_dim = odim = 256
                self.hidden_dim = hdim = 384
                self.context_dim = cdim = 384
            case _:
                raise ValueError(f"Unknown variant: {variant}")

        # feature network, context network, and update block
        self.fnet = Encoder(output_dim=odim, norm_fn="instance", dropout=dropout)
        self.cnet = Encoder(output_dim=cdim, norm_fn="batch", dropout=dropout)
        self.update_block = UpdateBlock(
            variant, corr_levels=4, corr_radius=4, use_gma=use_gma, use_legacy=use_legacy, hidden_dim=hdim
        )

        self.mask = nn.Sequential(
            nn.Conv2d(hdim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0)
        )

        if use_gma:
            self.attn = Attention(dim=cdim, heads=1, max_pos_size=160, dim_head=cdim)
        else:
            self.attn = None

        # Added the following for DEQ
        if self.use_wnorm:
            apply_weight_norm(self.update_block)

        self.deq = deq

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def _upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def _decode(self, z_out, coords0):
        net, coords1 = z_out
        up_mask = 0.25 * self.mask(net)
        flow_up = self._upsample_flow(coords1 - coords0, up_mask)

        return flow_up

    def forward(
        self,
        image1,
        image2,
        flow_gt=None,
        valid=None,
        fc_loss=None,
        flow_init=None,
        cached_result=None,
        writer=None,
        sradius_mode=False,
        **kwargs,
    ):
        """Estimate optical flow between pair of frames"""

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.use_mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.update_block.corr_radius)

        # run the context network
        with autocast(enabled=self.use_mixed_precision):
            # cnet = self.cnet(image1)
            # net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # net = torch.tanh(net)
            inp = self.cnet(image1)
            inp = torch.relu(inp)

            if self.attn:
                attn = self.attn(inp)
            else:
                attn = None

        bsz, _, H, W = inp.shape
        coords0, coords1 = self._initialize_flow(image1)
        net = torch.zeros(bsz, hdim, H, W, device=inp.device)

        if cached_result:
            net, flow_pred_prev = cached_result
            coords1 = coords0 + flow_pred_prev

        if flow_init is not None:
            coords1 = coords1 + flow_init

        if self.use_wnorm:
            reset_weight_norm(self.update_block)  # Reset weights for WN

        def func(h, c):
            if not self.use_all_grad:
                c = c.detach()
            with autocast(enabled=self.use_mixed_precision):
                new_h, delta_flow = self.update_block(
                    h, inp, corr_fn(c), c - coords0, attn
                )  # corr_fn(coords1) produces the index correlation volumes
            new_c = c + delta_flow  # F(t+1) = F(t) + \Delta(t)
            return new_h, new_c

        deq_func = DEQWrapper(func, (net, coords1))
        z_init = deq_func.list2vec(net, coords1)
        log = inp.get_device() == 0 and np.random.uniform(0, 1) < 2e-3

        z_out, info = self.deq(deq_func, z_init, log, sradius_mode, **kwargs)
        flow_pred = [self._decode(z, coords0) for z in z_out]

        if self.training:
            flow_loss, epe = fc_loss(flow_pred, flow_gt, valid)
            metrics = process_metrics(epe, info)

            return flow_loss, metrics
        else:
            (net, coords1), flow_up = z_out[-1], flow_pred[-1]

            return coords1 - coords0, flow_up, {"sradius": info["sradius"], "cached_result": (net, coords1 - coords0)}
