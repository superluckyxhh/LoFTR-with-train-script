import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import spatial_softargmax_2d
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange, repeat

from common.functions import * 
from common.nest import NestedTensor
from models.backbone import ResUNet
from models.loftr import LoFTRModule
from models.position import PositionEmbedding2D, PositionEmbedding1D

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1),
         torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

class MatchingNet(nn.Module):
    def __init__(
        self,
        d_coarse_model: int=256,
        d_fine_model: int=128,
        n_coarse_layers: int=6,
        n_fine_layers: int=4,
        n_heads: int=8,
        backbone_name: str='resnet18',
        matching_name: str='sinkhorn',
        match_threshold: float=0.2,
        window: int=5,
        border: int=1,
        sinkhorn_iterations: int=50,
    ):
        super().__init__()

        self.backbone = ResUNet(backbone_name, d_coarse_model, d_fine_model)
        self.position2d = PositionEmbedding2D(d_coarse_model)
        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window**2)

        self.coarse_matching = LoFTRModule(
            d_coarse_model, n_heads=n_heads,
            layer_names=['self', 'cross'] * n_coarse_layers
        )
        
        self.fine_matching = LoFTRModule(
            d_fine_model, n_heads=n_heads,
            layer_names=['sefl', 'cross'] * n_fine_layers
        )

        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(2*d_fine_model, d_fine_model, bias=True)

        self.border = border
        self.window = window
        self.num_iter = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.matching_name = matching_name
        self.step_coarse = self.backbone.scaling_step_coarse
        self.step_fine = self.backbone.scaling_step_fine

        if matching_name == 'sinkhorn':
            bin_score = nn.Parameter(torch.tensor(1.))
            self.register_parameter("bin_score", bin_score)
        
    def detect(self, samples: NestedTensor):
        images, masks = samples.decompose()
        coarse_featmap, fine_featmap = self.backbone(images)
        masks = F.interpolate(
            masks[None].float(),
            size=coarse_featmap.shape[-2:]
        ).to(torch.bool)[0]

        position = self.position2d(coarse_featmap)
        coarse_featmap = coarse_featmap + position

        return coarse_featmap, fine_featmap, masks
    
    def unfold_within_window(self, featmap):
        scale = self.step_coarse - self.step_fine
        stride = int(math.pow(2, scale))

        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window//2
        )

        featmap_unfold = rearrange(
            featmap_unfold,
            "B (C MM) L -> B L MM C",
            MM=self.window ** 2
        )
        return featmap_unfold


    def forward(
        self,
        samples0: NestedTensor,
        samples1: NestedTensor,
        gt_matches: torch.Tensor=None,
    ):
        device = samples0.device

        coarse_featmap0, fine_featmap0, mask0 = self.detect(samples0) #[B, d_model, H, W] & [B, 1, H, W]
        coarse_featmap1, fine_featmap1, mask1 = self.detect(samples1)

        desc0 = coarse_featmap0.flatten(2).transpose(1, 2)
        desc1 = coarse_featmap1.flatten(2).transpose(1, 2)
        desc_mask0 = mask0.flatten(1)
        desc_mask1 = mask1.flatten(1)

        mdesc0, mdesc1 = self.coarse_matching(
            desc0, desc1, mask0=desc_mask0, mask1=desc_mask1
        )

        mdesc0 = mdesc0 / (mdesc0.shape[-1] ** .5)
        mdesc1 = mdesc1 / (mdesc1.shape[-1] ** .5)

        scores = torch.einsum('bnd, bmd->bnm', mdesc0, mdesc1)

        if self.matching_name == 'sinkhorn':
            scores.masked_fill(
                ~(desc_mask0[..., None] * desc_mask1[:, None]),
                float('inf')
            )
            scores = log_optimal_transport(
                scores, self.bin_score, iters=self.num_iter
            )
        else:
            scores = scores / 0.1
            inf = torch.zeros_like(scores)
            valid = desc_mask0[..., None] * desc_mask1[:, None]
            inf[~valid] = -1e9
            scores = scores + inf

            scores_col = torch.softmax(scores, dim=1)
            scores_row = torch.softmax(scores, dim=2)
            scores = scores_col * scores_row

        if gt_matches is None:
            # For test step
            B =  coarse_featmap0.shape[0]
            hc0, wc0 = coarse_featmap0.shape[-2:]
            hc1, wc1 = coarse_featmap1.shape[-2:]
            
            axes_lengths = {
                'B': B, 'H0': hc0, 'W0': wc0, 'H1': hc1, 'W1':wc1
            }
            
            matches, mconf = mutual_nearest_neighbor_match(
                scores, axes_lengths,
                border=self.border,
                match_threshold=self.match_threshold,
                mask0=mask0, mask1=mask1,
                use_bins=(self.matching_name=='sinkhorn')
            )
            if len(matches) == 0:
                return {
                    'scores': scores,
                    'matches': matches,
                    'mconf': mconf,
                    'std': torch.empty(0, device=device),
                    'mkpts0': torch.empty((0, 2), device=device),
                    'mkpts1': torch.empty((0, 2), device=device)
                }
        else:
            # For Train step
            matches = gt_matches
            mconf = matches.new_tensor(1.).expand(matches.shape[:-1])
        
        fine_featmao0_unfold = self.unfold_within_window(fine_featmap0)
        fine_featmao1_unfold = self.unfold_within_window(fine_featmap1)

        local_desc = torch.cat([
            fine_featmao0_unfold[matches[:, 0], matches[:, 1]],
            fine_featmao1_unfold[matches[:, 0], matches[:, 2]]
        ], dim=0)

        center_desc = repeat(torch.cat([
            mdesc0[matches[:, 0], matches[:, 1]],
            mdesc1[matches[:, 0], matches[:, 2]]
            ], dim=0), 
            'N C -> N WW C', 
            WW=self.window**2)
        
        center_desc = self.proj(center_desc)

        local_desc = torch.cat([local_desc, center_desc], dim=-1)
        local_desc = self.merge(local_desc)
        
        local_position = self.position1d(local_desc)
        local_desc = local_desc + local_position

        desc0, desc1 = torch.chunk(local_desc, 2, dim=0)
        mdesc0, mdesc1 = self.fine_matching(desc0, desc1)

        c = self.window ** 2 // 2
        sim_matrix = torch.einsum('nd, nmd->nm', mdesc0[:, c, :], mdesc1)
        softmax_temp = 1. / mdesc0.shape[-1] ** .5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
        heatmap = heatmap.view(-1, self.window, self.window)

        coords_norm = spatial_softargmax_2d(heatmap[None], True)[0]
        grids_norm = create_meshgrid(
            self.window, self.window, True, device
        ).reshape(1, -1, 2)

        var = torch.sum(grids_norm ** 2 * heatmap.view(-1, self.window**2, 1), dim=1) - coords_norm**2
        std = torch.sum(torch.sqrt(torch.clmap(var, min=1e-10)), dim=-1)

        
        # Update wirh matches in original image resolution
        coarse_mkpts0 = ind2coord(matches[:, 1], coarse_featmap0.shape[-1])
        coarse_mkpts1 = ind2coord(matches[:, 2], coarse_featmap0.shape[-1])
        
        mkpts0 = upscale(coarse_mkpts0, self.step_coarse)
        mkpts1 = upscale(coarse_mkpts1, self.step_coarse)

        expected_coords = coords_norm * float(self.window // 2)
        expected_coords = upscale(expected_coords, self.step_fine)

        mkpts1 = mkpts1 + expected_coords

        return {
            'scores': scores,
            'matches': matches,
            'mconf': mconf,
            'std': std,
            'mkpts0': mkpts0,
            'mkpts1': mkpts1
        }

