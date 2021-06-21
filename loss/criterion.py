import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gemo.homographies as homo
from common import Image, NpArray
from common import NoGradientError

class MatchingCriterion(nn.Module):
    def __init__(
        self, 
        data_name: str,
        match_type: str='dual_doftmax',
        dist_thresh: float=5,
        weights: list=[1., 1.], 
        eps=1e-10
    ):
        super().__init__()

        self.data_name = data_name
        self.match_type = match_type
        self.dist_thresh = dist_thresh
        self.ws = weights
        self.eps = eps
    
    def set_weight(self, std, mask=None, regularizer=0.):
        inverse_std = 1. / torch.clamp(std + regularizer, min=self.eps)
        weight = inverse_std / torch.mean(inverse_std)
        weight = weight.detach()

        if mask is not None:
            weight = weight.masked_select(mask.bool())
            weight /= (torch.mean(weight) + self.eps)
        
        return weight

    def coarse_loss(self, preds, targets):
        assignments = targets['assignments']
        batch_dim = assignments.shape[0]

        scores = preds['scores']
        if self.match_type == 'sinkhorn':
            preds_matrix = -scores
        else:
            preds_matrix = -torch.log(scores + self.eps)
        
        loss = 0

        for b in range(batch_dim):
            assign_matrix: torch.Tensor = assignments[b]
            p_matrix: torch.Tensor = preds_matrix[b]
            n_matches = assign_matrix.sum()
            if n_matches == 0:
                print('None matches, continue')
                continue

            loss_per_batch = p_matrix.masked_select(assign_matrix)
            loss += torch.mean(loss_per_batch)
            if not torch.isfinite(loss):
                print('Found bad matches')
                raise NoGradientError

        return loss / float(batch_dim)
    
    # def compute_dist_within_homo(self, mkpts0, mkpts1, homograph):
    #     gt_masked = homo.warp_points(mkpts0, homograph)
    #     dist = torch.norm(mkpts1 - gt_masked, dim=-1)

    #     return dist
    
    def compute_dist_within_images(
        self, mkpts0, mkpts1, image0: Image, image1: Image
    ):
        mkpts0_r = image1.project(image0.unproject(mkpts0.T)).T
        dist = torch.norm(mkpts1, mkpts0, dim=-1)

        return dist
    
    def fine_loss(self, preds, targets):
        assignments = targets['assignments']
        batch_dim = assignments.shape[0]
        matches = targets['matches']

        loss = 0

        for b in range(batch_dim):
            batch_mask = matches[:, 0] == b
            n_matches = batch_mask.sum()
            if n_matches == 0:
                print('Found zero matches, continue')
                continue
                
            mkpts0 = preds['mkpts0'][batch_mask]
            mkpts1 = preds['mkpts1'][batch_mask]
            std = preds['std'][batch_mask]

            image0 = targets['images0'][b]
            image1 = targets['images1'][b]

            dist = self.compute_dist_within_images(
                mkpts0, mkpts1, image0, image1
            )
        
        weight = self.set_weight(std)
        loss += torch.mean(weight * dist)
        if not torch.isfinite(loss):
            print('Found bad matches')
            raise NoGradientError
        
        return loss / float(batch_dim)
    

    def forward(self, preds, targets):

        coarse_loss = self.coarse_loss(
            preds, targets
        )
        fine_loss = self.fine_loss(
            preds, targets
        )
        losses = self.ws[0] * coarse_loss + self.ws[1] * fine_loss

        loss_dict = {
            'losses': losses,
            'coarse_loss:': coarse_loss,
            'fine_loss': fine_loss
        }

        return loss_dict