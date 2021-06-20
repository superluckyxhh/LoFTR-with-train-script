import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NoGradientError

class MatchingCriterion(nn.Module):
    def __init__(self, weights: list, eps=1e-10):
        super().__init__()

        self.weights = weights
        self.eps = eps

    def compute_assign_loss(
        self,
        scores: torch.Tensor,
        assignments: torch.Tensor,
        loss_type: str='dual_softmax'
    ):
        batch_dim = assignments.shape[0]

        if loss_type == 'dual_softmax':
            scores = -torch.log(scores + self.eps)
        elif loss_type == 'sinkhorn':
            scores = -scores
        else:
            raise KeyError('Invalid loss type')

        losses = 0
        for b in range(batch_dim):
            assign: torch.Tensor = assignments[b]
            score: torch.Tensor = scores[b]

            if assign.sum() == 0:
                print('NotFound matches, continue')
                raise NoGradientError

            n1, n2 = assign.shape
            if (n1 + 1) == score.shape[0]:
                alpha = assign.new_tensor(False)
                assign = torch.cat([
                    torch.cat([assign, alpha.expand(n1, 1)], dim=-1),
                    torch.cat([alpha.expand(1, n2),
                               alpha.expand(1, 1)], dim=-1)
                ], dim=0)
                bins0 = (~(assign.cumsum(1, dtype=torch.int).bool()))
                bins1 = (~(assign.cumsum(0, dtype=torch.int).bool()))
                assign[:, -1] = bins0[:, -1]
                assign[-1, :] = bins1[-1, :]
                assign[-1, -1] = False

            loss_per_batch = score.masked_select(assign)
            losses += torch.mean(loss_per_batch)

            if not torch.isfinite(losses):
                print('Found bad loss!')
                print(f'[{loss_type}] Print lossL {loss_per_batch}')
                raise NoGradientError

        return losses / float(batch_dim)

    def forward(self, preds, targets):
        coarse_scores, fine_scores = preds
        coarse_assigns, fine_assigns = targets

        coarse_loss = self.compute_assign_loss(
            coarse_scores, coarse_assigns, loss_type='dual_softmax'
        )
        fine_loss = self.compute_assign_loss(
            fine_scores, fine_assigns, loss_type='sinkhorn'
        )
        losses = sum(w * loss for w, loss in zip(
            self.weights, [coarse_loss, fine_loss]
        ))

        return {
            'losses': losses,
            'coarse_loss:': coarse_loss,
            'fine_loss': fine_loss
        }




