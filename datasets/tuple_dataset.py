import random
import numpy as np
import torch
import torch.nn.functional as F
from einops.einops import rearrange
from scipy.spatial.distance import cdist
from common import Image
from common.functions import *

class TupleDataset:
    def __init__(self, item_dataset, tuples):
        self.item_dataset = item_dataset
        self.tuples = tuples

    def tuple_transform(self, tuple_):
        return tuple_

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        items = tuple(self.item_dataset[i] for i in self.tuples[idx])

        return self.tuple_transform(items)


class ConvisDataset:
    def __init__(
        self, item_dataset, pairs, scale, bins=False, th=4,
    ):
        self.item_dataset = item_dataset
        self.pairs = pairs
        self.scale = scale
        self.bins = bins
        self.th = th

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair0, pair1 = self.pairs[idx]

        image0: Image = self.item_dataset[pair0]
        image1: Image = self.item_dataset[pair1]

        window = 2 ** self.scale
        Hc0 = image0.shape[0] // window
        Wc0 = image0.shape[1] // window
        Hc1 = image1.shape[0] // window
        Wc1 = image1.shape[1] // window

        grid_points0 = grid_positions(Hc0, Wc0).T
        grid_points1 = grid_positions(Hc1, Wc1).T

        ipoints0 = upscale(grid_points0, self.scale)
        ipoints1 = upscale(grid_points1, self.scale)

        ipoints0_r = image1.project(image0.unproject(ipoints0.T)).T
        dists = cdist(ipoints0_r.numpy(), ipoints1.numpy())
        dists[np.isnan(dists)] = float('inf')

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)
        min1v = np.min(dists, axis=1)
        mask = min1v < self.th
        min1f = min2[min1v < self.th]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        mids1 = np.intersect1d(min1f, xx)
        mids0 = min1[mids1]

        n_pts0 = image0.length // window**2
        n_pts1 = image1.length // window**2

        assignment = torch.zeros((n_pts0, n_pts1)).bool()
        assignment[mids0, mids1] = True

        mask0 = F.interpolate(
            image0.mask[None].unsqueeze(0).float(),
            size=(Hc0, Wc0),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).flatten(1).bool()

        mask1 = F.interpolate(
            image1.mask[None].unsqueeze(0).float(),
            size=(Hc1, Wc1),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).flatten(1).bool()

        mask_good = (mask0[..., None] * mask1[:, None]).squeeze(0)
        assignment = assignment * mask_good

        if assignment.sum() == 0:
            return None

        if self.bins:
            n1, n2 = assignment.shape
            alpha = assignment.new_tensor(False)
            assignment = torch.cat([
                torch.cat([assignment, alpha.expand(n1, 1)], dim=-1),
                torch.cat([alpha.expand(1, n2), alpha.expand(1, 1)], dim=-1)
            ], dim=0)

            bins0 = (~(assignment.cumsum(1, dtype=torch.int).bool()))
            bins1 = (~(assignment.cumsum(0, dtype=torch.int).bool()))
            assignment[:, -1] = bins0[:, -1]
            assignment[-1, :] = bins1[-1, :]
            assignment[-1, -1] = False

        return image0, image1, assignment
