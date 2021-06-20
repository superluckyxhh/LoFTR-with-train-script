import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

def mask_border(m, b, v):
    m[:, :b]=v
    m[:, :, :b]=v
    m[:, :, :, :b]=v
    m[:, :, :, :, :b]=v
    m[:, -b:0]=v
    m[:, :, -b:0]=v
    m[:, :, :, -b:0]=v
    m[:, :, :, :, -b:0]=v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd]= v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0-bd:] = v
        m[b_idx, :, w0-bd:] = v
        m[b_idx, :, :, h1-bd:] = v
        m[b_idx, :, :, h1-bd:] = v

@torch.no_grad()
def matual_nearest_neighbor_match(
    scores,
    axes_lengths: dict,
    border: int,
    match_threshold: float,
    mask0=None,
    mask1=None,
    use_bins=False
):
    if use_bins:
        conf_matrix = scores[:, :-1, :-1].exp()
    else:
        conf_matrix = scores

    mask = conf_matrix > match_threshold
    mask = rearrange(
        mask,
        'B (H0 W0) (H1 W1) -> B H0 W0 H1 W1',
        **axes_lengths
    )
    if mask0 is not None:
        mask_border(mask, border, False)
    else:
        mask_border_with_padding(mask, border, False, mask0, mask1)

    mask = rearrange(
        mask,
        'B H0 W0 H1 W1 -> B (H0 W0) (H1 W1)',
        axes_lengths
    )

    max0 = conf_matrix.max(dim=2, keepdim=True)[0]
    max1 = conf_matrix.max(dim=1, keepdim=True)[0]

    mask = mask * (conf_matrix == max0) * (conf_matrix == max1)

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mconf = conf_matrix[b_ids, i_ids, j_ids]
    matches = torch.stack([b_ids, i_ids, j_ids]).T

    return matches[mconf != 0], mconf[mconf != 0]

def assignments_to_matches(assignments, use_bins=False):
    if use_bins:
        assignments = assignments[:, :-1, :-1]
    mask = assignments > 0
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mids = torch.stack([b_ids, i_ids, j_ids]).T

    return mids

def grid_positions(h, w, device='cpu', matrix=False):
    rows = torch.arange(
        0, h, device=device
    ).view(-1, 1).float().repeat(1, w)

    cols = torch.arange(
        0, w, device=device
    ).view(1, -1).float().repeat(h, 1)

    if matrix:
        return torch.stack([cols, rows], dim=0)
    else:
        return torch.cat([cols.view(1, -1), rows.view(1, -1)], dim=0)

def upscale(coord, scaling_steps):
    for _ in range(scaling_steps):
        coord = coord * 2.0 + 0.5
    return coord

def downscale(coord, scaling_steps):
    for _ in range(scaling_steps):
        coord = (coord - .5) / 2.
    return coord

def normalize(coord, h, w):
    c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord.device).float()
    coord_norm = (coord - c) / c
    return coord_norm

def denormalize(coord_norm, h, w):
    c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord_norm.device)
    coord = coord_norm * c + c
    return coord

def ind2coord(ind, w):
    ind = ind.unsqueeze(-1)
    x = ind % w
    y = ind // w
    coord = torch.cat([x, y], -1).float()
    return coord

def test_ind2coord(assignments, w, use_bins=True):
    if use_bins:
        assignments = assignments[:, :-1, :-1]
    mask = assignments > 0
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mids = torch.stack([b_ids, i_ids, j_ids]).T

    mkpts0 = ind2coord(i_ids, w)
    mkpts1 = ind2coord(j_ids, w)

    return mkpts0, mkpts1









