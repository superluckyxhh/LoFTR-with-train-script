import torch
from torch import Tensor
from typing import Optional, List
from collections import defaultdict

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, ** kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(*args, **kwargs)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def chunk_two_batches(self, dim=0):
        tensor0, tensor1 = torch.chunk(self.tensors, 2, dim=dim)
        mask0, mask1 = torch.chunk(self.mask, 2, dim=dim)

        nest0 = NestedTensor(tensor0, mask0)
        nest1 = NestedTensor(tensor1, mask1)
        return nest0, nest1

    def decompose(self):
        return self.tensors, self.mask

    def unsqueeze(self, dim):
        tensors = self.tensors.unsqueeze(dim)
        mask = self.mask.unsqueeze(dim)
        return NestedTensor(tensors, mask)

    @property
    def device(self):
        return self.tensors.shape

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=torch.bool, device=device)
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = True
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def collate_fn(batch):
    tensor_list, target_list = list(zip(*batch))

    bitmaps0, masks0 = [], []
    bitmaps1, masks1 = [], []

    for tuple_ in tensor_list:
        bitmaps0.append(tuple_['bitmap0'])
        bitmaps1.append(tuple_['bitmap1'])
        masks0.append(tuple_['mask0'])
        masks1.append(tuple_['mask1'])

    samples0 = nested_tensor_from_tensor_list(bitmaps0)
    samples1 = nested_tensor_from_tensor_list(bitmaps1)

    targets = defaultdict(list)
    for tuple_ in target_list:
        for k, v in tuple_.items():
            targets[k].append(v)

    targets = {k: torch.stack(v) for k, v in targets.items()}

    return (samples0, samples1), targets
