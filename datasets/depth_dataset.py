import math
import torch
import os
import json
import random
import numpy as np
import imageio
import typing
import h5py

from torch.utils.data import Dataset
from common import Image, NpArray
from datasets.tuple_dataset import ConvisDataset
from datasets.limited_dataset import LimitedConcatDataset

def _base_image_name(img_name):
    return img_name.split('.')[0]

def _crop(image: Image, crop_size):
    return image.scale(crop_size).pad(crop_size)

class ImageSet:
    def __init__(
        self,
        json_data,
        crop_size,
        root,
        do_crop=True,
    ):
        def maybe_add_root(path):
            if os.path.exists(path) and os.path.listdir(path):
                return path
            rooted_path = os.path.join(root, path)

            if not os.parh.exists(rooted_path) or not os.path.listdir(rooted_path):
                raise FileNotFoundError(
                    f"Couldn't find a directory as {path} or"
                    f"{rooted_path}"
                )

            return rooted_path
        self.image_path = maybe_add_root(json_data['image_path'])
        self.calib_path = maybe_add_root(json_data['calib_path'])
        self.depth_path = maybe_add_root(json_data['depth_path'])

        self.id2name = json_data['images']
        self.crop_size = crop_size
        self.do_crop = do_crop

    def _get_bitmap_path(self, image_name):
        base_name = _base_image_name(image_name)
        return os.path.join(self.image_path, base_name + '.jpg')
    
    def _get_bitmap(self, image_name):
        bitmap_path = self._get_bitmap_path(image_name)

        bitmap = imageio.imread(bitmap_path)
        bitmap = bitmap.astype(np.float32) / 255.

        return torch.from_numpy(bitmap).permute(2, 0, 1)
    
    def _get_depth(self, image_name):
        h5_name = _base_image_name(image_name) + '.h5'
        depth_path = os.path.join(self.depth_path, h5_name)

        h5 = h5py.File(depth_path, 'r')
        depth = h5['depth'][:].astype(np.float32)
        depth[depth == 0.] = float('NaN')

        return torch.from_numpy(depth).unsqueeze(0)
    
    def _get_KRT(self, image_name):
        calib_path = os.path.join(
            self.calib_path, f'calibration_{image_name}.h5'
        )

        values = []
        with h5py.File(calib_path, 'r') as calib_file:
            for f in ['K', 'R', 'T']:
                v = torch.from_numpy(calib_file[f][()]).to(torch.float32)
                values.append(v)
            
        return values

    def __getitem__(self, idx):
        image_name = self.id2name[idx]

        image = Image(
            *self._get_KRT(image_name), 
            self._get_bitmap(image_name), 
            self._get_depth(image_name),
        )

        if self.do_crop:
            return _crop(image, self.crop_size)
        else:
            return image


class SceneCovis(ConvisDataset):
    def __init__(
        self, 
        json_data, 
        crop_size,
        root, 
        scale,
        use_bins
    ):
        items = ImageSet(json_data, crop_size, root)
        pairs = json_data['pairs']

        super(SceneCovis, self).__init__(items, pairs, scale, bins=use_bins)

class DepthDataset(LimitedConcatDataset):
    def __init__(
        self, json_path, crop_size=(480, 640), scale=3,
        use_bins=False, limit=None, shuffle=False, warn=True
    ):
        self.crop_size = crop_size
        self.scale = scale
        self.use_bins = use_bins

        with open(json_path,'r') as json_file:
            json_data = json.load(json_file)

        root_path, _ = os.path.split(json_path)
        scene_datasets = []
        for scene in json_data:
            scene_datasets.append(SceneCovis(
                json_data[scene],
                crop_size,
                root_path,
                scale,
                use_bins
            ))
        super(DepthDataset, self).__init__(
            scene_datasets,
            limit=limit,
            shuffle=shuffle,
            warn=warn
            )
    
    @staticmethod
    def cpllate_fn(batch):
        batch = list(filter(lambda b: b is not None, batch))

        bitmaps = []
        masks = []
        images = []
        assignments = []

        for image0, _, _ in batch:
            images.append(image0)
            bitmaps.append(image0.bitmap)
            masks.append(image0.mask)
        
        for image1, _, _ in batch:
            images.append(image1)
            bitmaps.append(image1.bitmap)
            masks.append(image1.mask)
        
        for _, _, assign in batch:
            assignments.append(assign)
        
        bitmaps = torch.stack(bitmaps)
        masks = torch.stack(masks)
        images = np.array(images)
        assignments = torch.stack(assignments)

        return PinnableBatch(bitmaps, masks, assignments, images)

class PinnableBatch(typing.NamedTuple):
    bitmaps: torch.Tensor
    masks: torch.Tensor
    assignments: torch.Tensor
    images: NpArray[Image]

    def pin_memory(self):
        bitmaps = self.bitmaps.pin_memory()
        masks = self.masks.pin_memory()
        assignments = self.assignments.pin_memory()
        images = [im.pin_memory() for im in self.images]

        return PinnableBatch(bitmaps, masks, assignments, images)
    
    def to(self, * args, **kwargs):
        bitmaps = self.bitmaps.to(* args, **kwargs)
        masks = self.masks.to(* args, **kwargs)
        assignments = self.assignments.to(* args, **kwargs)
        images = self.images.copy()
        for i in range(images.size):
            images.flat[i] = images.flat[i].to(* args, **kwargs)
        
        return PinnableBatch(bitmaps, masks, assignments, images)

def build_depth(
    root,
    crop_size=(480, 640),
    scale=3,
    use_bins=False,
    train_limit=5000,
    test_limit=1000
):
    train_dataset = DepthDataset(
        os.path.join(root, 'train/dataset.json'),
        crop_size=crop_size,
        scale=scale,
        use_bins=use_bins,
        train_limit=train_limit,
        shuffle=True
    )

    test_dataset = DepthDataset(
        os.path.join(root, 'test/dataset.json'),
        crop_size=crop_size,
        scale=scale,
        use_bins=use_bins,
        test_limit=test_limit,
        shuffle=True
    )

    return train_dataset, test_dataset










