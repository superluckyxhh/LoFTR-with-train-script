import argparse
import os
import sys
import random
import json
import numpy as np
import torch

from typing import Iterable, Optional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (DataLoader, BatchSampler, RandomSampler,
                              SequentialSampler, DistributedSampler)
# import util
# from models import build_model #TODO:
# from datasets import build_dataset #TODO:
# from losses import build_criterion #TODO:
from common.error import NoGradientError
# from ... import build_chunk_dataset #TODO:
from common.logger import Logger, MetricLogger, SmoothedValue
from common.functions import *
from common.nest import NestedTensor
from configs import dynamic_load
import cv2
# from util.plotutils import save_matches

DEV = torch.device('cuda' if torch.is_available() else 'cpu')


def train(
    epoch: int, loader: Iterable, model: torch.nn.Module,
    criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
    max_norm=0., print_freq=10., tb_logger=None
):
    model.train()
    criterion.train()

    logger = MetricLogger(delimiter=' ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for batched_data in logger.log_every(loader, print_freq, header):
        bitmaps, masks, assignments, images = batched_data.to(DEV)
        bitmaps0, bitmaps1 = torch.chunk(bitmaps, 2, dim=0)
        masks0, masks1 = torch.chunk(masks, 2, dim=0)
        images0, images1 = np.split(images, 2)

        samples0 = NestedTensor(bitmaps0, masks0)
        samples1 = NestedTensor(bitmaps1, masks1)

        matches = assignments_to_matches(assignments)
        targets = {
            'assignments': assignments,
            'matches': matches,
            'images0': images0,
            'images1': images1,
        }

        if matches.shape[0] <= 0:
            print('Skip Non-enough matches [<=0].')
            continue

        preds = model(samples0, samples1, matches)

        try:
            loss_dict = criterion(preds, targets)
            loss = loss_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                )
            optimizer.step()
        except NoGradientError:
            print('Got No Gradient Error')
            sys.exit(1)

        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_item = {
        k: v.item() for k, v in loss_dict_reduced.items()
        }

        logger.update(**loss_dict_reduced_item)
        logger.update(lr=optimizer.param_groups[0]['lr'])
        if tb_logger is not None:
            if util.is_main_process():
                tb_logger.add_scalers(loss_dict_reduced, prefix='train')

    logger.synchronize_between_processes()
    print('Average stats:', logger)
    return {k: meter.global_avg for k, meter in logger.meters.items()}

@torch.no_grad()
def test(
    loader: Iterable, model: torch.nn.Module,
    metrics, print_freq=10., tb_logger=None
):
    model.eval()

    logger = MetricLogger(delimiter=' ')
    header = 'Test'

    for batched_data in logger.log_every(loader, print_freq, header):
        bitmaps, masks, assignments, images = batched_data.to(DEV)
        bitmaps0, bitmaps1 = torch.chunk(bitmaps, 2, dim=0)
        masks0, masks1 = torch.chunk(masks, 2, dim=0)
        images0, images1 = np.split(images, 2)

        samples0 = NestedTensor(bitmaps0, masks0)
        samples1 = NestedTensor(bitmaps1, masks1)

        matches = assignments_to_matches(assignments)
        targets = {
            'assignments': assignments,
            'matches': matches,
            'images0': images0,
            'images1': images1,
        }

        if matches.shape[0] == 0:
            print('Found No matches in batch, continue.')
            continue

        preds = model(samples0, samples1)
        stats = metrics(preds, targets)

def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', n_params)
    model = model.to(DEV)

    criterion, metrics = build_criterion(args)
    criterion = criterion.to(DEV)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids={args.gpu})
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_schedular.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6
    )
    train_dataset, test_dataset = build_dataset(args.data_name, args)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    batch_train_sampler = BatchSampler(
        train_sampler, args.batch_size, drop_last=True
    )

    dataloader_kwargs = {
        'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': 4,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_train_sampler,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    if args.load is not None:
        state_dict = torch.load(args.load, map_location='cpu')
        model_without_ddp.load_state_dict(state_dict['model'])

    save_name = f'{args.backbone_name}-{args.matching_name}'
    save_name += f'_dim{args.d_coarse_model}-{args.d_fine_model}'
    save_name += f'_depth{args.n_coarse_model}-{args.d_fine_model}'

    save_path = os.path.join(args.save_path, save_name)
    os.makedirs(save_path, exist_ok=True)
    if util.is_main_process():
        tensorboard_logger = Logger(save_path)
    else:
        tensorboard_logger = None

    print('Start Training...')
    for epoch in range(args.n_epochs):
        epoch = epoch + args.epoch_offset

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            max_norm=args.clip_max_norm,
            print_freq = args.log_interval,
            tb_logger=tensorboard_logger
        )
        scheduler.step()

        if epoch % args.save_interval == 0 or epoch == args.n_epochs - 1:
            if util.is_main_process():
                torch.save({
                    'model': model_without_ddp.state_dict()
                }, f'{save_path}/model-epoch{epoch}.pth')
        test_stats = {}
        log_stats = {
            'epoch': epoch,
            'n_params': n_params,
            'data_name': args.data_name,
            **{f'train_{k}':v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }
        with open(f'{save_path}/train.log', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='imcnet_config')
    global_cfgs = parser.parse_args()

    args = dynamic_load(global_cfgs.config_name)
    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )
    print(prm_str + '\n')
    print('=='*40 + '\n')

    main(args)