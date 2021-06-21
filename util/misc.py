import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List
from collections import  defaultdict, deque

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if average:
            values /= world_size
        reduce_dict = {k: v for k, v in zip(names, values)}
    return reduce_dict

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    import builtins as __builtins__
    builtins_print = __builtins__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtins_print(args, **kwargs)
    __builtins__.print = print

def init_distributed_mode(args):
    if "RANK" in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    torch.distributed.init_process_group(
        backend=args.dist_backend,init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank 
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank ==0)
