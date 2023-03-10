import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from datasets.color import *
from torchvision.datasets import CIFAR10, CIFAR100
import random
import os
import time
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np

datasets = {
    'MNIST' : MNIST,
    'FashionMNIST' : FashionMNIST,
    'NotMNIST' : NotMNIST,
    'CUB200' : CUB200,
    'Flowers102' : Flowers102,
    'Imagenet_R' : Imagenet_R,
    'SVHN' : SVHN,
    'TinyImageNet' : TinyImageNet,
    'CIFAR10' : CIFAR10,
    'CIFAR100' : CIFAR100,
}

class Trainer:
    def __init__(self, **kargs) -> None:
        self.world_size = kargs.get('world_size', 1)
        self.num_workers = kargs.get('num_workers', 4)

        self.model = kargs.get('model')
        self.dataset = kargs.get('dataset')
        self.criterion = kargs.get('criterion')
        self.optimizer = kargs.get('optimizer')
        self.scheduler = kargs.get('scheduler')
        self.model_args = kargs.get('model_args')
        self.optimizer_args = kargs.get('optimizer_args')
        self.scheduler_args = kargs.get('scheduler_args')

        self.num_nodes = kargs.get('num_nodes', 1)
        self.node_rank = kargs.get('node_rank', 0)
        self.ngpus_per_nodes = torch.cuda.device_count()
        self.world_size =  self.num_nodes * self.ngpus_per_nodes

        self.distributed = self.world_size > 1
        if 'MASTER_ADDR' not in os.environ.keys():
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '12701'

        self.dist_backend = 'nccl'
        self.dist_url = 'tcp://'
        pass

    def run(self):
        # Distributed Launch
        if self.ngpus_per_nodes > 1:
            mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
        else:
            self.main_worker(0)

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def get_rank(self):
        if self.distributed:
            return dist.get_rank()
        return 0
    
    def get_world_size(self):
        if self.distributed:
            return dist.get_world_size()
        return 1
    
    def main_worker(self, gpu) -> None:
        # Distributed Launch
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            self.rank = self.node_rank * self.ngpus_per_nodes + self.gpu
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            dist.barrier()
            self.setup_for_distributed(self.is_main_process())
        
        # Set Random Seed
        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
            cudnn.benchmark = False
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
        
        # Define Model
        self.model = self.model(**self.model_args)
        self.model_without_ddp = self.model
        self.criterion = self.criterion() if self.criterion is not "custom" else self.model.criterion
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu])
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_args)
        
        self.work()

    def work(self, *args, **kargs):
        raise NotImplementedError
