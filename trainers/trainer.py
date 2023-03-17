import torch
# import multiprocessing as mp
import torch.multiprocessing as mp
from datasets.color import *
import random
import os
import time
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from datasets.multiDatasets import multiDatasets
import torchvision.transforms as transforms

class Trainer:
    def __init__(self, **kargs) -> None:
        self.world_size = kargs.get('world_size', 1)
        self.num_workers = kargs.get('num_workers', 4)

        self.model = kargs.get('model')
        self.dataset = kargs.get('dataset')
        self.means = kargs.get('means')
        self.stds = kargs.get('stds')
        self.datapath = kargs.get('datapath')
        self.transforms = kargs.get('transforms')
        self.criterion = kargs.get('criterion')
        self.optimizer = kargs.get('optimizer')
        self.scheduler = kargs.get('scheduler')
        self.model_args = kargs.get('model_args')
        self.optimizer_args = kargs.get('optimizer_args')
        self.scheduler_args = kargs.get('scheduler_args')
        self.seed = kargs.get('seed', 0)

        self.batch_size = kargs.get('batch_size', 128)
        self.epochs = kargs.get('epochs', 100)
        self.log_interval = kargs.get('log_interval', 10)
        self.save_interval = kargs.get('save_interval', 10)
        self.save_path = kargs.get('save_path', './')
        self.name = kargs.get('name', 'default')


        self.num_nodes = kargs.get('num_nodes', 1)
        self.node_rank = kargs.get('node_rank', 0)

        self.use_amp = kargs.get('use_amp', False)
        self.mode = kargs.get('mode', 'train')

        self.ngpus_per_nodes = torch.cuda.device_count()
        self.world_size =  self.num_nodes * self.ngpus_per_nodes
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.batch_size = self.batch_size // self.world_size
        self.distributed = self.world_size > 1
        if 'MASTER_ADDR' not in os.environ.keys():
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '12701'

        self.dist_backend = 'nccl'
        self.dist_url = 'tcp://' + os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT']

        self.best_acc = 0 

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

        if torch.cuda.is_available():
            self.gpu    = gpu % self.ngpus_per_nodes
            self.device = torch.device(self.gpu)
            if self.distributed:
                self.local_rank = self.gpu
                self.rank = self.node_rank * self.ngpus_per_nodes + self.gpu
                time.sleep(self.rank * 0.1) # prevent port collision
                print(f'rank {self.rank} is running...')
                dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                        world_size=self.world_size, rank=self.rank)
                dist.barrier()
                self.setup_for_distributed(self.is_main_process())
        else:
            self.device = torch.device('cpu')
        
        # Set Random Seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed) # if use multi-GPU
            cudnn.deterministic = True
            cudnn.benchmark = False
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
        
        # Define Dataset
        if len(self.dataset) == 1:
            self.train_dataset = self.dataset[0](train=True, root=self.datapath, transform=transforms.Compose([self.transforms, transforms.ToTensor(), transforms.Normalize(self.means[0], self.stds[0])]) , download=True)
            self.test_dataset = self.dataset[0](train=False, root=self.datapath, transform=transforms.Compose([self.transforms, transforms.ToTensor()]), download=True)
        else:
            self.dataset = [self.dataset[i](train=True, root=self.datapath, transform=self.transforms.Compose([self.transforms, transforms.ToTensor(), transforms.Normalize(self.means[i], self.stds[i])])) for i in range(len(self.dataset))]
            self.train_dataset = multiDatasets(self.dataset, train=True, transform=self.transforms)
            self.test_dataset = multiDatasets(self.dataset, train=False, transform=self.transforms)

        # Define Model
        self.model = self.model(**self.model_args)
        self.model = self.model.to(self.device)
        self.model_without_ddp = self.model
        self.criterion = self.criterion() if self.criterion is not "custom" else self.model.criterion
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], find_unused_parameters=True)
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_args)
        
        self.setup()

    def setup(self):
        raise NotImplementedError
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        state = {
            'epoch': self.epoch,
            'arch': self.model.__class__.__name__,
            'state_dict': self.model.state_dict(),
            'best_acc'  : self.best_acc,
            'optimizer' : self.optimizer.state_dict(),
            'scheduler' : self.scheduler.state_dict(),
        }
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            self.epoch    = 0
            self.best_acc = 0
        else:
            print("=> loading checkpoint '{}'".format(checkpoint))
            self.epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_acc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint, checkpoint['epoch']))
            
    def train(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
        self.scheduler.step(epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        self.test_acc = 100. * correct / len(self.test_loader.dataset)
        test_loss /= len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(self.test_loader.dataset), self.test_acc))
        