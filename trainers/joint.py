import os
import torch
from .trainer import Trainer

class Joint(Trainer):
    def __init__(self, **kwargs):
        super(Joint, self).__init__(**kwargs)
        self.num_tasks = kwargs.get("num_tasks", 10)
        self.reset_optimizer = kwargs.get("reset_optimizer", False)

    def setup(self, *args, **kargs):
        rank = self.get_rank()
        world_size = self.get_world_size()
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.test_sampler)
        
        for epoch in range(self.epoch, self.epochs):
            self.train_sampler.set_epoch(epoch)
            self.test_sampler.set_epoch(epoch)

            self.train(epoch)
            self.test(epoch)
            
            if self.is_main_process():
                if epoch == 0:
                    self.save_checkpoint(os.path.join(self.save_path, 'best.pth'.format(epoch)))
                else:
                    if self.best_acc < self.test_acc:
                        self.best_acc = self.test_acc
                        self.save_checkpoint(os.path.join(self.save_path, 'best.pth'.format(epoch)))
                self.save_checkpoint(os.path.join(self.save_path, 'chechpoint.pth'.format(epoch)))
