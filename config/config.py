import argparse

import torch.nn as nn
from config.model_config import models
from config.scheduler_config import schedulers
from config.optimizer_config import optimizers

import argparse

parser = argparse.ArgumentParser(description = 'Train and Evaluate Model')

parser.add_argument("--model"           , type=str, help="model to use for training", default="resnet18")
parser.add_argument("--criterion"       , type=str, help="loss function to use for training", default="cross_entropy")
parser.add_argument("--optimizer"       , type=str, help="optimizer to use for training", default="adam")
parser.add_argument("--scheduler"       , type=str, help="scheduler to use for training", default="cosine")

parser.add_argument('--batch_size'      , type=int, default=64)
parser.add_argument('--epochs'          , type=int, default=10)
parser.add_argument('--num_workers'     , type=int, default=4)

# distributed training parameters
parser.add_argument('--dist_backend'    , type=str, default='nccl')
parser.add_argument('--dist_url'        , type=str, default='tcp://')
parser.add_argument('--num_nodes'       , type=int, default=1)
parser.add_argument('--node_rank'       , type=int, default=0)
parser.add_argument('--world_size'      , type=int, default=1)

criterions = {
    "cross_entropy" : nn.CrossEntropyLoss,
    "bce"          : nn.BCEWithLogitsLoss,
    "bce_logits"   : nn.BCELoss,
    "mse"          : nn.MSELoss,
    "l1"           : nn.L1Loss,
    "custom"       : "custom",
}

def config():
    args, rest = parser.parse_known_args()
    args.criterion = criterions[args.criterion]

    args.model, model_parser = models[args.model]
    args.model_args, rest = model_parser(rest)

    args.optimizer, optimizer_parser = optimizers[args.optimizer]
    args.optimizer_args, rest = optimizer_parser(rest)

    args.scheduler, scheduler_parser = schedulers[args.scheduler]
    args.scheduler_args, rest = scheduler_parser(rest)

    if len(rest) != 0:
        raise ValueError(f"Unrecognized arguments: {rest}")
    return args