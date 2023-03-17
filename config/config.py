import argparse

import os
import torch
import torch.nn as nn
from config.model_config import models
from config.scheduler_config import schedulers
from config.optimizer_config import optimizers
from config.dataset_config import datasets
from config.transforms_config import transforms
from config.trainer_config import trainers
import torchvision

import argparse

parser = argparse.ArgumentParser(description = 'Train and Evaluate Model')

sub_parser = parser.add_subparsers(dest='mode', help='mode to run in', required=True)

train = sub_parser.add_parser("train", help='train a model')

train.add_argument("--trainer"         , type=str, help="trainer to use for training", default="joint")
train.add_argument("--model"           , type=str, help="model to use for training", default="resnet18")
train.add_argument("--criterion"       , type=str, help="loss function to use for training", default="cross_entropy")
train.add_argument("--optimizer"       , type=str, help="optimizer to use for training", default="adam")
train.add_argument("--scheduler"       , type=str, help="scheduler to use for training", default="cosine")
train.add_argument("--dataset"         , type=str, help="dataset to use for training", default=["CIFAR10"], nargs='+')
train.add_argument("--datapath"        , type=str, help="dataset arguments to use for training", default="./data")
train.add_argument("--transforms"      , type=str, help="transforms to use for training", default=[], nargs='+')

train.add_argument('--use_amp'         , action='store_true', help='use automatic mixed precision')
train.add_argument('--batch_size'      , type=int, default=64)
train.add_argument('--epochs'          , type=int, default=10)
train.add_argument('--num_workers'     , type=int, default=4)
train.add_argument('--seed'            , type=int, default=1)
train.add_argument('--log_interval'    , type=int, default=10)
train.add_argument('--save_interval'   , type=int, default=10)
train.add_argument('--save_path'       , type=str, default="models")

# distributed training parameters
train.add_argument('--dist_backend'    , type=str, default='nccl')
train.add_argument('--dist_url'        , type=str, default='tcp://')
train.add_argument('--num_nodes'       , type=int, default=1)
train.add_argument('--node_rank'       , type=int, default=0)
train.add_argument('--world_size'      , type=int, default=1)

load  = sub_parser.add_parser("load", help='load a model')
load.add_argument("--path", type=str, help="path to model to load")

evaluation = sub_parser.add_parser("evaluation", help='evaluate a model')
evaluation.add_argument("--path", type=str, help="path to model to evaluate")

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
    if args.mode == "train":
        args.criterion = criterions[args.criterion]
        
        args.trainer, trainer_parser = trainers[args.trainer]
        args.trainer_args, rest = trainer_parser(rest)
        args.trainer_args = vars(args.trainer_args)

        args.model, model_parser = models[args.model]
        args.model_args, rest = model_parser(rest)
        args.model_args = vars(args.model_args)

        args.optimizer, optimizer_parser = optimizers[args.optimizer]
        args.optimizer_args, rest = optimizer_parser(rest)
        args.optimizer_args = vars(args.optimizer_args)

        args.scheduler, scheduler_parser = schedulers[args.scheduler]
        args.scheduler_args, rest = scheduler_parser(rest)
        args.scheduler_args = vars(args.scheduler_args)

        dss = []
        ncls = []
        means = []
        stds = []
        for dataset in args.dataset:
            ds, nc, ch, mn, sd  = datasets[dataset]
            dss.append(ds)
            ncls.append(nc)
            means.append(mn)
            stds.append(sd)
        args.dataset = [dataset for dataset in dss]
        args.num_classes = [num_classes for num_classes in ncls]
        args.means = [mean for mean in means]
        args.stds = [std for std in stds]
        tf = []
        for transform in args.transforms:
            transform, transform_parser = transforms[transform]
            transform_args, rest = transform_parser(rest)
            tf.append(transform(**vars(transform_args)))
        args.transforms = torchvision.transforms.Compose(tf)

        if len(rest) != 0:
            raise ValueError(f"Unrecognized arguments: {rest}")
        return vars(args)
    elif args.mode == "load":
        args = torch.load(os.path.joint(args.path, "config.pt"))
        return vars(args)
    elif args.mode == "evaluation":
        args = torch.load(os.path.joint(args.path, "config.pt"))
        return vars(args)