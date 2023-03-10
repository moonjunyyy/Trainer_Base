import torch
import argparse
from schedulers.warmup_cosing_anneling import WarmupCosineAnnealingLR 

def const(args):
    const = argparse.ArgumentParser()
    const.add_argument("--factor"       , type=float, default=1)
    const.add_argument("--total-iters"  , type=int, default=100)
    const.add_argument("--last-epoch"   , type=int, default=-1)
    return const.parse_known_args(args)

def warmup_cosine(args):
    warmup_cosine.add_argument("--T-max"        , type=int,   default=15)
    warmup_cosine.add_argument("--eta-min"      , type=float, default=1e-6)
    warmup_cosine.add_argument("--warmup-epochs" , type=int,   default=10)
    warmup_cosine.add_argument("--last-epoch"  , type=int,   default=-1)
    return warmup_cosine.parse_known_args(args)

def step(args):
    step = argparse.ArgumentParser()
    step.add_argument("--step_size"         , type=int, default=30)
    step.add_argument("--gamma"             , type=float, default=0.1)
    step.add_argument("--last-epoch"     , type=int, default=0)
    return step.parse_known_args(args)

def exponential(args):
    exponential = argparse.ArgumentParser()
    exponential.add_argument("--gamma"         , type=float, default=0.99)
    exponential.add_argument("--last-epoch"    , type=int, default=-1)
    return exponential.parse_known_args(args)

def cosine(args):
    cosine = argparse.ArgumentParser()
    cosine.add_argument("--T-max"            , type=int, default=10)
    cosine.add_argument("--eta-min"          , type=float, default=0)
    cosine.add_argument("--last-epoch"       , type=int, default=-1)
    return cosine.parse_known_args(args)

schedulers = {
    "step"          : (torch.optim.lr_scheduler.StepLR, step),
    "exponential"   : (torch.optim.lr_scheduler.ExponentialLR, exponential),
    "cosine"        : (torch.optim.lr_scheduler.CosineAnnealingLR, cosine),
    "const"         : (torch.optim.lr_scheduler.ConstantLR, const),
    "warmup_cosine" : (WarmupCosineAnnealingLR, warmup_cosine),
}
