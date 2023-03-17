import torch
import argparse

# This file contains the configuration for the optimizers

def adam(args):
    adam = argparse.ArgumentParser()
    adam.add_argument("--lr"                , type=float, default=0.001)
    adam.add_argument("--betas"             , type=tuple, default=(0.9, 0.999))
    adam.add_argument("--eps"               , type=float, default=1e-08)
    adam.add_argument("--weight_decay"      , type=float, default=0)
    adam.add_argument("--amsgrad"           , type=bool, default=False)
    return adam.parse_known_args(args)

def adamw(args):
    adamw = argparse.ArgumentParser()
    adamw.add_argument("--lr"               , type=float, default=0.001)
    adamw.add_argument("--betas"            , type=tuple, default=(0.9, 0.999))
    adamw.add_argument("--eps"              , type=float, default=1e-08)
    adamw.add_argument("--weight_decay"     , type=float, default=0)
    adamw.add_argument("--amsgrad"          , type=bool, default=False)
    return adamw.parse_known_args(args)

def sgd(args):
    sgd = argparse.ArgumentParser()
    sgd.add_argument("--lr"                 , type=float, default=0.001)
    sgd.add_argument("--momentum"           , type=float, default=0.9)
    sgd.add_argument("--dampening"          , type=float, default=0)
    sgd.add_argument("--weight_decay"       , type=float, default=0)
    sgd.add_argument("--nesterov"           , type=bool, default=False)
    return sgd.parse_known_args(args)

def rmsprop(args):
    rmsprop = argparse.ArgumentParser()
    rmsprop.add_argument("--lr"             , type=float, default=0.001)
    rmsprop.add_argument("--alpha"          , type=float, default=0.99)
    rmsprop.add_argument("--eps"            , type=float, default=1e-08)
    rmsprop.add_argument("--weight_decay"   , type=float, default=0)
    rmsprop.add_argument("--momentum"       , type=float, default=0)
    rmsprop.add_argument("--centered"       , type=bool, default=False)
    return rmsprop.parse_known_args(args)

optimizers = {
    "sgd"       : (torch.optim.SGD, sgd),
    "adam"      : (torch.optim.Adam, adam),
    "adamw"     : (torch.optim.AdamW, adamw),
    "rmsprop"   : (torch.optim.RMSprop, rmsprop),
}