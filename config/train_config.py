import torch
import argparse
from trainers import *

# This file contains the configuration for the models
# example:
# def model(args):
#     model = argparse.ArgumentParser()
#     model.add_argument("--pretrained"       , type=bool, default=True)
#     model.add_argument("--num_classes"      , type=int, default=10)
#     return model.parse_args(args)

def joint(args):
    joint = argparse.ArgumentParser()
    return joint.parse_known_args(args)

def cil(args):
    cil = argparse.ArgumentParser()
    cil.add_argument("--num_tasks"       , type=int, default=10)
    cil.add_argument("--reset-optimizer" , action="store_true")
    return cil.parse_known_args(args)

models = {
    "joint": (Joint, joint),
    "CIL" : (CIL, cil),
    # "Online" : (Online, online),
}