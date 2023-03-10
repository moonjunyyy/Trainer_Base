import torch
import argparse
from models import *

# This file contains the configuration for the models
# example:
# def model(args):
#     model = argparse.ArgumentParser()
#     model.add_argument("--pretrained"       , type=bool, default=True)
#     model.add_argument("--num_classes"      , type=int, default=10)
#     return model.parse_args(args)

def resnet18(args):
    model = argparse.ArgumentParser()
    model.add_argument("--num_layer"  , type=int, default=18)
    model.add_argument("--num_classes" , type=int, default=10)
    model.add_argument("--pretrained" , type=bool, default=True)
    return model.parse_known_args(args)

models = {
    "resnet18": (ResNet, resnet18),
}