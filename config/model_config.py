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

def ouroboros(args):
    model = argparse.ArgumentParser()
    model.add_argument("--input_size" , type=int, default=224)
    model.add_argument("--hidden_size" , type=int, default=768)
    model.add_argument("--ffn_size" , type=int, default=3072)
    model.add_argument("--patch_size" , type=int, default=16)
    model.add_argument("--num_encoder_layer" , type=int, default=6)
    model.add_argument("--num_decoder_layer" , type=int, default=3)
    model.add_argument("--dropout" , type=float, default=0.1)
    model.add_argument("--num_classes" , type=int, default=100)
    return model.parse_known_args(args)

models = {
    "resnet18": (ResNet, resnet18),
    "ouroboros": (Ouroboros, ouroboros)
}