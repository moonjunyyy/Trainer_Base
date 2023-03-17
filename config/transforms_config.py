import torchvision.transforms as transforms
import argparse

def AutoAugment(args):
    autoaugment = argparse.ArgumentParser()
    return autoaugment.parse_known_args(args)

def ColorJitter(args):
    colorjitter = argparse.ArgumentParser()
    colorjitter.add_argument("--brightness" , type=float, default=0)
    colorjitter.add_argument("--contrast" , type=float, default=0)
    colorjitter.add_argument("--saturation" , type=float, default=0)
    colorjitter.add_argument("--hue" , type=float, default=0)
    return colorjitter.parse_known_args(args)

def GaussianBlur(args):
    gaussianblur = argparse.ArgumentParser()
    gaussianblur.add_argument("--kernel_size" , type=int, default=3)
    gaussianblur.add_argument("--sigma" , type=float, default=1.0)
    return gaussianblur.parse_known_args(args)

def RandomAffine(args):
    randomaffine = argparse.ArgumentParser()
    randomaffine.add_argument("--degrees" , type=float, default=0)
    randomaffine.add_argument("--translate" , type=float, default=0)
    randomaffine.add_argument("--scale" , type=float, default=0)
    randomaffine.add_argument("--shear" , type=float, default=0)
    randomaffine.add_argument("--fillcolor" , type=int, default=0)
    return randomaffine.parse_known_args(args)

def RandomCrop(args):
    randomcrop = argparse.ArgumentParser()
    randomcrop.add_argument("--size" , type=int, default=0)
    randomcrop.add_argument("--padding" , type=int, default=0)
    randomcrop.add_argument("--pad_if_needed" , action="store_true")
    randomcrop.add_argument("--fill" , type=int, default=0)
    randomcrop.add_argument("--padding_mode" , type=str, default="constant")
    return randomcrop.parse_known_args(args)

def RandomErasing(args):
    randomerasing = argparse.ArgumentParser()
    randomerasing.add_argument("--p" , type=float, default=0.5)
    randomerasing.add_argument("--scale" , type=float, default=(0.02, 0.33))
    randomerasing.add_argument("--ratio" , type=float, default=(0.3, 3.3))
    randomerasing.add_argument("--value" , type=int, default=0)
    randomerasing.add_argument("--inplace" , action="store_true")
    return randomerasing.parse_known_args(args)

def RandomGrayscale(args):
    randomgrayscale = argparse.ArgumentParser()
    randomgrayscale.add_argument("--p" , type=float, default=0.1)
    return randomgrayscale.parse_known_args(args)

def RandomHorizontalFlip(args):
    randomhorizontalflip = argparse.ArgumentParser()
    randomhorizontalflip.add_argument("--p" , type=float, default=0.5)
    return randomhorizontalflip.parse_known_args(args)

def RandomPerspective(args):
    randomperspective = argparse.ArgumentParser()
    randomperspective.add_argument("--distortion_scale" , type=float, default=0.5)
    randomperspective.add_argument("--p" , type=float, default=0.5)
    return randomperspective.parse_known_args(args)

def RandomResizedCrop(args):
    randomresizedcrop = argparse.ArgumentParser()
    randomresizedcrop.add_argument("--size" , type=int, default=(32,32))
    randomresizedcrop.add_argument("--scale" , type=float, default=(0.08, 1.0))
    randomresizedcrop.add_argument("--ratio" , type=float, default=(0.75, 1.3333333333333333))
    return randomresizedcrop.parse_known_args(args)

def RandomRotation(args):
    randomrotation = argparse.ArgumentParser()
    randomrotation.add_argument("--degrees" , type=float, default=0)
    randomrotation.add_argument("--expand" , action="store_true")
    randomrotation.add_argument("--center" , type=int, default=None)
    return randomrotation.parse_known_args(args)

def RandomSizedCrop(args):
    randomsizedcrop = argparse.ArgumentParser()
    randomsizedcrop.add_argument("--size" , type=int, default=0)
    randomsizedcrop.add_argument("--scale" , type=float, default=(0.08, 1.0))
    randomsizedcrop.add_argument("--ratio" , type=float, default=(0.75, 1.3333333333333333))
    return randomsizedcrop.parse_known_args(args)

def RandomVerticalFlip(args):
    randomverticalflip = argparse.ArgumentParser()
    randomverticalflip.add_argument("--p" , type=float, default=0.5)
    return randomverticalflip.parse_known_args(args)

def Resize(args):
    resize = argparse.ArgumentParser()
    resize.add_argument("--size" , type=int, default=0)
    return resize.parse_known_args(args)

transforms = {
    'AutoAugment': (transforms.AutoAugment, AutoAugment),
    'ColorJitter': (transforms.ColorJitter, ColorJitter),
    'GaussianBlur': (transforms.GaussianBlur, GaussianBlur),
    'RandomAffine': (transforms.RandomAffine, RandomAffine),
    'RandomCrop': (transforms.RandomCrop, RandomCrop),
    'RandomErasing': (transforms.RandomErasing, RandomErasing),
    'RandomGrayscale': (transforms.RandomGrayscale, RandomGrayscale),
    'RandomHorizontalFlip': (transforms.RandomHorizontalFlip, RandomHorizontalFlip),
    'RandomPerspective': (transforms.RandomPerspective, RandomPerspective),
    'RandomResizedCrop': (transforms.RandomResizedCrop, RandomResizedCrop),
    'RandomRotation': (transforms.RandomRotation, RandomRotation),
    'RandomVerticalFlip': (transforms.RandomVerticalFlip, RandomVerticalFlip),
    'Resize': (transforms.Resize, Resize),
}