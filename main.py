import os
import torch
from pprint import pprint
# from config import parser
from config.config import config

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING']='1'

def main():
    args = config()
    pprint(args)
    # print(args)

    if args["mode"] == "train":
        trainer = args["trainer"](**args)
        trainer.load_checkpoint(None)
        trainer.run()

    elif args["mode"] == "load":
        trainer = args["trainer"](**args)
        trainer.load_checkpoint(chpt)
        trainer.run()

    elif args["mode"] == "evaluation":
        trainer = args["trainer"](**args)
        chpt = torch.load(os.path.join(args["path"], "best.pt"))
        trainer.load_checkpoint(chpt)
        trainer.eval()
    return

if __name__ == '__main__':
    main()