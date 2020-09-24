import argparse

import torch
from torch._C import device

from ..experiments import utils

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-md', '--markers-dir',
                        help="Directory to images and markers."
                        +"Markers and images must have the same name with diffetent extension. "
                        +"Markers must have txt extensions, "
                        +"and images might have any image extension or npy extension.", 
                        required=True)

    parser.add_argument('-d', '--dataset-dir',
                        help="Dataset to train the mlp.",
                        required=True)

    parser.add_argument('-ts', '--train-split',
                        help="Split to train the mlp.",
                        required=True)

    parser.add_argument("-ad", "--architecture-dir",
                        help="Architecture dir", 
                        required=True)
    
    parser.add_argument("-od", '--outputs-dir',
                        help="Where to save outputs produced during traning such as ift datasets.",
                        required=True)
    
    parser.add_argument("-mn", "--model-filename",
                        help="Model filename",
                        required=True)
    
    parser.add_argument('-e', '--epochs',
                        help="number of epochs for training",
                        type=int,
                        default=15)
    
    parser.add_argument('-g', '--gpus',
                        help='gpus to use',
                        nargs='*',
                        type=int)
    
    args = parser.parse_args()
    
    return args

def get_device(gpus):
    gpu = torch.cuda.is_available()

    if(gpus is None or not gpu):
        device = torch.device('cpu')
    else:
        device = torch.device(gpus[0])

    return device

def main():
    args = get_arguments()

    architecture = utils.load_architecture(args.architecture_dir)
    images, markers = utils.load_images_and_markers(args.markers_dir)

    print(images.shape)

    device = get_device(args.gpus)

    model = utils.build_model(architecture, images, markers, device=device)

if __name__ == "__main__":
    main()