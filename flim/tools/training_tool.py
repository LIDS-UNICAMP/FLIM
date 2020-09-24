import argparse

import os

import torch
from torch.nn.parallel.data_parallel import data_parallel
import torchvision.transforms as transforms

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

    parser.add_argument('-bs', '--batch-size',
                        help="Batch size used for training",
                        type=int,
                        default=4)

    parser.add_argument('-lr', '--learning-rate',
                        help="Learning rate for optimizer",
                        type=float,
                        default=10e-3)

    parser.add_argument('-wd', '--weight-decay',
                        help="Weight decay",
                        default=10e-3,
                        type=float)
    
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

    device = get_device(args.gpus)

    model = utils.build_model(architecture, images, markers, device=device)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = utils.configure_dataset(args.dataset_dir, args.train_split, transform)

    utils.train_mlp(model,
                    dataset,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.weight_decay,
                    device=device)

    utils.save_model(model, args.outputs_dir, args.model_filename)
    

if __name__ == "__main__":
    main()