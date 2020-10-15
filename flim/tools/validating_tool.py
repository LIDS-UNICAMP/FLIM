import argparse

import os

import torch
from torch.nn.parallel.data_parallel import data_parallel
import torchvision.transforms as transforms

from ..experiments import utils

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset-dir',
                        help="Dataset to train the mlp.",
                        required=True)

    parser.add_argument('-vs', '--val-split',
                        help="Split to train the mlp.",
                        required=True)

    parser.add_argument("-ad", "--architecture-dir",
                        help="Architecture dir", 
                        required=True)
    
    parser.add_argument("-mp", "--model-path",
                        help="Saved model path.",
                        required=True)
    
    parser.add_argument('-g', '--gpus',
                        help='gpus to use',
                        nargs='*',
                        type=int)

    parser.add_argument('-s', '--svm',
                        help='Use SVM as classifier',
                        action="store_true")

    parser.add_argument("-smp", "--svm-path",
                        help="Saved SVM path. Ex. svm.joblib.",
                        required=False)
    
    parser.add_argument("-rb",
                        "--remove-border",
                        help="Remove border of size before classifier.",
                        type=int,
                        default=0)
    
    parser.add_argument("-tm",
                        "--torchvision-model",
                        help="Torchvision model",
                        choices=["vgg16_bn"],
                        default=None)
    
    parser.add_argument("-nc",
                        "--number-classes",
                        help="Number of classes",
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


    device = get_device(args.gpus)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = utils.configure_dataset(args.dataset_dir, args.val_split, transform)
    input_shape = list(dataset[0][0].permute(1, 2, 0).size())
    
    architecture = None
    
    if args.torchvision_model is None:
        architecture = utils.load_architecture(args.architecture_dir)
        model = utils.load_model(args.model_path, architecture, input_shape, remove_border=args.remove_border)
    else:
        model = utils.get_torchvision_model(args.torchvision_model,
                                            args.number_classes,
                                            pretrained=False,
                                            device=device)
        
        
        model = utils.load_torchvision_model_weights(model, args.model_path)
        model.to(device)

    if args.svm:
        clf = utils.load_svm(args.svm_path)
        utils.validate_svm(model, clf, dataset, device=device)
    elif args.torchvision_model is not None or "classifier" in architecture:
        utils.validate_model(model, dataset)
    else:
        print("No classifier to evaluate...")

if __name__ == "__main__":
    main()