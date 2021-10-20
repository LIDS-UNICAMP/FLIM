import argparse
from ast import Str

import os
from os import access, makedirs

import shutil

import torch

import numpy as np

import torchvision.transforms as transforms

import matplotlib as mpl

from skimage import io

from ..experiments import utils
from ..experiments._dataset import ToTensor


def get_device(gpus):
    gpu = torch.cuda.is_available()

    if gpus is None or not gpu:
        device = torch.device("cpu")
    else:
        device = torch.device(gpus[0])

    return device


def select_images_to_put_markers(
    dataset_dir, split_path, markers_dir, class_proportion=0.05
):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = utils.configure_dataset(dataset_dir, split_path, transform)

    _, images_names = utils.select_images_to_put_markers(dataset, class_proportion)

    if not os.path.exists(markers_dir):
        os.makedirs(markers_dir)

    for image_name in images_names:
        src = os.path.join(dataset_dir, image_name)
        dst = os.path.join(markers_dir, image_name)

        shutil.copy(src=src, dst=dst)


def _handle_train(args):
    device = get_device(args.gpus)

    torch.manual_seed(42)
    np.random.seed(42)
    if device != "cpu":
        torch.backends.cudnn.deterministic = True

    architecture = None

    if args.torchvision_model is None:
        architecture = utils.load_architecture(args.architecture_dir)

    if not args.load_lids_model and not args.torchvision_model and args.markers_dir:
        images, markers = utils.load_images_and_markers(args.markers_dir)
    else:
        images, markers = None, None

    if args.dataset_dir is not None:
        transform = transforms.Compose([ToTensor()])
        dataset = utils.configure_dataset(args.dataset_dir, args.train_split, transform)

        _first_image = dataset[0][0]

        if _first_image.ndim > 1:
            input_shape = _first_image.permute(1, 2, 0).shape
        else:
            input_shape = _first_image.shape
    else:
        dataset = None
        input_shape = None

    # print(input_shape)

    if architecture is not None and not args.load_lids_model:
        model = utils.build_model(
            architecture,
            images,
            markers,
            input_shape=input_shape,
            train_set=dataset,
            remove_border=args.remove_border,
            relabel_markers=True,
            batch_size=args.batch_size,
            default_std=args.default_std,
            device=device,
        )
    elif architecture is not None:
        model = utils.build_model(
            architecture,
            images,
            markers,
            input_shape=input_shape,
            remove_border=args.remove_border,
            relabel_markers=False,
            batch_size=args.batch_size,
            default_std=args.default_std,
            device=device,
        )

    else:
        model = utils.get_torchvision_model(
            args.torchvision_model,
            args.number_classes,
            pretrained=args.pretrained,
            device=device,
        )
    if args.load_lids_model:
        model = utils.load_lids_model(model, args.lids_model_dir, args.train_split)

    if args.backpropagation:
        utils.train_model(
            model,
            dataset,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            step=args.step,
            device=device,
            only_classifier=args.only_classifier,
        )
    if args.svm:
        svm = utils.train_svm(
            model, dataset, args.batch_size, args.svm_max_iter, device
        )

        utils.save_svm(svm, args.outputs_dir, args.svm_filename)

    utils.save_model(model, args.outputs_dir, args.model_filename)
    utils.save_lids_model(
        model, architecture, args.train_split, args.outputs_dir, args.model_filename
    )

    if args.intermediate:
        utils.save_intermediate_outputs(
            model,
            dataset,
            args.outputs_dir,
            batch_size=args.batch_size,
            format=args.format,
            layers=args.layers,
            remove_border=args.remove_border,
            device=device,
        )


def _handle_select(args):
    select_images_to_put_markers(
        args.dataset_dir, args.split, args.markers_dir, args.class_proportion
    )


def _handle_split(args):
    print("Splitting dataset...")
    train_split, val_split, test_split = utils.split_dataset(
        args.dataset_dir, args.train_size, args.val_size, args.test_size
    )
    with open(f"{args.split_name}-train.txt", "w") as f:
        for filename in train_split:
            f.write(filename)
            f.write("\n")

    with open(f"{args.split_name}-val.txt", "w") as f:
        for filename in val_split:
            f.write(filename)
            f.write("\n")

    with open(f"{args.split_name}-test.txt", "w") as f:
        for filename in test_split:
            f.write(filename)
            f.write("\n")


def handle_explain(args):
    device = get_device(args.gpus)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = utils.configure_dataset(args.dataset_dir, args.train_split, transform)
    input_shape = list(dataset[0][0].permute(1, 2, 0).size())

    architecture = None

    if args.torchvision_model is None:
        architecture = utils.load_architecture(args.architecture_dir)
        model = utils.load_model(
            args.model_path, architecture, input_shape, remove_border=args.remove_border
        )
    else:
        model = utils.get_torchvision_model(
            args.torchvision_model, args.number_classes, pretrained=False, device=device
        )

        model = utils.load_torchvision_model_weights(model, args.model_path)
        model.to(device)
    image_indices = args.image_indices

    if args.intermediate:
        utils.get_intermediate_outputs(
            model,
            dataset,
            args.outputs_dir,
            batch_size=args.batch_size,
            format=args.format,
            device=device,
        )

    outputs_dir = os.path.join(args.outputs_dir, "CAMs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    if image_indices is None:
        image_indices = list(range(len(dataset)))

    for image_index in image_indices:
        image, label = dataset[image_index]
        cam = utils.compute_grad_cam(model, image, args.target_layers, label, device)
        color_map = mpl.cm.get_cmap("jet")
        c_cam = color_map(cam)
        image_name = dataset.images_names[image_index]
        image = utils.image_to_rgb(image.permute(1, 2, 0))
        io.imsave(os.path.join(outputs_dir, image_name), c_cam[0][:, :, :3] + image)


def handle_arch(args):
    layers_spec_dir = args.layers_spec_dir

    arch = utils.create_arch(layers_spec_dir)

    utils.save_arch(arch, args.architecture_path)


def get_arguments():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train", help="Train model.")

    parser_train.add_argument(
        "-md",
        "--markers-dir",
        help="Directory to images and markers."
        + "Markers and images must have the same name with diffetent extension. "
        + "Markers must have txt extensions, "
        + "and images might have any image extension or npy extension.",
    )

    parser_train.add_argument(
        "-d", "--dataset-dir", help="Dataset to train the mlp.", required=False
    )

    parser_train.add_argument("-ts", "--train-split", help="Split to train the mlp.")

    parser_train.add_argument("-ad", "--architecture-dir", help="Architecture dir")

    parser_train.add_argument(
        "-od",
        "--outputs-dir",
        help="Where to save outputs produced during traning such as ift datasets.",
        required=True,
    )

    parser_train.add_argument(
        "-mn", "--model-filename", help="Model filename", required=True
    )

    parser_train.add_argument(
        "-e", "--epochs", help="number of epochs for training", type=int, default=15
    )

    parser_train.add_argument(
        "-bs", "--batch-size", help="Batch size used for training", type=int, default=4
    )

    parser_train.add_argument(
        "-lr",
        "--learning-rate",
        help="Learning rate for optimizer",
        type=float,
        default=10e-3,
    )

    parser_train.add_argument(
        "-wd", "--weight-decay", help="Weight decay", default=1e-3, type=float
    )

    parser_train.add_argument("-g", "--gpus", help="gpus to use", nargs="*", type=int)

    parser_train.add_argument(
        "-dstd", "--default-std", help="Default std", default=1e-6, type=float
    )

    parser_train.add_argument(
        "-s", "--svm", help="Use SVM as classifier", action="store_true"
    )

    parser_train.add_argument(
        "-smi",
        "--svm-max-iter",
        help="Maximum of iterations to train SVM",
        type=int,
        default=1000,
        required=False,
    )

    parser_train.add_argument(
        "-smn", "--svm-filename", help="SVM filename. Ex. svm.joblib.", required=False
    )

    parser_train.add_argument(
        "-b",
        "--backpropagation",
        help="Use backpropagation to trian layers.",
        action="store_true",
    )

    parser_train.add_argument(
        "-c",
        "--only-classifier",
        help="Train with only the classifier with backpropagation.",
        action="store_true",
    )

    parser_train.add_argument(
        "-ld",
        "--load-lids-model",
        help="Load model saved in LIDS format.",
        action="store_true",
    )

    parser_train.add_argument("-ldd", "--lids-model-dir", help="LIDS model dir.")

    parser_train.add_argument(
        "-rb",
        "--remove-border",
        help="Remove border of size before classifier.",
        type=int,
        default=0,
    )

    parser_train.add_argument(
        "-st", "--step", help="Step for leraning rate scheduler.", type=int, default=15
    )

    parser_train.add_argument(
        "-tm",
        "--torchvision-model",
        help="Torchvision model",
        choices=["vgg16_bn"],
        default=None,
    )

    parser_train.add_argument(
        "-nc", "--number-classes", help="Number of classes", type=int
    )

    parser_train.add_argument(
        "-pt", "--pretrained", help="Use pretrained weigths", action="store_true"
    )

    parser_train.add_argument(
        "-i",
        "--intermediate",
        help="Save the output of intermediate layers",
        action="store_true",
    )

    parser_train.add_argument(
        "-f",
        "--format",
        help="Format to save intermediate layers output",
        choices=["zip", "mimg", "npy"],
    )

    parser_train.add_argument(
        "-l",
        "--layers",
        help="Save the output of these layers.",
        nargs="*",
    )

    parser_train.set_defaults(func=_handle_train)

    parser_select = subparsers.add_parser("select", help="Select images to put markers")

    parser_select.add_argument(
        "-md",
        "--markers-dir",
        help="Directory to save images to put markers",
        required=True,
    )

    parser_select.add_argument(
        "-d", "--dataset-dir", help="Dataset directory.", required=True
    )

    parser_select.add_argument("-s", "--split", help="Data set split.", required=True)

    parser_select.add_argument(
        "-p",
        "--class-proportion",
        help="How many images of each class to select. Must be in range (0, 1].",
        type=float,
        default=0.5,
    )

    parser_select.set_defaults(func=_handle_select)

    parser_split = subparsers.add_parser(
        "split", help="Split dataset in train, val and test."
    )

    parser_split.add_argument("-d", "--dataset-dir", help="Dataset dir", required=True)

    parser_split.add_argument("-n", "--split-name", help="Split name.", required=True)

    parser_split.add_argument(
        "-ts", "--train-size", help="Train size", required=True, type=float
    )

    parser_split.add_argument(
        "-vs", "--val-size", help="Val size", default=None, type=float
    )

    parser_split.add_argument(
        "-tts", "--test-size", help="Test size", default=None, type=float
    )

    parser_split.set_defaults(func=_handle_split)

    parser_explain = subparsers.add_parser("explain", help="Analyze model outputs.")

    parser_explain.add_argument(
        "-d", "--dataset-dir", help="Dataset to train the mlp.", required=True
    )

    parser_explain.add_argument(
        "-ts", "--train-split", help="Split to train the mlp.", required=True
    )

    parser_explain.add_argument("-ad", "--architecture-dir", help="Architecture dir")

    parser_explain.add_argument(
        "-rb",
        "--remove-border",
        help="Remove border of size before classifier.",
        type=int,
        default=0,
    )

    parser_explain.add_argument(
        "-od",
        "--outputs-dir",
        help="Where to save outputs produced during traning such as ift datasets.",
        required=True,
    )

    parser_explain.add_argument(
        "-mp", "--model-path", help="Saved model path.", required=True
    )

    parser_explain.add_argument("-g", "--gpus", help="gpus to use", nargs="*", type=int)

    parser_explain.add_argument(
        "-tm",
        "--torchvision-model",
        help="Torchvision model",
        choices=["vgg16_bn"],
        default=None,
    )

    parser_explain.add_argument(
        "-i",
        "--image-indices",
        help="Images to compute Grad-CAM. If none is passed, it will use all images.",
        nargs="*",
        type=int,
    )

    parser_explain.add_argument(
        "-t", "--target-layers", help="Target layers", nargs="*"
    )

    parser_explain.add_argument(
        "-nc", "--number-classes", help="Number of classes", type=int
    )

    parser_explain.add_argument(
        "--intermediate", help="Save intermediate outputs", action="store_true"
    )

    parser_explain.set_defaults(func=handle_explain)

    parser_arch = subparsers.add_parser("arch", help="Architecture model.")

    parser_arch.add_argument("-l", "--layers-spec-dir", help="Layers directory.")

    parser_arch.add_argument("-a", "--architecture-path", help="Architecture path.")

    parser_arch.set_defaults(func=handle_arch)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        print("Run train -h for help.")

    return args


def main():
    args = get_arguments()


if __name__ == "__main__":
    main()
