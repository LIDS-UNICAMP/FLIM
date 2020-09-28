import argparse

import napari

import numpy as np

from PIL import Image

from skimage import io
from skimage import util

from os import path

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('image',
                        help="Image to segment.")

    parser.add_argument('-m',
                        '--markers',
                        help="Image markers for the the segmentation.",
                        default=None)

    args = parser.parse_args()

    return args


def load_label_image(label_path):
    if label_path.endswith('.txt'):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        label_infos = [int(info) for info in lines[0].split(" ")]

        #images dimensions are flipped
        image_shape = (label_infos[2], label_infos[1])
        label_image = np.zeros(image_shape, dtype=np.int)

        for line in lines[1:]:
            split_line = line.split(" ")
            y, x, label = int(split_line[0]), int(split_line[1]), int(split_line[3])
            label_image[x][y] = label
        
        assert (label_image != 0).sum() == label_infos[0], "There are zero markers. Be careful!!"
    else:
        label_image = np.array(Image.open(label_path))

    return label_image

def load_image(image_dir):
    image = np.array(Image.open(image_dir))

    return image

def save_markers(markers, markers_dir):
    markers = markers.astype(np.int)
    mask = markers != 0
    
    number_of_markers = mask.sum()
    markers_shape = markers.shape
    
    x_coords, y_coords = np.where(mask)
    
    with open(markers_dir, 'w') as f:
        f.write(f"{number_of_markers} {markers_shape[1]} {markers_shape[0]}\n")
        for x, y in zip(x_coords, y_coords):
            f.write(f"{y} {x} {-1} {markers[x][y]}\n")

def create_viewer(image_dir,
                  markers_dir=None,
                  ):

    image = load_image(image_dir)
    initial = np.zeros(image.shape[:2], dtype=np.int)

    if markers_dir is not None:
        markers = load_label_image(markers_dir)
    else:
        markers = initial

    with napari.gui_qt():

        viewer = napari.Viewer(title="Interative tool. I hate my life.")
        viewer.add_image(image, name="image")
        viewer.add_labels(markers, name='markers', opacity=1)

        @viewer.bind_key('r')
        def refresh(viewer):
            initial = np.zeros(image.shape[:2], dtype=np.int)
            viewer.layers['markers'].data = initial
            # viewer.layers['instability map'].data = initial

        
        @viewer.bind_key('s')
        def write_segmentation(viewer):
            print("Saving markers...")
            nonlocal markers_dir
            markers = viewer.layers['markers'].data

            if markers_dir is not None:
                markers_dir = image_dir.split('.')[0] + ".txt"
            save_markers(markers, markers_dir)

            



def main():
    args  = get_arguments()
    create_viewer(
        args.image,
        args.markers)

if __name__ == "__main__":
    main()