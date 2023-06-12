# Standalone script for maximizing the brightnesses of TIFF images so that max pixel value is 1

import numpy as np
from tifffile import imread, imwrite

def maximize_pure_colours(image, substituted_value = 1.0):
    # Find the maximum pixel value for each color channel
    max_values = np.max(image, axis=(0, 1))

    # Create masks for each color channel
    red_mask = (image[:, :, 0] != 0.0) & (image[:, :, 1] == 0.0) & (image[:, :, 2] == 0.0)
    green_mask = (image[:, :, 0] == 0.0) & (image[:, :, 1] != 0.0) & (image[:, :, 2] == 0.0)
    blue_mask = (image[:, :, 0] == 0.0) & (image[:, :, 1] == 0.0) & (image[:, :, 2] != 0.0)

    # Create a copy of the image to avoid modifying the original
    result = np.copy(image)

    # Set matching pixels to 1.0 for the corresponding color channel
    result[red_mask] = [substituted_value, 0.0, 0.0]
    result[green_mask] = [0.0, substituted_value, 0.0]
    result[blue_mask] = [0.0, 0.0, substituted_value]

    return result

def process_images(directories_list, substituted_value = 1.0):
    for directory in directories_list:
        image = imread(directory)
        image_brightened = maximize_pure_colours(image = image, substituted_value = substituted_value)
        new_directory = directory.rsplit(".", 1)[0] + "_brightened.tif"
        imwrite(new_directory, image_brightened)
