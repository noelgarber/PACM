import numpy as np
import os
from tifffile import imread, imwrite

parent_directory = input("Please enter the parent directory where TIFF files are stored:  ")
filenames = os.listdir(parent_directory)
paths = [os.path.join(parent_directory, filename) for filename in filenames]

output_directory = input("Enter the directory to save frequency domain representations:  ")

shifted_frequency_domains = []

for filename, path in zip(filenames, paths):
    image = imread(path)
    if image.ndim > 2:
        image = image[:,:,0]
        print(f"The image at {path} has multiple channels, so the first was used.")

    frequency_domain = np.fft.fft2(image)
    frequency_domain_shifted = np.fft.fftshift(frequency_domain)
    shifted_frequency_domains.append(frequency_domain_shifted)

    magnitude_spectrum = np.abs(frequency_domain_shifted)
    output_representation = np.log(1+magnitude_spectrum)

    output_path = os.path.join(output_directory, filename.rsplit(".",1)[0] + "_frequency_domain.TIF")
    imwrite(output_path, output_representation)

mask_directory = input("Enter the directory of the mask to use for suppressing undesired frequencies:  ")
mask = imread(mask_directory)
if mask.ndim > 2:
    mask = mask[:,:,0]
bool_mask = mask.astype(bool)

filtered_output_directory = input("Enter the directory to save filtered images:  ")

for i, frequency_domain_shifted in enumerate(shifted_frequency_domains):
    filtered_frequencies_shifted = frequency_domain_shifted.copy()
    filtered_frequencies_shifted[bool_mask] = 0

    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_frequencies_shifted)).real
    output_path = os.path.join(filtered_output_directory, filenames[i].rsplit(".",1)[0] + "_filtered.TIF")
    imwrite(output_path, filtered_image)

print("Done!")