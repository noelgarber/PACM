import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imshow

def detect_circle(img, dilate_to_edge = True, verbose = False):
    print(f"Input image type: {type(img)}")

    # Ensure that only the 1st layer is used
    if img.ndim == 3:
        print(f"detect_circle() warning: input image has {img.shape[2]} layers/channels, but only grayscale is supported; the 1st layer will be used.",
              "\n\tThis can be caused by the presence of an alpha channel, which will be disregarded.") if verbose else None
        img = img[:,:,0]
    elif img.ndim != 2:
        raise Exception(f"detect_circle() input image array has {img.ndim} dimensions; expected 2 or 3.")

    # Convert image to 8-bit integer
    img_8bit = img * 255
    img_8bit = img_8bit.astype(np.uint8)
    img_contrasted = img_8bit - img_8bit.min() # ensures that the lowest pixel value = 0
    img_contrasted = (img_contrasted/img_contrasted.max()) * 255
    img_contrasted = img_contrasted.astype(np.uint8)

    # Detect circles using Hough transform
    max_radius = int(np.sqrt((img_8bit.shape[0]-1)**2 + (img_8bit.shape[1]-1)**2))

    circles = cv2.HoughCircles(img_contrasted, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=10, param2=5, minRadius=0, maxRadius=max_radius)
    if circles is None:
        print("detect_circle() warning: no circles were found!")
        return None, None, None

    # Draw circle outline in red
    circles = np.uint16(np.around(circles))
    center = (circles[0][0][0], circles[0][0][1])
    if dilate_to_edge:
        img_height, img_width = np.shape(img)[:2]
        radius = min(center[0], center[1], img_width - center[0], img_height - center[1])
    else:
        radius = circles[0][0][2]
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    cv2.circle(img_color, center, radius, (0, 0, 255), 1)

    # Compute sums of pixels inside and outside circle
    mask = np.zeros(img_8bit.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    inside_sum = np.sum(img[mask == 255])
    outside_sum = np.sum(img[mask == 0])

    # Compute the "ellipsoid index" (degree to which signal is constrained to be within the circle/ellipsoid)
    inside_count = len(img[mask == 255])
    outside_count = len(img[mask == 0])
    mean_inside = inside_sum / inside_count
    mean_outside = outside_sum / outside_count
    ellipsoid_index = mean_inside / mean_outside

    return img_color, inside_sum, outside_sum, ellipsoid_index

if __name__ == "__main__":
    print("This script describes a function for finding spots in image snippets. It should only be run alone for debugging.")
    path = input("Enter the path to the grayscale image: ")
    image = imread(path).astype(np.float32)
    outlined_image, sum_inside, sum_outside, ellipsoid_index = detect_circle(image, dilate_to_edge = True, verbose = True)
    if outlined_image is not None:
        print(f"Sum inside: {sum_inside}")
        print(f"Sum outside: {sum_outside}")
        print(f"Ellipsoid index: {ellipsoid_index}")
        imshow(outlined_image)
        plt.show()