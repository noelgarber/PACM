import numpy as np

def draw_color_circle(image, midpoints, radius, circle_color = "green"):
    '''
    Simple function for drawing circle(s) of defined radius around a defined midpoint in a primary color.

    Args:
        image (np.ndarray): 2D (grayscale) or 3D (color) numpy array of floating points representing an image
        midpoints (list of tuples): one or more coordinates of the midpoint(s) of the circle(s), in (y,x) format
        radius (int): the radius of the circle; must not exceed the distance to the nearest border from the midpoint
        circle_color (str): the color of the circle to draw; default is green

    Returns:
        color_image (np.ndarray): 3D numpy array of floating points representing the image with the drawn circle
    '''

    # Convert 2D grayscale array to 3D color array
    if image.ndim == 2:
        color_image = np.stack([image]*3, axis=-1)
    elif image.ndim == 3:
        color_image = image
    else:
        raise Exception(f"draw_color_circle error: image has {image.ndim} dimensions, but 2 or 3 were expected.")

    # Define the color of the circle
    if circle_color in ["red", "Red", "r", "R"]:
        r, g, b = (image.max(), 0, 0)
    elif circle_color in ["green", "Green", "g", "G"]:
        r, g, b = (0, image.max(), 0)
    elif circle_color in ["blue", "Blue", "b", "B"]:
        r, g, b = (0, 0, image.max())
    else:
        raise Exception(f"draw_color_circle error: circle_color was given as {circle_color}, but must be one of [Red, Green, Blue, red, green, blue, R, G, B, r, g, b].")

    # Draw circle in green on the color image
    h, w = image.shape[:2]
    y_idxs, x_idxs = np.ogrid[:h, :w]
    for midpoint in midpoints:
        y, x = midpoint # assume numpy notation where (y,x) defines coordinates
        dists = np.sqrt((x_idxs - x)**2 + (y_idxs - y)**2)
        circle = (dists <= radius) & (dists >= radius-1)
        color_image[circle] = [r, g, b]

    return color_image

def concatenate_images(image_list):
    """
    Concatenates a list of small images into a larger image, using the top-left corner
    coordinates provided for each small image.

    Args:
        image_list (list): A list of tuples, where each tuple is (image, top_left_coords);
            image (np.ndarray): 3D numpy array representing a color image snippet in RGB format
            top_left_coords (tuple): top-left corner coordinates of the image snippet in a larger concatenated image

    Returns:
        concatenated_image (np.ndarray): 3D numpy array representing the concatenated image
    """
    # Determine the size of the final concatenated image
    max_x, max_y = 0, 0
    for _, (x, y) in image_list:
        max_x = max(max_x, x + _.shape[0])
        max_y = max(max_y, y + _.shape[1])

    # Create a blank image of the correct size
    channels = image_list[0][0].shape[2]
    concatenated_image = np.zeros((max_x, max_y, channels), dtype=image_list[0][0].dtype)

    # Paste each small image onto the concatenated image at the correct location
    for image, (x, y) in image_list:
        concatenated_image[x:x + image.shape[0], y:y + image.shape[1], :] = image

    return concatenated_image

def circle_stats(grayscale_image, center, radius, buffer_width = 0):
    '''
    Function to obtain pixel counts/sums inside and outside of a defined circle

    Args:
        grayscale_image (np.ndarray): image as 2D numpy array
        center (tuple): pair of (x,y) values representing the center point of the circle; x=horizontal and y=vertical
        radius (int): the radius of the circle to define
        buffer_width (int): the excluded buffer zone between pixels inside the circle and pixels outside the circle;
                            can be set to a positive value to remove effects of bleedover into the outer area during
                            background adjustment

    Returns:
        pixels_inside (np.int64): number of pixels inside the defined circle
        pixels_outside (np.int64): number of pixels outside the defined circle
        sum_inside (same dtype as input image, e.g. np.float16): sum of pixel values inside the circle
        sum_outside (same dtype as input image, e.g. np.float16): sum of pixel values outside the circle
    '''
    # Create a mesh grid of coordinates for the image
    x, y = np.meshgrid(np.arange(grayscale_image.shape[1]), np.arange(grayscale_image.shape[0]))

    # Calculate the distance between each pixel and the center
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a binary mask to identify the pixels inside the circle
    mask = distance <= radius
    dilated_mask = distance <= (radius + buffer_width)

    # Flatten the image and mask arrays
    flat_image = grayscale_image.flatten()
    flat_mask = mask.flatten()
    flat_dilated_mask = dilated_mask.flatten()

    # Count the number of pixels inside the circle and outside the circle
    pixels_inside = np.sum(flat_mask)
    pixels_outside = np.sum(~flat_dilated_mask)

    # Calculate the sum of pixel values inside the circle and outside the circle
    sum_inside = np.sum(flat_image[flat_mask])
    sum_outside = np.sum(flat_image[~flat_dilated_mask])

    # Calculate the ellipsoid index
    mean_intensity_inside = sum_inside / pixels_inside
    mean_intensity_outside = sum_outside / pixels_outside
    if mean_intensity_outside > 0:
        ellipsoid_index = mean_intensity_inside / mean_intensity_outside
    else:
        ellipsoid_index = "inf"

    # Find the background-adjusted sum of pixels inside the defined circle
    background_adjusted_inside_sum = sum_inside - (pixels_inside * mean_intensity_outside)

    return pixels_inside, pixels_outside, sum_inside, sum_outside, ellipsoid_index, background_adjusted_inside_sum

def reverse_log_transform(array, base = 1):
    '''
    Applies the inverse log (exp) function to convert logarithmic pixel encoding to linear encoding

    Args:
        array: a 2D numpy array representing the grayscale image; pixel values must be encoded as floats between 0 and 1
        base:  the base of the logarithmic encoding scale, which varies depending on format and camera manufacturer
               if base == "e", base will be set to 2.718281828459045 (inverse of natural logarithm)

    Returns:
        array_out: the modified 2D numpy array, where pixel values are linearly correlated with luminance

    Raises:
        Exception: image array values out of range (pixel values must be floats between 0 and 1)
    '''
    if base == "e":
        from math import e
        base = e

    # Substitute negative values with 0
    if array.min() < 0:
        array = np.where(array < 0, 0, array)

    if array.max() > 1 and array.max() < 2:
        print(f"reverse_log_transform warning: array max was {array.max()} and will be normalized to 1.0")
        array = array / array.max()
    elif array.max() >= 2:
        raise Exception("reverse_log_transform error: image array values out of range (greater than 2, but floats between 0 and 1 are expected)")

    array_out = np.power(array, base)

    return array_out