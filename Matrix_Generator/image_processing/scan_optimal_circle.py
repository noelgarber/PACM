import numpy as np
import cv2

def check_array_dims(array, expected_ndim, array_name = None):
    '''
    Simple function for checking if an object is a numpy array, and if so, whether it has the expected dimensions count

    Args:
        array: the object to check
        expected_ndim (int): the number of dimensions that are expected in the array

    Returns:
        has_error (bool): whether an error is present
        error_message (str): the error message; will be None if has_error is False
    '''
    if array_name is None:
        array_name = "input object"

    if isinstance(array, np.ndarray):
        if array.ndim != expected_ndim:
            raise TypeError(f"{array_name} is not a numpy array")
    else:
        raise TypeError(f"{array_name} has {array.ndim} dimensions, but {expected_ndim} were expected")

def spot_circle_scan(image_snippet, source_image, midpoint_coords, enforced_radius = None, alphanumeric_coords = None,
                     radius_variance_multiplier = 0.33, radius_shrink_multiplier = 0.9,
                     value_to_maximize = "inside_sum", verbose = False):
    '''
    Function for scanning through a 2D numpy array to find the circle representing the spot that maximizes the signal

    Args:
        image_snippet (np.ndarray):         grayscale snippet showing the spot being analyzed
        source_image (np.ndarray):          grayscale source image from which the image_snippet originates
        midpoint_coords (tuple):            coordinates of the inferred spot midpoint in source_image, given as (y,x)
        enforced_radius (int):              optional variable for enforcing a specified radius for the defined spot
        alphanumeric_coords (str):          a coordinate name, e.g. A1, for the spot in the source image
        radius_variance_multiplier (float): value between 0 and 1 that represents the bounds for scanning around the
                                            midpoint, as a fraction of the spot radius
        radius_shrink_multiplier (float):   value between 0 and 1 that is used to reduce the radius of the spot
        verbose (bool):                     whether to display debugging information

    Returns:
        final_midpoint (tuple):             coordinates of the final identified midpoint of the optimal defined spot
        results_dict (dict):                a dictionary of results values such that:
                                                "inside_sum" (float):      sum of pixel values within the defined circle
                                                "outside_sum" (float):     sum of pixel values outside the circle
                                                "inside_count" (int):      pixel count within the defined circle
                                                "outside_count" (int):     pixel count outside the defined circle
                                                "ellipsoid_index" (float): ratio of mean pixel values inside/outside
                                                "spot_midpoint" (tuple):   coordinates of the spot midpoint
    '''

    # Print input stats if verbose
    if verbose and alphanumeric_coords is not None:
        print(f"Running spot_circle_scan for midpoint coordinates {midpoint_coords} ({alphanumeric_coords}) in source_image of shape {source_image.shape}")
    elif verbose:
        print(f"Running spot_circle_scan for midpoint coordinates {midpoint_coords} in source_image of shape {source_image.shape}")

    # Check that input images are of correct type and dimensions
    check_array_dims(array = image_snippet, expected_ndim = 2, array_name = "image_snippet")
    check_array_dims(array = source_image, expected_ndim = 2, array_name = "source_image")

    # Check that the midpoint coordinates are valid
    if not 0 <= midpoint_coords[0] <= source_image.shape[0]-1 or not 0 <= midpoint_coords[1] <= source_image.shape[1]-1:
        raise IndexError(f"midpoint_coords at {midpoint_coords} are not within range for source_image of shape {source_image.shape}")

    # Check that the multipliers are valid
    if not 0 <= radius_variance_multiplier <= 1:
        raise Exception(f"spot_circle_scan error: radius_variance_multiplier is {radius_variance_multiplier}, but a value between 0 and 1 is required.")
    elif not 0 < radius_shrink_multiplier <= 1:
        raise Exception(f"spot_circle_scan error: radius_shrink_multiplier is {radius_shrink_multiplier}, but must be >0 and <=1.")

    # Declare the radius of the expected spot
    if enforced_radius is not None:
        spot_radius = int(enforced_radius * radius_shrink_multiplier)
    else:
        image_midpoint = (round(image_snippet.shape[0] / 2), round(image_snippet.shape[1] / 2))  # format is (y,x)
        spot_radius = int(min(image_midpoint) * radius_shrink_multiplier)

    # Declare the deviation amount, in pixels, from the spot image midpoint, to scan through when finding the optimal actual midpoint
    max_variance = int(radius_variance_multiplier * spot_radius)

    # Begin the cycle for scanning around the given coordinates to find optimal coordinates
    maximized_value = -1000
    y_midpoint, x_midpoint = midpoint_coords  # coordinates refer to source_image, not image snippet
    final_midpoint, results_dict = None, None

    for y_deviated in np.arange(y_midpoint - max_variance, y_midpoint + max_variance):
        for x_deviated in np.arange(x_midpoint - max_variance, x_midpoint + max_variance):
            current_midpoint = (y_deviated, x_deviated)
            inverted_current_midpoint = (x_deviated, y_deviated)  # cv2 takes (x,y) instead of (y,x)

            # Draw a circle (as a mask) around the defined current midpoint
            mask = np.zeros(source_image.shape, dtype=np.uint8)
            cv2.circle(mask, inverted_current_midpoint, spot_radius, 255, -1)

            # Calculate pixel value sum and count inside the defined circle
            inside_sum = np.sum(source_image[mask == 255])
            if inside_sum < 0:
                inside_sum = 0
            inside_count = len(source_image[mask == 255])

            # Find image snippet borders in source image
            y_top = y_midpoint - image_snippet.shape[0]
            y_bottom = y_midpoint + image_snippet.shape[0]
            x_left = x_midpoint - image_snippet.shape[1]
            x_right = x_midpoint + image_snippet.shape[1]

            # Avoid out-of-bounds errors
            if y_top < 0:
                y_top = 0
            if y_bottom > source_image.shape[0]-1:
                y_bottom = source_image.shape[0]-1
            if x_left < 0:
                x_left = 0
            if x_right > source_image.shape[1]-1:
                x_right = source_image.shape[1]-1

            # Calculate pixel value sum and count outside the circle, bounded by the image_snippet borders
            square_around_spot = source_image[y_top:y_bottom, x_left:x_right]
            square_mask_segment = mask[y_top:y_bottom, x_left:x_right]
            outside_sum = np.sum(square_around_spot[square_mask_segment == 0])
            if outside_sum < 0:
                outside_sum = 0
            outside_count = len(square_around_spot[square_mask_segment == 0])

            # Calculate the mean pixel value inside and outside the defined circle, and an index ratio of them
            mean_inside = inside_sum / inside_count
            mean_outside = outside_sum / outside_count
            if mean_inside != 0 and mean_outside != 0:
                ellipsoid_index = mean_inside / mean_outside
            elif mean_inside != 0 and mean_outside == 0:
                ellipsoid_index = 999
            else:
                ellipsoid_index = 0

            # Set the testing value based on the user-defined mode
            if value_to_maximize == "inside_sum":
                testing_value = inside_sum
            elif value_to_maximize == "mean_inside":
                testing_value = mean_inside
            elif value_to_maximize == "ellipsoid_index":
                testing_value = ellipsoid_index
            else:
                raise ValueError(f"spot_circle_scan error: value_to_maximize is set to {value_to_maximize}, but must be one of [\"inside_sum\", \"mean_inside\", \"ellipsoid_index\"]")

            # If the testing value is better than the last iteration, set the results dict
            if testing_value >= maximized_value:
                maximized_value = testing_value
                final_midpoint = current_midpoint
                results_dict = {
                    "inside_sum": inside_sum,
                    "outside_sum": outside_sum,
                    "inside_count": inside_count,
                    "outside_count": outside_count,
                    "ellipsoid_index": ellipsoid_index,
                    "spot_midpoint": current_midpoint,
                }

    if final_midpoint is None:
        raise Exception("spot_circle_scan error: final_midpoint is None")
    elif results_dict is None:
        raise Exception("spot_circle_scan error: results_dict is None")
    elif verbose:
        print(f"\tfinal_midpoint = {final_midpoint}")
        print(f"\tresults_dict = {results_dict}")

    return final_midpoint, results_dict