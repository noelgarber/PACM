

import numpy as np
import pandas as pd
import os
import string
import matplotlib.pyplot as plt
import csv
import warnings
from tifffile import imread, imwrite, imshow
from scipy.signal import find_peaks

#Function to convert a 2-column CSV file into a dictionary, where the first column holds keys and the second column holds values
def csv_to_dict(filepath):
    result = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key = row[0]
            value = row[1]
            result[key] = value
    return result

#Function to append elements to the value of a key-value pair in a dictionary, where the value is a list
def dict_value_append(input_dict, key, element_to_append):
    if input_dict.get(key) == None:
        input_dict[key] = [element_to_append]
    else:
        value_list = input_dict.get(key)
        value_list.append(element_to_append)
        input_dict[key] = value_list

#---------------------------------------------------------------------------------------------------------------------------------------------

'''
Define a new SpotArray class, which will contain the numpy arrays from a source tiff file
As arguments, it requires: 
    tiff_path = the path to the tiff file
    spot_dimensions = a tuple of (number of spots in width, number of spots in height)
Usage: 
    var = SpotArray(tiff_path = ______, spot_dimensions = _______, verbose = False/True, suppress_warnings = False/True)
'''
class SpotArray:
    def __init__(self, tiff_path, spot_dimensions, verbose = False, suppress_warnings = False):

        self.grid_shape = spot_dimensions

        # Initialize main grayscale image stored in tiff_path
        img = imread(tiff_path)
        try:
            layers = img.shape[2]
        except:
            layers = 1
        if layers > 1:grid_peak_finder(
            img = img[:,:,0]
            warnings.warn("SpotArray warning: multiple layers were found when importing " + tiff_path + "; the first layer was used.") if not suppress_warnings else None
        self.grayscale_array = img
        self.linear_array = self.reverse_log_transform(img) # This creates an image where all the pixel values are squared, restoring the linear correlation between pixel value and true brightness.

        # Auto-extract copy, scan, and probe information from filename
        filename = tiff_path.split("/")[-1].split(",")[0]
        self.copy_number, self.scan_number, self.probe_name = None, None, None
        print("\t\tcomputing background-adjusted signal and ellipsoid_index...") if verbose else None

        # Make a dictionary holding alphanumeric spot coordinates as keys, storing tuples of (background_adjusted_signal, ellipsoid_index, peak_intersect, top_left_corner)
        self.spot_info_dict = self.ellipsoid_constrain(spot_images = image_slices, dilation_factor = ellipsoid_dilation_factor, centering = center_spots, constrain_verbose = False)

        # Draw crosshairs on the individual spot peaks, which may not perfectly align with the hlinepeaks and vlinepeaks intersect points
        print("\t\tmaking image highlighting detected spots...") if verbose else None
        for element in filename.split("_"):
            if "Copy" in element or "copy" in element:
                self.copy_number = element[4:]
            elif "Scan" in element or "scan" in element:
                self.scan_number = element[4:]
            elif "Probe" in element or "probe" in element:
                self.probe_name = element[5:]

        # Analyze the array automatically with the default variables
        self.analyze_array()

    # Function to reverse the log transform on an image such that intensity linearly correlates with luminance.
    def reverse_log_transform(self):
        if self.grayscale_array.max() > 1:
            raise Exception("SpotArray.reverse_log_transform error: image array values out of range (expected: float between 0 and 1)")
        array_squared = np.power(self.grayscale_array, 2)
        return array_squared

    # High-level function that segments and analyzes the spots in the array
    def analyze_array(self, ellipsoid_dilation_factor = 1, show_sliced_image = False, show_crosshairs_image = False,
                      show_individual_spot_images = False, center_spots = True, verbose = False):
        print("\tProcessing: Copy", self.copy_number, " - Scan", self.scan_number, "- Probe", self.probe_name) if verbose else None
        print("\t\tfinding grid peaks...") if verbose else None

        self.vlpeaks_indices, self.vlmins_indices, self.hlinepeaks_indices, self.hlinemins_indices = self.grid_peak_finder(verbose = verbose)

        print("\t\tslicing image...") if verbose else None

        image_peak_coordinates, image_slices, self.sliced_image = self.image_slicer(image_ndarray = self.linear_array, vlinepeaks_indices = self.vlpeaks_indices, vlinemins_indices = self.vlmins_indices,
                                                                                    hlinepeaks_indices = self.hlinepeaks_indices, hlinemins_indices = self.hlinemins_indices,
                                                                                    render_sliced_image = True, slicer_debugging = show_individual_spot_images)

        # Display popup of sliced image if prompted
        imshow(self.sliced_image) if show_sliced_image else None
        plt.show() if show_sliced_image else None

        print("\t\tcomputing background-adjusted signal and ellipsoid_index...") if verbose else None

        # Make a dictionary holding alphanumeric spot coordinates as keys, storing tuples of (background_adjusted_signal, ellipsoid_index, peak_intersect, top_left_corner)
        self.spot_info_dict = self.ellipsoid_constrain(spot_images = image_slices, dilation_factor = ellipsoid_dilation_factor, centering = center_spots, verbose = False)

        # Draw crosshairs on the individual spot peaks, which may not perfectly align with the hlinepeaks and vlinepeaks intersect points
        print("\t\tmaking image highlighting detected spots...") if verbose else None

        self.sliced_image_crosshairs = self.draw_crosshairs(color_image = self.sliced_image, spot_info = self.spot_info_dict, crosshair_width = 5)

        # Display popup of sliced image with drawn crosshairs if prompted
        imshow(self.sliced_image_crosshairs / self.sliced_image_crosshairs.max()) if show_crosshairs_image else None
        plt.show() if show_crosshairs_image else None

        # In addition to assigning to internal variables, also returns a tuple of results that can be optionally assigned when this function is invoked
        analyzed_array_tuple = (self.copy_number, self.scan_number, self.probe_name, self.linear_array, self.spot_info_dict, self.sliced_image_crosshairs)
        return analyzed_array_tuple

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Function to define the coordinates of the spot array grid. 
    Uses the handle_mismatch() and infer_peaks() methods in this class. 
        - First creates lists of the pixel intensity sums of vertical and horizontal lines of pixels in self.linear_array. 
        - Then uses scipy.signal.find_peaks() to find peaks and valleys in these lists. 
    Outputs results as a tuple of (vertical_line_peaks, vertical_line_mins, horizontal_line_peaks, horizontal_line_mins) where: 
        vertical_line_peaks, vertical_line_mins = peaks and minima in the sums of vertical lines of pixels across the horizontal axis of the image
        horizontal_line_peaks, horizontal_line_mins = peaks and minima in the sums of horizontal lines of pixels across the vertical axis of the image
    '''
    def grid_peak_finder(self, verbose = False):
        grid_width, grid_height = self.grid_shape

        # Find the sums of vertical and horizontal lines of pixels in the grayscale image array
        vlsums, hlsums = self.linear_array.sum(axis=0), self.linear_array.sum(axis=1)

        vlpeaks, _ = find_peaks(vlsums)
        vlmins, _ = find_peaks(vlsums * -1)
        hlpeaks, _ = find_peaks(hlsums)
        hlmins, _ = find_peaks(hlsums * -1) actual_peaks_count = the number of detected peaks

        if len(vlpeaks) != grid_width:
            vlpeaks, vlmins = handle_mismatch(line_sums = vlsums, actual_peaks = vlpeaks, actual_mins = vlmins,
                                              expected_peaks_count = grid_width, line_axis_name = "vertical",
                                              tolerance_spot_frac = 0.25, verbose = verbose)
        else:
            print("\t\t\tfound correct number of vertical line peaks")

        if len(hlpeaks) != grid_height:
            hlpeaks, hlmins = handle_mismatch(line_sums = hlsums, actual_peaks=hlpeaks, actual_mins=hlmins,
                                              expected_peaks_count = grid_height, line_axis_name = "horizontal",
                                              tolerance_spot_frac = 0.25, verbose = verbose)
        else:
            print("\t\t\tfound correct number of horizontal line peaks")

        return (vlpeaks, vlmins, hlpeaks, hlmins)

    '''
    Function to conditionally apply infer_peaks() when there is a mismatch between the detected peak count and the expected peak count. 
    As input, it takes: 
        line_sums = array of vertical or horizontal line sums
        actual_peaks = the detected peaks (which are given as indices referring to the original line_sums)
        actual_mins - the detected mins (indices referring to line_sums)
        expected_peaks_count = the number of peaks that are expected based on the grid dimensions (number of spots expected)
        line_axis_name = the line sum axis; must be "vertical" or "horizontal"
        tolerance_spot_frac = the fraction of the spot dimension (in pixels) that is the allowed distance between peaks for them to be declared mergeable
        extra_peaks_proportion = the fraction of the expected peak count that is allowed for the collapse_extra_peaks method to be used
    It returns: 
        output_line_peaks = new array of line peaks based on conditionally applying infer_peaks()
        output_line_mins = new array of line mins based on conditionally applying infer_peaks()
    '''
    def handle_mismatch(self, line_sums, actual_peaks, actual_mins, expected_peaks_count, line_axis_name,
                        tolerance_spot_frac = 0.25, extra_peaks_proportion = 0.25, verbose = False):
        actual_peaks_count = len(actual_peaks)
        extra_peaks_ceiling = (extra_peaks_proportion + 1) * expected_peaks_count

        print("\t\t\tgrid_peak_finder warning: found", actual_peaks_count, line_axis_name, "line peaks, but", expected_peaks_count, "were expected.") if verbose else None

        if actual_peaks_count < expected_peaks_count or actual_peaks_count > extra_peaks_ceiling:
            print("\t\t\t\tinferring", line_axis_name, "line peaks from dimensions...") if verbose else None
            output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        elif actual_peaks_count > expected_peaks_count and actual_peaks_count <= extra_peaks_ceiling:
            print("\t\t\t\taveraging extra", line_axis_name, "line peaks that are within", tolerance_spot_frac * 100, "% of average spot dimension...") if verbose else None
            output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, collapse_extra_peaks = True,
                                                                   detected_peaks=actual_peaks, tolerance_spot_frac = tolerance_spot_frac, verbose = verbose)
            if len(output_line_peaks) != expected_peaks_count:
                print("\t\t\t\tfailed to correct number of peaks by averaging within the tolerance; reverting to inferring peaks by grid dimensions...") if verbose else None
                output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        else:
            output_line_peaks, output_line_mins = actual_peaks, actual_mins

        return output_line_peaks, output_line_mins

    '''
    Infers line peak indices based on grid dimensions (length or width) when searching for actual peaks is not possible. 
    As input, takes: 
        line_sums = vertical_line_sums or horizontal_line_sums
        grid_dimension_length = grid_width or grid_height, respectively
    Important: 
        For infer_peaks() to generate valid results, input images must be cropped right to the border of the spots on all sides, with no extra black space
    Outputs a NumPy array of peaks. 
    '''
    def infer_peaks(self, line_sums, expected_peaks, collapse_extra_peaks = False, detected_peaks = None, tolerance_spot_frac = 0.25, verbose = False):
        mean_spot_dimension = len(line_sums) / expected_peaks

        inferred_line_peaks = np.arange(expected_peaks) * mean_spot_dimension
        inferred_line_peaks = inferred_line_peaks + (mean_spot_dimension / 2)  # starts halfway across the first inferred spot square, making the assumption that the peak is in the middle
        inferred_line_peaks = inferred_line_peaks.round().astype(int)  # rounds and gives integers, as indices must be ints

        inferred_line_mins = np.arange(expected_peaks + 1) * mean_spot_dimension
        inferred_line_mins = inferred_line_mins.round().astype(int)
        if inferred_line_mins[-1] > (len(line_sums) - 1):
            inferred_line_mins[-1] = len(line_sums) - 1  # catches error where the ending number, rounded up, might otherwise go out of bounds

        if collapse_extra_peaks:
            print("\t\t\t\tcollapsing extra peaks (detected" + str(len(detected_peaks)) + ")...")
            peak_deltas = detected_peaks[1:] - detected_peaks[0:-1]
            tolerated_delta = mean_spot_dimension * tolerance_spot_frac
            deltas_indices = np.where(peak_deltas <= tolerated_delta)[0]

            append_peaks = np.empty(0)
            remove_peaks = np.empty(0)
            for delta_index in deltas_indices:
                mean_detected_peak = round((detected_peaks[delta_index] + detected_peaks[delta_index + 1]) / 2)
                append_peaks = np.append(append_peaks, mean_detected_peak)
                remove_peaks = np.append(remove_peaks, [delta_index, delta_index + 1])
            append_peaks, remove_peaks = append_peaks.astype(int), remove_peaks.astype(int)

            line_peaks = detected_peaks.copy()
            line_peaks = np.delete(line_peaks, remove_peaks)
            line_peaks = np.append(line_peaks, append_peaks)
            line_peaks = np.sort(line_peaks)

            print("\t\t\t\tdone; collapsed", len(detected_peaks), "peaks to", len(line_peaks)) if verbose else None

            line_mins = inferred_line_mins # Currently does not apply the collapse_extra_peaks method to minima, but this feature may be added in a later release
        else:
            line_peaks = inferred_line_peaks
            line_mins = inferred_line_mins

        return line_peaks, line_mins

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Takes a complete spot array image as a 2D numpy array and slices it into images of each individual spot. 
    To do this, it takes inputs of the indices of vertical and horizontal line peaks and minima. 
    If render_sliced_image is set to True, it will also draw a new image showing the slice lines. 
    '''
    def image_slicer(self, image_ndarray, vlinepeaks_indices, vlinemins_indices, hlinepeaks_indices, hlinemins_indices,
                     render_sliced_image = True, slicer_debugging = False):
        if render_sliced_image:
            max_pixel = image_ndarray.max()
            color_image = np.repeat(image_ndarray[:,:,np.newaxis], 3, axis=2) #Red=[:,:,0], Green=[:,:,1], Blue=[:,:,2]

        alphabet = list(string.ascii_uppercase)  # Used for declaring coordinates later

        vlpeaks_prev_mins, vlpeaks_next_mins = self.mins_between_peaks(vlinepeaks_indices, vlinemins_indices)
        hlpeaks_prev_mins, hlpeaks_next_mins = self.mins_between_peaks(hlinepeaks_indices, hlinemins_indices)

        peak_coordinates_dict = {}
        sliced_spot_dict = {}
        for i, horizontal_peak in enumerate(hlinepeaks_indices):
            row_letter = alphabet[i]
            for j, vertical_peak in enumerate(vlinepeaks_indices):
                col_number = j + 1
                alphanumeric_coordinates = row_letter + str(col_number)

                horizontal_prev_min = int(
                    hlpeaks_prev_mins.get(horizontal_peak))  # horizontal peaks are along the vertical axis
                horizontal_next_min = int(hlpeaks_next_mins.get(horizontal_peak))
                vertical_prev_min = int(
                    vlpeaks_prev_mins.get(vertical_peak))  # vertical peaks are along the horizontal axis
                vertical_next_min = int(vlpeaks_next_mins.get(vertical_peak))

                peak_coordinates = (horizontal_peak, vertical_peak)  # (height, width)
                peak_coordinates_dict[alphanumeric_coordinates] = peak_coordinates

                sliced_spot = image_ndarray[horizontal_prev_min:horizontal_next_min,
                              vertical_prev_min:vertical_next_min]  # height range, width range
                if slicer_debugging:
                    imshow(sliced_spot, cmap="gray")
                    plt.show()

                top_left_corner = (horizontal_prev_min, vertical_prev_min)  # height x width
                sliced_spot_dict[alphanumeric_coordinates] = (top_left_corner, sliced_spot)

        if render_sliced_image:
            # Mark peaks with blue lines
            for horizontal_peak in hlinepeaks_indices:
                color_image[:,:,2][horizontal_peak, :] = max_pixel
                color_image[:,:,0][horizontal_peak, :] = 0
                color_image[:,:,1][horizontal_peak, :] = 0
            for vertical_peak in vlinepeaks_indices:
                color_image[:,:,2][:, vertical_peak] = max_pixel
                color_image[:,:,0][:, vertical_peak] = 0
                color_image[:,:,1][:, vertical_peak] = 0

            # Mark mins (borders) with red lines
            for horizontal_min in hlinemins_indices:
                color_image[:,:,0][horizontal_min, :] = max_pixel
                color_image[:,:,1][horizontal_min, :] = 0
                color_image[:,:,2][horizontal_min, :] = 0
            for vertical_min in vlinemins_indices:
                color_image[:,:,0][:, vertical_min] = max_pixel
                color_image[:,:,1][:, vertical_min] = 0
                color_image[:,:,2][:, vertical_min] = 0

        if render_sliced_image:
            return peak_coordinates_dict, sliced_spot_dict, color_image
        else:
            return peak_coordinates_dict, sliced_spot_dict

    '''
    In a 1D array of peaks generated by scipy.signal.find_peaks(), finds the corresponding minima on either side of the peaks. 
    As input, takes a 1D array of peak values from find_peaks(array) and a 1D array of min values from find_peaks(array*-1). 
    Returns two dictionaries showing the previous and next minima on either side of a particular peak. 
    '''
    def mins_between_peaks(self, peaks_array, mins_array):
        # where peaks_array contains indices of peaks in an image
        next_mins_dict = {}
        previous_mins_dict = {}

        inter_peak_spaces = np.empty(0, dtype=int)
        for i, peak in enumerate(peaks_array[0:-1]):
            next_peak = peaks_array[i + 1]
            next_mins = self.array_between(mins_array, peak, next_peak)
            next_min = next_mins.mean()

            next_mins_dict[peak] = next_min
            previous_mins_dict[next_peak] = next_min

            inter_peak_space = next_peak - peak
            inter_peak_spaces = np.append(inter_peak_spaces, inter_peak_space)

        inter_peak_space = inter_peak_spaces.mean()
        previous_mins_dict[peaks_array[0]] = peaks_array[0] - (inter_peak_space / 2)
        next_mins_dict[peaks_array[-1]] = peaks_array[-1] + (inter_peak_space / 2)

        return previous_mins_dict, next_mins_dict

    '''
    Finds values in a numpy array that are between a minimum and maximum value.
    Output values are returned as np.ndarray. 
    '''
    def array_between(self, numpy_array, min_threshold, max_threshold):
        boolean_array = np.logical_and(numpy_array > min_threshold, numpy_array < max_threshold)
        in_range_indices = np.where(boolean_array)[0]
        in_range_values = np.empty((0))
        for i in in_range_indices:
            in_range_values = np.append(in_range_values, numpy_array[i])
        return in_range_values

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Function that returns a dictionary of spot coordinates where the value is a tuple of (background-adjusted signal, ellipsoid_index)
    Uses the center_spot() method in this class, found below. 
    Inputs: 
        spot_images = a dictionary of spot coordinates where the value is a tuple of (top_left_corner, spot_image array)
        dilation_factor = a multiplier to enlarge (>1) or constrict (<1) the defined constraining ellipsoid
        centering = a boolean
            True => uses center_spot() to find the peak intersect point to define a surrounding ellipsoid.
            False => sets the center point of the spot as equal to the center point of the spot image for defining a surrounding ellipsoid. 
        return_coordinates_list = a boolean for whether to return an additional simple list of alphanumeric spot coordinates.
    Outputs in returned dict: 
        background_adjusted_signal = (sum of pixel values inside the ellipsoid) - (mean pixel value outside the ellipsoid)*(number of pixels inside the ellipsoid)
        ellipsoid_index = (mean pixel value inside the ellipsoid) / (mean pixel value outside the ellipsoid)
            Values >>> 1 represent signals which are highly constrained to the ellipsoid. 
            Values ~1 represent smears that are not ellipsoidal. 
            Values <1 usually represent situations where most of the signal is non-specific bleed-over from neighbouring spots. 
    '''
    def ellipsoid_constrain(self, spot_images, dilation_factor = 1, centering = True, return_coordinates_list = False, verbose = False):
        print("\t\t\trunning ellipsoid_constrain()...") if verbose else None

        output_dict = {}

        if return_coordinates_list:
            coordinates_list = []

        for spot_coordinates, value in spot_images.items():
            top_left_corner, spot_image = value
            print("\t\t\ttop left corner of current spot:", top_left_corner) if verbose else none

            spot_image_height = len(spot_image)
            spot_image_width = len(spot_image[0])

            print("\t\t\tSpot image height: ", spot_image_height, "px", "\n\t\t\tSpot image width: ", spot_image_width, "px") if verbose else None

            '''
            The ellipsoid equation is (x-c)^2/(a^2) + (y-d)^2/(b^2) = 1, where <1 is inside the ellipsoid, and >1 is outside. 
                a = horizontal radius (stretch)     b = vertical radius (stretch)
                c = horizontal translation          d = vertical translation
                The middle of the circle is a point at (c, d). 
                    CAUTION: The coordinates are reversed for NumPy image arrays, which are a vertical list of horizontal lines. 
                    For arrays, the midpoint is (d, c). Coordinates are effectively (y,x) rather than (x,y). 
            '''

            if centering:
                peak_intersect, radii = self.center_spot(spot_image = spot_image, tolerance_fraction = 0.5, tolerance_mode = "whole", verbose = verbose)
                radii[0] = radii[0] * dilation_factor
                radii[1] = radii[1] * dilation_factor

            else:
                radii = [round((spot_image_height / 2) * dilation_factor), round((spot_image_width / 2) * dilation_factor)]
                peak_intersect = (radii[0], radii[1])

            print("\t\t\tpeak_intersect =", peak_intersect, "\n\t\t\tvertical_radius =", vertical_radius,
                  "\n\t\t\thorizontal_radius =", horizontal_radius, "\n\t\t\tdilation factor =", dilation_factor) if verbose else None

            a, b = horizontal_radius, vertical_radius
            c, d = peak_intersect[1], peak_intersect[0]

            print("\t\t\tThe inside-ellipsoid equation is (x-c)^2/(a^2) + (y-d)^2/(b^2) < 1",
                  "\n\t\t\tHere, the equation is (x-" + str(c) + ")^2/(" + str(a) + "^2) + (y-" + str(d) + ")^2/(" + str(b) + "^2) < 1,",
                  "\n\t\t\twhere x and y are the pixel coordinates that may or may not be within the ellipsoid.") if verbose else None

            pixels_inside, pixels_outside = 0, 0
            sum_intensities_inside, sum_intensities_outside = 0, 0

            for i, row in enumerate(spot_image):
                print("\t\t\tWorking on row", i, "of spot_image...") if verbose else None
                for j, pixel_value in enumerate(row):
                    print("\t\t\tevaluating column index", j, "in this row...", "\n\t\t\tpixel_value is", pixel_value) if verbose else None

                    values_dict = {
                        "x": j,
                        "y": i,
                        "a": a,
                        "b": b,
                        "c": c,
                        "d": d
                    }

                    pixel_is_inside = self.ellipsoid_evaluator(values_dict, verbose = verbose)
                    print("\t\t\tpixel_is_inside =", pixel_is_inside) if verbose else None

                    if pixel_is_inside:
                        pixels_inside += 1
                        sum_intensities_inside += pixel_value
                    else:
                        pixels_outside += 1
                        sum_intensities_outside += pixel_value

            mean_intensity_inside = sum_intensities_inside / pixels_inside
            mean_intensity_outside = sum_intensities_outside / pixels_outside

            # Assume that everything outside the circle is background, and subtract to get the true signal
            background_adjusted_signal = sum_intensities_inside - (pixels_inside * mean_intensity_outside)

            '''
            Make an index where >>>1 means strong positive, ~1 means negative/smear, 
            and <1 indicates that the majority of the signal comes from neighbouring spots bleeding over.
            '''
            ellipsoid_index = mean_intensity_inside / mean_intensity_outside

            # To the output dict, add a tuple containing the background-adjusted signal and the ellipsoid index
            output_dict[spot_coordinates] = (background_adjusted_signal, ellipsoid_index, peak_intersect, top_left_corner)
            if return_coordinates_list:
                coordinates_list.append(spot_coordinates)

        '''
        Returns a dictionary where the key is spot coordinates and the value is a tuple containing 
        (background-adjusted signal, ellipsoid index)
        '''
        if return_coordinates_list:
            return output_dict, coordinates_list
        else:
            return output_dict

    #----------
    '''
    Function that finds the center and radii of a sliced spot image.
    Marks the centre of the spot as the intersection of the peak values for horizontal and vertical summed lines in the image array. 
    If more than one peak is found in either the horizontal or vertical dimension, the peaks are averaged. 
    Inputs: 
        spot_image = the sliced spot image as a numpy array
        tolerance_fraction = a value from 0 to 1 for the allowed deviation from image center while finding peaks/crosshairs/actual spot center
            0 = no tolerance
            1 = total tolerance
        tolerance_mode = "whole" or "subslice"
            "whole": find_peaks applied to whole image and then nudged based on tolerance_fraction
            "subslice": find_peaks applied to subslice of spot image sliced based on tolerance_fraction
    Returns: 
        peak_intersect = (height, width) as a point
        vertical_radius, horizontal_radius = the radii of the spot, defined by the distance to the nearest image border from the peak_intersect
    '''

    def center_spot(self, spot_image, tolerance_fraction, tolerance_mode = "whole", verbose = False):
        print("\t\t\trunning center_spot()...") if verbose else None

        spot_image_height, spot_image_width = len(spot_image), len(spot_image[0])

        print("\t\t\tspot_image_height =", spot_image_height, "\n\t\t\tspot_image_width =", spot_image_width,
              "\n\t\t\tfinding midpoints...") if verbose else None

        image_midpoints = (round(spot_image_height / 2), round(spot_image_width / 2)) # vertical midpoint, horizontal midpoint
        print("\t\t\timage vertical midpoint =", image_midpoints[0], "\n\t\t\timage horizontal midpoint =", image_midpoints[1],
              "\n\t\t\tappying the tolerance_fraction of", tolerance_fraction, "...") if verbose else None

        tolerated_variances = (round(tolerance_fraction * image_midpoints[0]), round(tolerance_fraction * image_midpoints[1]))
        print("\t\t\ttolerated_vertical_variance =", tolerated_variances[0], "\n\t\t\ttolerated_horizontal_variance =", tolerated_variances[1],
              "\n\t\t\tfinding spot_vlinesums and spot_hlinesums...") if verbose else None

        if tolerance_mode == "subslice":
            spot_vlinesums = spot_image[:,image_midpoints[1] - tolerated_variances[1]: image_midpoint[1] + tolerated_variances[1]].sum(axis=0)
            spot_hlinesums = spot_image[image_midpoints[0] - tolerated_variances[0]: image_midpoints[0] + tolerated_variances[0],:].sum(axis=1)
        elif tolerance_mode == "whole":
            spot_vlinesums = spot_image.sum(axis=0)
            spot_hlinesums = spot_image.sum(axis=1)
        else:
            raise Exception("Error in center_spot: tolerance_mode \"" + tolerance_mode + "\" is not an accepted mode. Expected \"whole\" or \"subslice\").")

        print("\t\t\tspot_vlinesums:", spot_vlinesums, "\n\t\t\tspot_hlinesums:", spot_hlinesums) if verbose else None

        # Note that if tolerance_mode is "subslice", these values are unadjusted subslice indices
        spot_vlinepeaks, _ = find_peaks(spot_vlinesums)
        spot_hlinepeaks, _ = find_peaks(spot_hlinesums)

        print("\t\t\tspot_vlinepeaks:", spot_vlinepeaks, "\n\t\t\tspot_hlinepeaks:", spot_hlinepeaks,
              "\n\t\t\ttaking mean of peaks...") if verbose else None

        # Get single mean peak
        spot_vlinepeak = self.collapse_peaks(peaks=spot_vlinepeaks, values=spot_vlinesums)
        spot_hlinepeak = self.collapse_peaks(peaks=spot_hlinepeaks, values=spot_hlinesums)

        print("\t\t\tspot_vlinepeak:", spot_vlinepeak, "\n\t\t\tspot_hlinepeak:", spot_hlinepeak) if verbose else None

        if tolerance_mode == "subslice":
            # Adjust indices of slice to apply to whole spot image
            print("\t\t\tperforming index adjustment...") if verbose else None
            spot_vlinepeak = spot_vlinepeak + (image_midpoints[1] - tolerated_variances[1])
            spot_hlinepeak = spot_hlinepeak + (image_midpoints[0] - tolerated_variances[0])
        elif tolerance_mode == "whole":
            # Nudge indices to obey tolerance_fraction
            print("\t\t\tnudging peaks to obey tolerated variances...") if verbose else None
            spot_vlinepeak = self.nudge_peak(peak=spot_vlinepeak, tolerated_variance=tolerated_variances[1],
                                        spot_midpoint=image_midpoints[1])
            spot_hlinepeak = self.nudge_peak(peak=spot_hlinepeak, tolerated_variance=tolerated_variances[0],
                                        spot_midpoint=image_midpoints[0])

        peak_intersect = (spot_hlinepeak, spot_vlinepeak)

        print("\t\t\tcorrected spot_vlinepeak:", spot_vlinepeak, "\n\t\t\tcorrected spot_hlinepeak:", spot_hlinepeak,
              "\n\t\t\tpeak_intersect:", peak_intersect) if verbose else None

        # Find nearest distance to border and use this for the radius, for both dimensions
        print("\t\t\tCalculating nearest distance to borders for use as vertical and horizontal radii...") if verbose else None

        # Make a list of vertical and horizontal radii, respectively
        radii = [self.nearest_border(index=spot_hlinepeak, length=spot_image_height, verbose=True), self.nearest_border(index=spot_vlinepeak, length=spot_image_width, verbose=True)]

        return peak_intersect, radii

    '''
    Simple function to evaluate whether a point defined by (x,y) is within a defined ellipsoid. 
    The ellipsoid is defined by the equation x^2+y^2=1, with scaling factors. 
    Returns a truth value for whether the point is inside the ellipsoid. 
    '''
    def ellipsoid_evaluator(self, values_dict, return_value = False, verbose = False):
        print("\t\t\t", values_dict) if verbose else None

        value = ((values_dict.get("x") - values_dict.get("c")) ** 2) / (values_dict.get("a") ** 2) + ((values_dict.get("y") - values_dict.get("d")) ** 2) / (values_dict.get("b") ** 2)
        print("\t\t\t((x-c)**2)/(a**2) + ((y-d)**2)/(b**2) =", value) if verbose else None

        if value <= 1:
            inside = True
        elif value > 1:
            inside = False

        print("\t\t\tinside =", inside) if verbose else None

        if return_value:
            return value, inside
        else:
            return inside

    '''
    Simple function to take a list of peaks from find_peaks() and collapse it to a single mean peak. 
    Input: 
        peaks = list of peaks (indices)
        values = the values that find_peaks() was originally applied to
    Output: 
        peak = a single integer
    '''
    def collapse_peaks(self, peaks, values):
        if len(peaks) > 0:
            peak = round(peaks.mean())
        else:
            print("\t\t\tWarning: no peaks; defaulting to center.")
            peak = round(len(values) / 2)
        return peak

    # Simple function to check if a peak index is within the tolerated bounds, and if not, nudge it to the inner edge of the bounds.
    def nudge_peak(self, peak, tolerated_variance, spot_midpoint):
        if peak < (spot_midpoint - tolerated_variance):
            peak = spot_midpoint - tolerated_variance
        elif peak > (spot_midpoint + tolerated_variance):
            peak = spot_midpoint + tolerated_variance
        return peak

    # Finds the distance to the nearest border to a given index in a range of indices.
    def nearest_border(self, index, length, verbose = False):
        if (index - 0) <= (length - index):
            radius = index - 0
            if verbose:
                print("\t\t\tradius from distance to index[0]:", radius)
        else:
            radius = length - index
            if verbose:
                print("\t\t\tradius from distance to index[-1]:", radius)
        return radius

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Function to draw crosshairs on the true peak points for each spot in the unsliced image. 
    As input, takes: 
        image = an image as a numpy array
        spot_info = the dictionary returned by ellipsoid_constrain()
        crosshair_length = the length, in pixels, of the crosshair lines to be drawn
        crosshair_brightness = the brightness of the drawn crosshairs as a float pixel intensity (max. 1.0)
        crosshair_width = the width of the crosshair lines. Must be an odd integer; if not odd, +1 will be added. 
    Returns the image with crosshairs drawn on all the spots. 
    '''
    def draw_crosshairs(self, color_image, spot_info, crosshair_width = 3):
        max_green_pixel = color_image[:,:,1].max()

        if crosshair_width % 2 == 0:
            crosshair_width = crosshair_width + 1  # catches even widths

        deviation = int((crosshair_width - 1) / 2)

        for spot_coordinates, value_tuple in spot_info.items():
            background_adjusted_signal, ellipsoid_index, peak_intersect, top_left_corner = value_tuple
            real_peak_intersect = (top_left_corner[0] + peak_intersect[0], top_left_corner[1] + peak_intersect[1])

            # Draw horizontal green crosshair
            color_image[:,:,1][real_peak_intersect[0] - deviation: real_peak_intersect[0] + deviation, real_peak_intersect[1] - crosshair_width: real_peak_intersect[1] + crosshair_width] = max_green_pixel
            color_image[:,:,[0,2]][real_peak_intersect[0] - deviation: real_peak_intersect[0] + deviation, real_peak_intersect[1] - crosshair_width: real_peak_intersect[1] + crosshair_width] = 0

            # Draw vertical green crosshair
            color_image[:,:,1][real_peak_intersect[0] - crosshair_width: real_peak_intersect[0] + crosshair_width, real_peak_intersect[1] - deviation: real_peak_intersect[1] + deviation] = max_green_pixel
            color_image[:,:,[0,2]][real_peak_intersect[0] - crosshair_width: real_peak_intersect[0] + crosshair_width, real_peak_intersect[1] - deviation: real_peak_intersect[1] + deviation] = 0

        return color_image
    #-------------------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------------
#Begin processing the images

print("Please enter the dimensions of the array (number of spots in width x number of spots in height).")
spot_grid_width = int(input("Width (number of spots):  "))
spot_grid_height = int(input("Height (number of spots):  "))
spot_grid_dimensions = (spot_grid_width, spot_grid_height)
print("-----------------------")

image_directory = input("Enter the full directory where TIFF images are stored: ")
filenames_list = os.listdir(image_directory)
print("Loading and processing files as SpotArray objects...")
spot_arrays = []
for filename in filenames_list: 
    print("\tLoading", filename)
    file_path = os.path.join(image_directory, filename)
    spot_array = SpotArray(tiff_path = file_path, spot_dimensions = spot_grid_dimensions, verbose = True)
    spot_arrays.append(spot_array)

print("-----------------------")

#TODO Change the dataframe assembly process to work with SpotArray class objects

print("Assembling dataframe and saving images...")

data_df = pd.DataFrame() #initialize blank dataframe

output_directory = os.path.join(os.getcwd(), "image_prep_output")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

rlt_output_directory = os.path.join(output_directory, "reverse_log_transformed_images")
if not os.path.exists(rlt_output_directory):
    os.makedirs(rlt_output_directory)

crosshairs_output_directory = os.path.join(output_directory, "crosshairs-marked_spot_images")
if not os.path.exists(crosshairs_output_directory):
    os.makedirs(crosshairs_output_directory)

bas_cols_dict = {} #Dictionary of lists of background-adjusted signal column names, where the key is the probe name
ei_cols_dict = {} #Dictionary of lists of ellipsoid index column names, where the key is the probe name
new_cols_dict = {} #Dictionary that includes both of the above, along with the copy and scan numbers, in the form of (copy, scan, bas_col, ei_col)

for analyzed_array_tuple in analyzed_array_images: 
    copy, scan, probe, array_image, spot_info_dict, sliced_image_crosshairs = analyzed_array_tuple

    col_prefix = probe + "\nCopy " + str(copy) + "\nScan " + str(scan)
    bas_col = col_prefix + "\nBackground-Adjusted_Signal"
    ei_col = col_prefix + "\nEllipsoid_Index"

    #Assign column names to dict by probe name
    dict_value_append(bas_cols_dict, probe, bas_col)
    dict_value_append(ei_cols_dict, probe, ei_col)
    dict_value_append(new_cols_dict, probe, (copy, scan, bas_col, ei_col))

    #Assign dataframe values
    for spot_coord, signal_tuple in spot_info_dict.items(): 
        background_adjusted_signal, ellipsoid_index, _, _ = signal_tuple

        data_df.at[spot_coord, bas_col] = background_adjusted_signal
        data_df.at[spot_coord, ei_col] = ellipsoid_index

    #Save modified image
    imwrite(os.path.join(rlt_output_directory, "Copy" + str(copy) + "_Scan" + str(scan) + "_" + probe + "_reverse-log-transform.tif"), array_image)
    imwrite(os.path.join(crosshairs_output_directory, "Copy" + str(copy) + "_Scan" + str(scan) + "_" + probe + "_crosshairs.tif"), sliced_image_crosshairs)

#Declare probe order for sorting dataframe columns
probes_ordered = []
input_probe_order = input("Would you like to specify the order of probes for sorting columns? (Y/N)  ")
if input_probe_order == "Y":
    print("\tThe probes in this dataset are:", list(ei_cols_dict.keys()))
    print("\tPlease enter the probes in the order you wish them to appear. Hit enter when done.")
    no_more_probes = False
    while not no_more_probes:
        next_probe = input("Probe name:  ")
        if next_probe != "":
            probes_ordered.append(next_probe)
        else:
            no_more_probes = True
else:
    probes_ordered = list(ei_cols_dict.keys())
    print("\tUsing arbitrary probe order:", probes_ordered)

#Sorting dataframe and testing significance of hits
print("Organizing dataframe...")

sorted_cols = ["Peptide_Name"] #Adds a column to receive peptide names later
data_df.insert(0, "Peptide_Name", "")
for current_probe in probes_ordered:
    col_tuples = new_cols_dict.get(current_probe)
    col_tuples = sorted(col_tuples, key = lambda x: x[0]) #Sorts by copy number
    new_cols_dict[current_probe] = col_tuples
    for col_tuple in col_tuples:
        sorted_cols.append(col_tuple[2]) #Appends background_adjusted_signal column name
        sorted_cols.append(col_tuple[3]) #Appends ellipsoid_index column name
    data_df.insert(1, current_probe + "_call", "")
    sorted_cols.append(current_probe + "_call")

data_df = data_df[sorted_cols]

#Test significance
print("Testing significance of hits...")
ei_sig_thres = float(input("\tEnter the ellipsoid index threshold above which a hit is considered significant:  "))

for current_probe in probes_ordered:
    call_col = current_probe + "_call"
    ei_cols = ei_cols_dict.get(current_probe)
    data_df[call_col] = data_df.apply(lambda x: "Pass" if (x[ei_cols] > ei_sig_thres).all() else "", axis = 1)

#Add peptide names
add_names = input("Add peptide names from CSV file mapping coordinates to names? (Y/N)  ")
if add_names == "Y":
    names_path = input("\tEnter the path containing the CSV with coordinate-name pairs:  ")
    names_dict = csv_to_dict(names_path)
    print("Dictionary of coordinate-name pairs:")
    print(names_dict)

for i, row in data_df.iterrows():
    print("Row:", i)
    pep_name = names_dict.get(i)
    print("Peptide name:", pep_name)
    data_df.at[i, "Peptide_Name"] = pep_name

data_df.to_csv(os.path.join(output_directory, "preprocessed_data.csv"))

print("Done!")