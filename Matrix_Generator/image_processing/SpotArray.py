# This script defines the SpotArray class for representing and processing arrays of spots arranged as a grid.

import numpy as np
import os
import string
from Matrix_Generator.image_processing.image_utils import reverse_log_transform
from tifffile import imread, imshow
from scipy.signal import find_peaks

class SpotArray:
    '''
    SpotArray class to contain spot image data and methods

    This class contains spot image data as 2D numpy arrays, metadata, and quantitation methods.

    Attributes:
        self.grid_shape (tuple):              (number of spots in width, number of spots in height)
        self.filename (str):                  the filename pointing to the source TIFF image

        self.copy_number (str):               the copy/replicate number, parsed from the filename
        self.scan_number (str):               the scan order number, for when spot blots have been probed multiple times; it is a good
                                              idea to record this due to the possibility of signal carryover from incomplete stripping
        self.probe_name (str):                the name of the probe (e.g. a recombinant protein or antibody)

        self.grayscale_array (np.ndarray):    original grayscale array from the inputted tiff image, as a 2D numpy array
        self.linear_array (np.ndarray):       linearized version of grayscale_array, if it used log pixel encoding
        self.image_shape (tuple):             (pixels in width, pixels in height), i.e. opposite of numpy (x,y)

        self.vlpeaks_indices (np.ndarray):    indices for vertical line peaks along the horizontal axis of the image
        self.vlmins_indices (np.ndarray):     indices for vertical line minima along the horizontal axis of the image
        self.hlinepeaks_indices (np.ndarray): indices for horizontal line peaks along the vertical axis of the image
        self.hlinemins_indices (np.ndarray):  indices for horizontal line peaks along the vertical axis of the image

        self.sliced_image (np.ndarray):       color image showing red borders between spots and mean peak coords in blue
        self.outlined_image (arr):            the sliced image with outlines/crosshairs added to mark spot centers
        self.spot_info_dict (dict):           key (str) [alphanumeric spot coordinates] =>
                                              (unadjusted_signal, background_adjusted_signal, ellipsoid_index,
                                              spot_midpoint, top_left_corner)

    Methods:
        __init__:            initialization function that imports the tiff image and runs analyze_array()
        analyze_array:       main function that quantifies the array
        grid_peak_finder:    function to mark spots based on known grid dimensions
        check_line_peaks:    subsidiary function of grid_peak_finder that is used to check whether auto-detected peaks
                             are correct or not
        handle_mismatch:     subsidiary function of grid_peak_finder that resolves mismatches between the detected peak
                             count and the expected peak count
        infer_peaks:         subsidiary function of grid_peak_finder that infers line peak indices based on grid
                             dimensions when searching for peaks is not possible/successful
        image_slicer:        function for slicing a grayscale spot array image into image snippets representing each spot
        mins_between_peaks:  subsidiary function of image_slicer that finds minima on either side of defined peaks
        array_between:       subsidiary function of image_slicer that finds values in a numpy array that are between a
                             minimum and maximum value
        ellipsoid_constrain: function that returns a dictionary of spot coordinates --> tuple of quantified results
        draw_crosshairs:     function to draw crosshairs on the true peak points for each spot in the unsliced image
    '''

    def __init__(self, tiff_path, spot_dimensions, metadata = (None, None, None), show_sliced_image = False,
                 show_outlined_image = False, suppress_warnings = False, pixel_log_base = 1,
                 ending_coord = None, arbitrary_coords_to_drop = None, buffer_width = 0, verbose = False):
        '''
        Initialization function invoked when a new instance of the SpotArray class is created

        Args:
            tiff_path (str): the path to the tiff file where a spot image is stored
            spot_dimensions (tuple): a tuple of (number of spots in width, number of spots in height), i.e. (x,y)
            metadata (tuple): a tuple of (probe_name, copy_number, scan_number)
            verbose (Boolean): whether to report initialization progress/steps in the terminal (default: False)
            suppress_warnings (Boolean): whether to suppress warnings that occur during initialization (default: False)
            pixel_log_base (int or float): if a logarithmic pixel encoding was used, this should be the logarithm's base

        Returns:
            None

        Raises:
            Warning: multiple layers found; uses the first layer when reading multi-layer TIFFs. Can be caused by an alpha-layer being present.
        '''
        self.grid_shape = spot_dimensions
        self.filename = os.path.basename(tiff_path).rsplit(".", 1)[0] # gets the filename without the extension
        self.probe_name, self.copy_number, self.scan_number = metadata

        # Initialize main grayscale image stored in tiff_path
        img = imread(tiff_path)
        try:
            layers = img.shape[2]
        except:
            layers = 1
        if layers > 1:
            img = img[:,:,0]
            print(f"\t\tCaution: {layers} layers were found when importing " + self.filename + ", but 1 was expected; the first layer was used.") if not suppress_warnings else None

        self.grayscale_array = img
        self.image_shape = img.shape[1], img.shape[0] #image_width, image_height
        self.linear_array = reverse_log_transform(img, base = pixel_log_base) # Ensures that pixel values linearly correlate with luminance

        # Analyze the array automatically with the default variables
        self.analyze_array(ellipsoid_dilation_factor = 1, show_sliced_image = show_sliced_image,
                           show_outlined_image = show_outlined_image, show_individual_spot_images = False,
                           center_spots_mode = "iterative", ending_coord = ending_coord,
                           arbitrary_coords_to_drop = arbitrary_coords_to_drop, buffer_width = buffer_width,
                           verbose = verbose)

    def analyze_array(self, ellipsoid_dilation_factor = 1, show_sliced_image = False, show_outlined_image = False,
                      show_individual_spot_images = False, center_spots_mode = "iterative",
                      ending_coord = None, arbitrary_coords_to_drop = None, buffer_width = 0, verbose = False):
        '''
        Main function to analyze and quantify grids of spots

        Args:
            ellipsoid_dilation_factor (float):  multiplier that dilates identified spot borders
            show_sliced_image (bool):           whether to display the source image with gridlines after spot borders are found
            show_outlined_image (bool):         whether to display the image with detected spots circled in green
            show_individual_spot_images (bool): whether to consecutively display individual spots for debugging
            center_spots_mode (str):            the mode for centering spots within their respective sliced squares
            ending_coord (str):                 the last coord after which coords should be dropped;
                                                e.g. if set to D5, all coords from D6 onwards will be deleted
            arbitrary_coords_to_drop (list):    list of coords to drop if desired
            buffer_width (int):                 a positive integer used to dilate circles, when mode is iterative, before background adjustment
            verbose (Boolean):                  whether to display progress information for debugging

        Returns:
            analyzed_array_tuple (tuple):       (copy_number, scan_number, probe_name, linear_array, spot_info_dict,
                                                outlined_image)
        '''
        if verbose:
            print("\tProcessing: Copy", self.copy_number, " - Scan", self.scan_number, "- Probe", self.probe_name)
            print(f"\t\tfinding grid peaks in image of shape {self.linear_array.shape}...")

        # Find the indices of the vertical and horizontal maxima and minima
        self.vlpeaks_indices, self.vlmins_indices, self.hlinepeaks_indices, self.hlinemins_indices = self.grid_peak_finder(show_line_sums = False, verbose = verbose)

        # Slice the image based on vertical and horizontal maxima and minima
        print("\t\tslicing image...") if verbose else None
        image_peak_coordinates, image_slices, self.sliced_image = self.image_slicer(image_ndarray = self.linear_array, vlinepeaks_indices = self.vlpeaks_indices, vlinemins_indices = self.vlmins_indices,
                                                                                    hlinepeaks_indices = self.hlinepeaks_indices, hlinemins_indices = self.hlinemins_indices,
                                                                                    render_sliced_image = True, show_individual_spot_stats = False,
                                                                                    slicer_debugging = show_individual_spot_images, verbose = verbose)

        # Display popup of sliced image if prompted
        if show_sliced_image:
            imshow(self.sliced_image / self.sliced_image.max())
            import matplotlib.pyplot as plt
            plt.show()

        # Make a dictionary holding alphanumeric spot coordinates as keys --> tuples of
        # (unadjusted_signal, background_adjusted_signal, ellipsoid_index, peak_intersect, top_left_corner)
        print("\t\tcomputing background-adjusted signal and ellipsoid_index...") if verbose else None
        self.outlined_image, self.spot_info_dict = self.ellipsoid_constrain(spot_images = image_slices, dilation_factor = ellipsoid_dilation_factor,
                                                                            centering_mode = center_spots_mode, buffer_width = buffer_width, verbose = verbose)

        # Remove specified coords
        self.spot_info_dict = self.drop_specified_coords(self.spot_info_dict, ending_coord = ending_coord,
                                                         arbitrary_coords_to_drop = arbitrary_coords_to_drop)

        # Display popup of sliced image with drawn crosshairs if prompted
        if show_outlined_image:
            imshow(self.outlined_image / self.outlined_image.max())
            import matplotlib.pyplot as plt
            plt.show()

        # In addition to assigning to internal variables, also returns a tuple of results that can be optionally assigned when this function is invoked
        analyzed_array_tuple = (self.copy_number, self.scan_number, self.probe_name, self.linear_array, self.spot_info_dict, self.outlined_image)
        return analyzed_array_tuple

    '''
    ------------------------------------------------------------------------------------------------------------
    The following are backend functions used by analyze_array(); they generally should not be used on their own.
    ------------------------------------------------------------------------------------------------------------
    '''

    def drop_specified_coords(self, spot_info_dict, ending_coord = None, arbitrary_coords_to_drop = None):
        '''
        Simple function to delete entries for spot coordinates that are blank or otherwise specified to be excluded

        Please note that this function does not currently support coords with more than one letter.

        Args:
            spot_info_dict (dict):           the dictionary of spots
            ending_coord (str):              if given, it is the last coord after which subsequent coords will be deleted
            arbitrary_coords_to_drop (list): if given, it is the list of alphanumeric spot coordinates to remove

        Returns:
            updated_spot_dict (dict):        a new dictionary of spots excluding those to be dropped
        '''
        updated_spot_dict = spot_info_dict.copy()

        # Remove entries after the ending coordinate
        if ending_coord is not None:
            ending_letter = ending_coord[0]
            ending_letter_idx = string.ascii_uppercase.index(ending_letter)
            ending_number = int(ending_coord[1:])

            for key in spot_info_dict.keys():
                key_letter = key[0]
                key_letter_idx = string.ascii_uppercase.index(key_letter)
                key_number = int(key[1:])

                if key_letter_idx > ending_letter_idx:
                    updated_spot_dict.pop(key)
                elif key_letter_idx == ending_letter_idx and key_number > ending_number:
                    updated_spot_dict.pop(key)

        # Remove entries that are in the arbitrary list of coordinates to drop
        if arbitrary_coords_to_drop is not None:
            for key in updated_spot_dict.keys():
                if key in arbitrary_coords_to_drop:
                    updated_spot_dict.pop(key)

        return updated_spot_dict


    def grid_peak_finder(self, show_line_sums = False, verbose = False):
        '''
        Function to define the coordinates of the spot array grid.

        This function uses the handle_mismatch() and infer_peaks() methods in this class.
            - First creates lists of pixel value sums of vertical and horizontal lines of pixels in self.linear_array.
            - Then uses scipy.signal.find_peaks() to find peaks and valleys in these lists.

        Args:
            show_line_sums (Boolean): whether to show a plot of the vertical and horizontal line sums
            verbose (Boolean): whether to output additional information for debugging

        Returns:
            vertical_line_peaks (np.ndarray):   numpy list of horizontal indices where peaks exist in summed vertical lines
            vertical_line_mins (np.ndarray):    numpy list of horizontal indices where minima exist in summed vertical lines
            horizontal_line_peaks (np.ndarray): numpy list of vertical indices where peaks exist in summed horizontal lines
            horizontal_line_mins (np.ndarray):  numpy list of vertical indices where minima exist in summed horizontal lines
        '''
        grid_width, grid_height = self.grid_shape
        image_width, image_height = self.image_shape

        # Find the sums of vertical and horizontal lines of pixels in the grayscale image array
        vlsums, hlsums = self.linear_array.sum(axis=0), self.linear_array.sum(axis=1)

        if show_line_sums:
            import matplotlib.pyplot as plt
            print("\t\t\tShowing vertical line sums...")
            plt.plot(vlsums)
            plt.show()
            print("\t\t\tShowing horizontal line sums...")
            plt.plot(hlsums)
            plt.show()

        # Find peaks and valleys in the vertical and horizontal line sums
        vlpeaks, _ = find_peaks(vlsums)
        vlmins, _ = find_peaks(vlsums * -1)
        hlpeaks, _ = find_peaks(hlsums)
        hlmins, _ = find_peaks(hlsums * -1)

        # Handle vertical line peaks (horizontal axis) and horizontal line peaks (vertical axis); use default variance
        print("\t\t\tProcessing horizontal axis (vertical line peaks/mins)...") if verbose else None
        vlpeaks, vlmins = self.check_line_peaks(line_axis_name = "vertical", line_sums = vlsums, line_peaks = vlpeaks, line_mins = vlmins,
                                                expected_peaks_count = grid_width, length_px = image_width, verbose = verbose)
        print("\t\t\t---", "\n\t\t\tProcessing vertical axis (horizontal line peaks/mins)...") if verbose else None
        hlpeaks, hlmins = self.check_line_peaks(line_axis_name = "horizontal", line_sums = hlsums, line_peaks = hlpeaks, line_mins = hlmins,
                                                expected_peaks_count = grid_height, length_px = image_height, verbose = verbose)
        print("\t\t\t---") if verbose else None

        return vlpeaks, vlmins, hlpeaks, hlmins

    def check_line_peaks(self, line_axis_name, line_sums, line_peaks, line_mins, expected_peaks_count, length_px, allowed_variance = 0.2, verbose = False):
        '''
        Function to check whether the detected line peaks match the number of expected spots and correct as necessary

        Args:
            line_axis_name (str): name of the axis of the line sums, i.e. "vertical" or "horizontal"
            line_sums (arr of floats): list of sums of lines of pixels along the orthogonal axis
            line_peaks (arr of ints): list of indices in line_sums where local peaks exist
            line_mins (arr of ints): list of indices in line_sums where local minima exist
            expected_peaks_count (int): expected number of peaks
            length_px (int): number of pixels in the image in the orthogonal axis to lines of pixels being summed
            allowed_variance (float): multiplier that defines the allowed deviation from expected spacing between peaks
            verbose (bool): whether to output additional information for debugging

        Returns:
            line_peaks (arr of ints): adjusted input line_peaks depending on whether a mismatch was found
            line_mins (arr of ints): adjusted input line_mins depending on whether a mismatch was found
        '''
        # Mismatch handling
        if len(line_peaks) != expected_peaks_count:
            '''
            Mismatch handling for when the number of vertical/horizontal line peaks in the horizontal/vertical axis does 
            not equal the number of spots expected in the horizontal axis
            '''
            print(f"\t\t\tPeak Count Warning: number of {line_axis_name} line peaks does not match expected peaks count (found {len(line_peaks)}, expected {expected_peaks_count}); invoking self.handle_mismatch()") if verbose else None
            line_peaks, line_mins = self.handle_mismatch(line_sums = line_sums, actual_peaks = line_peaks, actual_mins = line_mins,
                                                         expected_peaks_count = expected_peaks_count, line_axis_name = line_axis_name,
                                                         length_px = length_px, tolerance_spot_frac = 0.25, verbose = verbose)
        else:
            print(f"\t\t\tSuccess: found correct number of {line_axis_name} line peaks") if verbose else None

        # Handle case where only 1 or 2 rows/columns exist
        if len(line_peaks) == 1:
            print(f"\t\t\t\tNotice: only 1 {line_axis_name} line peak exists; setting {line_axis_name} mins flanking either side") if verbose else None
            line_mins = [0, len(line_sums)-1]
            return line_peaks, line_mins
        elif len(line_peaks) == 2 and len(line_mins) == 1:
            print(f"\t\t\t\tNotice: only 2 {line_axis_name} line peaks and 1 min between them; adding flanking mins on either side") if verbose else None
            if line_peaks[0] < line_mins[0] < line_peaks[1]:
                line_mins = [0, line_mins[0], len(line_sums)-1]
            else:
                line_min_between = round((line_peaks[0] + line_peaks[1]) / 2)
                line_mins = [0, line_min_between, len(line_sums)-1]
            return line_peaks, line_mins

        # Find distances between adjacent indices of vertical and horizontal line peaks and mins
        line_peaks_deltas = line_peaks[1:] - line_peaks[:-1]
        line_mins_deltas = line_mins[1:] - line_mins[:-1]

        # Begin testing line peak and min spacing
        print("\t\t\tTesting line peak and min spacing...") if verbose else None

        peak_min_delta, peak_max_delta = line_peaks_deltas.min(), line_peaks_deltas.max()
        valley_min_delta, valley_max_delta = line_mins_deltas.min(), line_mins_deltas.max()

        allowed_min_delta, allowed_max_delta = (((1 - allowed_variance) * len(line_sums) / expected_peaks_count), ((1 + allowed_variance) * len(line_sums) / expected_peaks_count))

        if peak_min_delta < allowed_min_delta:
            # If the min spacing between line peaks is less than the tolerated min, reverts to inferring peaks
            print(f"\t\t\t\tSpacing Warning: irregular line peak spacing; smallest space was {peak_min_delta}, but the allowed minimum is {allowed_min_delta}; defaulting to inferring peaks from grid dimensions") if verbose else None
            line_peaks, line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        elif peak_max_delta > allowed_max_delta:
            # If the max spacing between line peaks is greater than the tolerated max, reverts to inferring peaks
            print(f"\t\t\t\tSpacing Warning: irregular line peak spacing; largest space was {peak_max_delta}, but the allowed maximum is {allowed_max_delta}; defaulting to inferring peaks from grid dimensions") if verbose else None
            line_peaks, line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        elif valley_min_delta < allowed_min_delta:
            # If the min spacing between line minima is less than the tolerated min, reverts to inferring peaks
            print(f"\t\t\t\tSpacing Warning: irregular line valley spacing; smallest space was {valley_min_delta}, but the allowed minimum is {allowed_min_delta}; defaulting to inferring peaks from grid dimensions") if verbose else None
            line_peaks, line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        elif valley_max_delta > allowed_max_delta:
            # If the max spacing between line minima is greater than the tolerated max, reverts to inferring peaks
            print(f"\t\t\t\tSpacing Warning: irregular line valley spacing largest space was {valley_max_delta}, but the allowed maximum is {allowed_max_delta}; defaulting to inferring peaks from grid dimensions") if verbose else None
            line_peaks, line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        else:
            print(f"\t\t\t\tPassed: line spacing is within tolerances") if verbose else None

        return line_peaks, line_mins

    def handle_mismatch(self, line_sums, actual_peaks, actual_mins, expected_peaks_count, line_axis_name, length_px,
                        tolerance_spot_frac = 0.25, extra_peaks_proportion = 0.1, deltas_threshold = 1.5, verbose = False):
        '''
        Function to resolve mismatches between the detected peak count and the expected peak count.

        Args:
            line_sums (arr of floats): list of sums of lines of pixels along the orthogonal axis
            actual_peaks (arr of ints): the detected peaks (which are given as indices referring to the line_sums)
            actual_mins (arr of ints): the detected mins (indices referring to line_sums)
            expected_peaks_count (int): the number of peaks that are expected based on the grid dimensions (number of spots expected)
            line_axis_name (str): the line sum axis; must be "vertical" or "horizontal"
            tolerance_spot_frac (float): the fraction of the spot dimension (in pixels) that is the allowed distance between peaks for them to be declared mergeable
            extra_peaks_proportion (float): the fraction of the expected peak count that is allowed for the collapse_extra_peaks method to be used
            deltas_threshold (float or None): if float, it is used to test the distances between output line minima to ensure no aberrant differences
                                              default is 1.5, allowing 50% variance from mean distances between minima

        Returns:
            output_line_peaks (arr of ints): new array of line peaks based on conditionally applying infer_peaks()
            output_line_mins (arr of ints): new array of line mins based on conditionally applying infer_peaks()
        '''
        actual_peaks_count = len(actual_peaks)
        extra_peaks_ceiling = (extra_peaks_proportion + 1) * expected_peaks_count
        extra_peaks_ceiling = round(extra_peaks_ceiling)

        if actual_peaks_count < expected_peaks_count or actual_peaks_count > extra_peaks_ceiling:
            print("\t\t\t\tinferring", line_axis_name, "line peaks from dimensions...") if verbose else None
            output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
        elif actual_peaks_count > expected_peaks_count and actual_peaks_count <= extra_peaks_ceiling:
            print("\t\t\t\taveraging extra", line_axis_name, "line peaks that are within", tolerance_spot_frac * 100, "% of average spot dimension...") if verbose else None

            output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, collapse_extra_peaks = True,
                                                                   detected_peaks=actual_peaks, tolerance_spot_frac = tolerance_spot_frac, verbose = verbose)
            print("\t\t\t\tgot", len(output_line_peaks), "line peaks and", len(output_line_mins), "line mins") if verbose else None

            if deltas_threshold is not None:
                line_mins_deltas = output_line_mins[1:] - output_line_mins[:-1]
                deltas_mean_expected = (length_px / expected_peaks_count) * deltas_threshold
                excessive_variance = any(line_mins_deltas > (deltas_threshold * deltas_mean_expected)) #boolean value
                print("\t\t\t\texcessive variances between line mins were detected...") if excessive_variance else None
            else:
                excessive_variance = False

            if len(output_line_peaks) != expected_peaks_count:
                print("\t\t\t\tfailed to correct number of peaks by averaging within the tolerance: wrong number of peaks found",
                      "\n\t\t\t\treverting to inferring peaks by grid dimensions...") if verbose else None
                output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)
            elif excessive_variance:
                print("\t\t\t\tfailed to correct number of peaks by averaging within the tolerance: excessive variance was detected",
                      "\n\t\t\t\treverting to inferring peaks by grid dimensions...") if verbose else None
                output_line_peaks, output_line_mins = self.infer_peaks(line_sums = line_sums, expected_peaks = expected_peaks_count, verbose = verbose)

        else:
            output_line_peaks, output_line_mins = actual_peaks, actual_mins

        return output_line_peaks, output_line_mins

    def infer_peaks(self, line_sums, expected_peaks, collapse_extra_peaks = False, detected_peaks = None, tolerance_spot_frac = 0.25, verbose = False):
        '''
        Infers line peak indices based on grid dimensions (length or width) when searching for peaks is not possible.

        For infer_peaks() to generate valid results, input images must be cropped right to the border of the spots on
        all sides, with no extra black space.

        Args:
            line_sums (np.ndarray):         pixel value sums of vertical or horizontal lines being assessed
            expected_peaks (int):           number of expected peaks based on grid dimensions
            collapse_extra_peaks (bool):    whether to collapse extra peaks if too many peaks have been found
            detected_peaks (np.ndarray):    array of detected peaks to collapse if collapse_extra_peaks is True
            tolerance_spot_frac (float):    tolerance multiplier when assessing distance between peaks
            verbose (bool):                 whether to display debugging information

        Returns:
            line_peaks (np.ndarray):        array of line peak indices
            line_mins (np.ndarray):         array of line min indices
        '''
        mean_spot_dimension = len(line_sums) / expected_peaks

        inferred_line_peaks = np.arange(expected_peaks) * mean_spot_dimension
        inferred_line_peaks = inferred_line_peaks + (mean_spot_dimension / 2)  # starts halfway across the first inferred spot square, making the assumption that the peak is in the middle
        inferred_line_peaks = inferred_line_peaks.round().astype(int)  # rounds and gives integers, as indices must be ints

        inferred_line_mins = np.arange(expected_peaks + 1) * mean_spot_dimension
        inferred_line_mins = inferred_line_mins.round().astype(int)
        if inferred_line_mins[-1] > (len(line_sums) - 1):
            inferred_line_mins[-1] = len(line_sums) - 1  # catches error where the ending number, rounded up, might otherwise go out of bounds

        if collapse_extra_peaks:
            print("\t\t\t\tcollapsing extra peaks (detected " + str(len(detected_peaks)) + ")...") if verbose else None
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

    def image_slicer(self, image_ndarray, vlinepeaks_indices, vlinemins_indices, hlinepeaks_indices, hlinemins_indices,
                     render_sliced_image = True, show_individual_spot_stats = False, slicer_debugging = False, verbose = False):
        '''
        Function for slicing a grayscale spot array image into image snippets representing each spot.

        Args:
            image_ndarray (np.ndarray):      2D array representing the grayscale image to slice
            vlinepeaks_indices (np.ndarray): array of indexes of vertical line peaks along the horizontal axis
            vlinemins_indices (np.ndarray):  array of indexes of vertical line mins along the horizontal axis
            hlinepeaks_indices (np.ndarray): array of indexes of horizontal line peaks along the vertical axis
            hlinemins_indices (np.ndarray):  array of indexes of horizontal line mins along the vertical axis
            render_sliced_image (bool):      whether to render the sliced image as a colour image showing slice lines
            slicer_debugging (bool):         whether to show each image snippet sequentially for troubleshooting
            verbose (bool):                  whether to display text-based debugging information

        Returns:
            peak_coordinates_dict (dict):    dictionary of alphanumeric_coords --> peak_coordinates
            sliced_spot_dict (dict):         dictionary of alphanumeric_coords --> values_dict
                                                 values_dict (dict):  "top_left_corner", "spot_midpoint_coords",
                                                                      "spot_radius_min", "spot_radius_max",
                                                                      "spot_image_snippet"
            color_image (np.ndarray):        rendered sliced image showing detected gridlines;
                                             returned only if render_sliced_image is True
        '''
        print("\t\t\tstarting image_slicer()...") if verbose else None

        # Show the input image if slicer debugging is enabled
        if slicer_debugging:
            print("\t\t\tshowing input image...")
            import matplotlib.pyplot as plt
            imshow(image_ndarray, cmap="gray")
            plt.show()

        if render_sliced_image:
            max_pixel = image_ndarray.max()
            color_image = np.repeat(image_ndarray[:,:,np.newaxis], 3, axis=2) #Red=[:,:,0], Green=[:,:,1], Blue=[:,:,2]

        alphabet = list(string.ascii_uppercase)  # Used for declaring coordinates later

        print("\t\t\tfinding minima between line peaks (horizontal and vertical lines)...") if verbose else None
        vlpeaks_prev_mins, vlpeaks_next_mins = self.mins_between_peaks(peaks_array = vlinepeaks_indices, mins_array = vlinemins_indices, max_index = image_ndarray.shape[1])
        hlpeaks_prev_mins, hlpeaks_next_mins = self.mins_between_peaks(peaks_array = hlinepeaks_indices, mins_array = hlinemins_indices, max_index = image_ndarray.shape[0])

        print("\t\t\tfound", len(vlpeaks_prev_mins), "minima to the left and", len(vlpeaks_next_mins), "to the right of vlpeaks",
              "\n\t\t\tfound", len(hlpeaks_prev_mins), "minima to the left and", len(hlpeaks_next_mins), "to the right of hlpeaks") if verbose else None

        peak_coordinates_dict = {}
        sliced_spot_dict = {}
        for i, horizontal_peak in enumerate(hlinepeaks_indices):
            row_letter = alphabet[i]
            for j, vertical_peak in enumerate(vlinepeaks_indices):
                col_number = j + 1
                alphanumeric_coordinates = row_letter + str(col_number)
                print(f"\t\t\t\tprocessing {alphanumeric_coordinates} where hlinepeak = {horizontal_peak} and vlinepeak = {vertical_peak}") if show_individual_spot_stats else None

                horizontal_prev_min = int(hlpeaks_prev_mins.get(horizontal_peak))  # horizontal peaks are along the vertical axis
                horizontal_next_min = int(hlpeaks_next_mins.get(horizontal_peak))
                vertical_prev_min = int(vlpeaks_prev_mins.get(vertical_peak))  # vertical peaks are along the horizontal axis
                vertical_next_min = int(vlpeaks_next_mins.get(vertical_peak))

                peak_coordinates = (horizontal_peak, vertical_peak)  # (height, width)
                peak_coordinates_dict[alphanumeric_coordinates] = peak_coordinates
                print(f"\t\t\t\t\tpeak_coordinates = {peak_coordinates}") if show_individual_spot_stats else None

                print(f"\t\t\t\t\tgetting sliced spot at range [{horizontal_prev_min}:{horizontal_next_min}, {vertical_prev_min}:{vertical_next_min}]") if show_individual_spot_stats else None
                sliced_spot = image_ndarray[horizontal_prev_min:horizontal_next_min,
                              vertical_prev_min:vertical_next_min]  # height range, width range

                # Define the coordinates of the top left corner, midpoint, and radius of a given image snippet in the source image (self.linear_array)
                top_left_corner = (horizontal_prev_min, vertical_prev_min)  # height x width (y,x)
                sliced_spot_midpoint = (round(sliced_spot.shape[0] / 2), round(sliced_spot.shape[1] / 2))
                spot_radius_min = min(sliced_spot_midpoint)
                spot_radius_max = max(sliced_spot_midpoint)

                # Derive the coordinates of the spot midpoint in the source image
                spot_midpoint_coords = (top_left_corner[0] + sliced_spot_midpoint[0], top_left_corner[1] + sliced_spot_midpoint[1])

                # Package the results into a dictionary
                values_at_coord = {
                    "top_left_corner": top_left_corner,
                    "spot_midpoint_coords": spot_midpoint_coords,
                    "spot_radius_min": spot_radius_min,
                    "spot_radius_max": spot_radius_max,
                    "spot_image_snippet": sliced_spot
                }

                # Assign the values dict to a dictionary with the alphanumeric coordinates as the key
                sliced_spot_dict[alphanumeric_coordinates] = values_at_coord

                if slicer_debugging:
                    print(f"\t\t\t{alphanumeric_coordinates} info: {values_at_coord}")
                    import matplotlib.pyplot as plt
                    imshow(sliced_spot, cmap="gray")
                    plt.show()

        if render_sliced_image:
            # Mark peaks with blue lines
            for horizontal_peak in hlinepeaks_indices:
                color_image[:,:,0][horizontal_peak, :], color_image[:,:,1][horizontal_peak, :], color_image[:,:,2][horizontal_peak, :] = (0, 0, max_pixel)
            for vertical_peak in vlinepeaks_indices:
                color_image[:,:,0][:, vertical_peak], color_image[:,:,1][:, vertical_peak], color_image[:,:,2][:, vertical_peak] = (0, 0, max_pixel)

            # Mark mins (borders) with red lines
            for horizontal_min in hlinemins_indices:
                color_image[:,:,0][horizontal_min, :], color_image[:,:,1][horizontal_min, :], color_image[:,:,2][horizontal_min, :] = (max_pixel, 0, 0)
            for vertical_min in vlinemins_indices:
                color_image[:,:,0][:, vertical_min], color_image[:,:,1][:, vertical_min], color_image[:,:,2][:, vertical_min] = (max_pixel, 0, 0)

        if render_sliced_image:
            return peak_coordinates_dict, sliced_spot_dict, color_image
        else:
            return peak_coordinates_dict, sliced_spot_dict

    def mins_between_peaks(self, peaks_array, mins_array, max_index):
        '''
        Function for finding minima on either side of peaks defined by scipy.signal.find_peaks().

        Args:
            peaks_array (np.ndarray):  array of indices of peaks
            mins_array (np.ndarray):   array of indices of valleys/mins surrounding the peaks

        Returns:
            previous_mins_dict (dict): dictionary of peak indices --> min indices immediately before
            next_mins_dict (dict):     dictionary of peak indices --> min indices immediately after
        '''

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

        # Handle case where the leftmost min (before the first peak) is negative, such as when the leftmost peak is closer to the left than the mean inter-peak space
        far_left_min = round(peaks_array[0] - (inter_peak_space / 2))
        if far_left_min < 0:
            far_left_min = 0
        previous_mins_dict[peaks_array[0]] = far_left_min

        # Handle case where the rightmost min (after the last peak) exceeds the maximum allowed index
        far_right_min = round(peaks_array[-1] + (inter_peak_space / 2))
        if far_right_min > max_index:
            far_right_min = max_index
        next_mins_dict[peaks_array[-1]] = far_right_min

        return previous_mins_dict, next_mins_dict

    def array_between(self, numpy_array, min_threshold, max_threshold):
        '''
        Finds values in a numpy array that are between a minimum and maximum value.

        Args:
            numpy_array (np.ndarray): array of values
            min_threshold (int or float): bottom threshold
            max_threshold (int or float): top threshold

        Returns:
            in_range_values (np.ndarray): array of values that are between the threshold values
        '''
        boolean_array = np.logical_and(numpy_array > min_threshold, numpy_array < max_threshold)
        in_range_indices = np.where(boolean_array)[0]
        in_range_values = np.empty((0))
        for i in in_range_indices:
            in_range_values = np.append(in_range_values, numpy_array[i])
        return in_range_values

    #-------------------------------------------------------------------------------------------------------------------------------------------------------

    def ellipsoid_constrain(self, spot_images, dilation_factor = 1, centering_mode = "iterative",
                            spot_radius_mode = "inferred", return_coordinates_list = False, buffer_width = 0,
                            verbose = False):
        '''
        Function that returns a dictionary of spot coordinates where the value is a tuple of:
            unadjusted_signal (float): sum of pixel values inside the ellipse defining the spot
            background_adjusted_signal (float): sum of pixel values inside the spot ellipse, minus area-adjusted signal from outside the ellipse
            ellipsoid_index (float): the ratio of mean pixel values inside the spot ellipse to mean pixel values outside the ellipse
            spot_midpoint (tuple): a tuple representing coordinates of the center of the defined spot ellipse
            top_left_corner (tuple): a tuple representing coordinates of the top left corner of each spot image snippet
            stitched_image (np.ndarray): if hough_stitch mode is used, is an image showing the outlined detected spots

        The ellipsoid centering mode can be any of the following options:
            "hough": Hough circle transform method
            "hough_stitch": Hough circle transform, also returning a stitched image showing the defined circles
            "blob": uses self.center_blob_spot()
            "line_peaks": uses self.center_peak_spot()
            None: defaults to defining an ellipsoid based on the height and width of each spot image snippet

        Args:
            spot_images (dict): a dictionary of spot coordinates where the value is a tuple of (top_left_corner, spot_image)
            dilation_factor (float): a multiplier to enlarge or constrict the defined constraining ellipsoid
            centering_mode (str): mode for how to center and constrain the spot ellipsoid
            return_coordinates_list (bool): whether to reutrn a list of coordinates for the spots, in addition to the results dictionary
            buffer_width (int): a positive integer used to dilate circles before declaring outside pixels for use in local background adjustment
            verbose (bool): whether to display debugging information

        Returns:
            output_dict (dict): a dictionary of spot coordinates where the value is a tuple of (unadjusted_signal, background_adjusted_signal, ellipsoid_index, spot_midpoint, top_left_corner, stitched_image)
        '''

        print("\t\t\trunning ellipsoid_constrain()...") if verbose else None

        if spot_radius_mode == "inferred":
            # Get the minimum radius expected for spots in a grid where spacing is equal
            mean_spot_height = self.linear_array.shape[0] / self.grid_shape[1] # height in pixels / height in spots
            mean_spot_width = self.linear_array.shape[1] / self.grid_shape[0] # width in pixels / width in spots
            spot_radius_min = int(min([mean_spot_height, mean_spot_width]) / 2)
        elif spot_radius_mode == "snippets":
            # Get the mean radius of all spots, then take the minimum value and enforce it as the radius for all spots
            spot_radii_list = []
            for values_dict in spot_images.values():
                spot_image_radius = (values_dict.get("spot_radius_min") + values_dict.get("spot_radius_max")) / 2
                spot_image_radius = round(spot_image_radius)
                spot_radii_list.append(spot_image_radius)
            spot_radius_min = min(spot_radii_list)
        else:
            raise ValueError(f"ellipsoid_constrain error: spot_radius_mode was set to {spot_radius_mode}, but one of [\"inferred\", \"snippets\"] was expected.")

        # Perform spot image quantification
        output_dict = {}
        coordinates_list = []
        image_stitching_list = []
        final_midpoints_list = []

        for spot_coordinates, values_dict in spot_images.items():
            top_left_corner = values_dict.get("top_left_corner")
            spot_midpoint_coords = values_dict.get("spot_midpoint_coords")
            spot_image = values_dict.get("spot_image_snippet")
            spot_image_height, spot_image_width = spot_image.shape

            if centering_mode == "iterative":
                # Import the iterative scanning algorithm as a function
                from Matrix_Generator.image_processing.scan_optimal_circle import spot_circle_scan

                # Find the optimal circle for the given spot
                final_midpoint_coords, results = spot_circle_scan(image_snippet = spot_image, source_image = self.linear_array,
                                                                  midpoint_coords = spot_midpoint_coords, enforced_radius = spot_radius_min,
                                                                  alphanumeric_coords = spot_coordinates, radius_variance_multiplier = 0.33,
                                                                  radius_shrink_multiplier = 0.9, value_to_maximize = "ellipsoid_index", verbose = False)

                # Append midpoint coordinates to the list for drawing circles later
                final_midpoints_list.append(final_midpoint_coords)

                # Declare quantified metrics
                ellipsoid_index = results.get("ellipsoid_index")
                spot_midpoint = results.get("spot_midpoint")
                mean_intensity_outside = results.get("outside_sum") / results.get("outside_count")
                background_adjusted_signal = results.get("inside_sum") - (results.get("inside_count") * mean_intensity_outside)
                unadjusted_signal = results.get("inside_sum")

            elif centering_mode == "hough" or centering_mode == "hough_stitch":
                from Matrix_Generator.image_processing.hough_circle_detector import detect_circle as hough_detect_circle
                results = hough_detect_circle(spot_image, dilate_to_edge = True, verbose = True)
                ellipsoid_index = results.get("ellipsoid_index")
                spot_midpoint = results.get("spot_midpoint")
                mean_intensity_outside = results.get("outside_sum") / results.get("outside_count")
                background_adjusted_signal = results.get("inside_sum") - (results.get("inside_count") * mean_intensity_outside)
                unadjusted_signal = results.get("inside_sum")
                if centering_mode == "hough_stitch":
                    image_for_stitching = results.get("outlined_image")
                    image_stitching_list.append((image_for_stitching, top_left_corner))

            else:
                print("\t\t\tcaution: centering_mode", centering_mode, "is not recognized; defaulting to None") if centering_mode != None else None

                # Import the general circle quantification function
                from Matrix_Generator.image_processing.image_utils import circle_stats

                # Compute necessary input variables
                spot_radii = np.array([round((spot_image_height / 2) * dilation_factor), round((spot_image_width / 2) * dilation_factor)])
                spot_midpoint = (spot_radii[0], spot_radii[1])
                spot_radius = spot_radii.min() * dilation_factor  # enforce circles when ellipsoids are oblong

                # Quantify the defined circle
                pixels_inside, pixels_outside, unadjusted_signal, sum_outside, ellipsoid_index, background_adjusted_signal = circle_stats(grayscale_image,
                                                                                                                                          center = spot_midpoint,
                                                                                                                                          radius = spot_radius,
                                                                                                                                          buffer_width = buffer_width)

            # To the output dict, add a tuple containing the background-adjusted signal and the ellipsoid index
            output_dict[spot_coordinates] = (unadjusted_signal, background_adjusted_signal, ellipsoid_index, spot_midpoint, top_left_corner)
            coordinates_list.append(spot_coordinates)

        # If a method was used that supports circling the detected spots, generate an image showing circles in green
        if centering_mode == "iterative":
            from Matrix_Generator.image_processing.image_utils import draw_color_circle
            outlined_image = draw_color_circle(self.sliced_image, final_midpoints_list, spot_radius_min, "green")
        elif centering_mode == "hough_stitch":
            from Matrix_Generator.image_processing.image_utils import concatenate_images
            outlined_image = concatenate_images(image_stitching_list)
        else:
            outlined_image = None

        '''
        Returns a dictionary where the key is spot coordinates and the value is a tuple containing 
        (background-adjusted signal, ellipsoid index)
        '''
        if return_coordinates_list:
            return outlined_image, output_dict, coordinates_list
        else:
            return outlined_image, output_dict

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
    def draw_crosshairs(self, color_image, spot_info, crosshair_diameter = 5, crosshair_width = 3):
        max_green_pixel = color_image[:,:,1].max()

        if crosshair_width % 2 == 0:
            crosshair_width = crosshair_width + 1  # catches even widths

        deviation = int((crosshair_width - 1) / 2)

        for spot_coordinates, value_tuple in spot_info.items():
            unadjusted_signal, background_adjusted_signal, ellipsoid_index, peak_intersect, top_left_corner = value_tuple
            real_peak_intersect = (top_left_corner[0] + peak_intersect[0], top_left_corner[1] + peak_intersect[1])

            # Draw horizontal green crosshair
            color_image[:,:,1][real_peak_intersect[0] - deviation: real_peak_intersect[0] + deviation, real_peak_intersect[1] - crosshair_diameter: real_peak_intersect[1] + crosshair_diameter] = max_green_pixel
            color_image[:,:,[0,2]][real_peak_intersect[0] - deviation: real_peak_intersect[0] + deviation, real_peak_intersect[1] - crosshair_diameter: real_peak_intersect[1] + crosshair_diameter] = 0

            # Draw vertical green crosshair
            color_image[:,:,1][real_peak_intersect[0] - crosshair_diameter: real_peak_intersect[0] + crosshair_diameter, real_peak_intersect[1] - deviation: real_peak_intersect[1] + deviation] = max_green_pixel
            color_image[:,:,[0,2]][real_peak_intersect[0] - crosshair_diameter: real_peak_intersect[0] + crosshair_diameter, real_peak_intersect[1] - deviation: real_peak_intersect[1] + deviation] = 0

        return color_image