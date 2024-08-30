import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite, imshow
import matplotlib.pyplot as plt
import os
import warnings
from Motif_Predictor.load_predictor_config import load_config

predictor_params = load_config(verbose=True)
default_db_path = predictor_params["db_params"]["db_path"]
cwd = os.getcwd()

def interpolate_color(color1, color2, t):
    t = np.clip(t, 0, 1)
    interpolated_color = color1 + t * (color2 - color1)
    return interpolated_color

regular_font_path = os.path.join(os.getcwd(), "fonts", "DejaVuSans.ttf")
bold_font_path = os.path.join(os.getcwd(), "fonts", "DejaVuSans-Bold.ttf")
def render_text(text, vertical_resolution, use_bold=False, trim_vertical=True):
    """
    Renders a block of text at a specified vertical resolution with antialiasing.

    Args:
        text (str):                    text to render (can be multi-line)
        vertical_resolution (int):     vertical resolution of each line
        use_bold (bool):               whether to use bold font
        trim_vertical (bool):          whether to trim top and bottom whitespace;
                                       may result in vertical shape not matching vertical resolution

    Returns:
        cropped_img (np.ndarray):      the rendered text as a numpy image array
    """
    # Determine the font size and load the font
    font_size = vertical_resolution
    font_path = bold_font_path if use_bold else regular_font_path
    font = ImageFont.truetype(font_path, font_size)

    # Split text into lines
    lines = text.split("\n")

    # Create a dummy image to calculate text size and position
    dummy_img = Image.new("RGB", (1, 1), (0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    # Get font metrics for proper spacing
    ascent, descent = font.getmetrics()
    line_spacing = font.getmask("Ag").getbbox()[3] + descent  # height of one line including descent

    # Get the maximum width and total height of the text block
    max_width = 0
    total_height = 0
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = text_bbox[2] - text_bbox[0]
        max_width = max(max_width, line_width)
        total_height += line_spacing

    # Draw the text with antialiasing at calculated positions
    img = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    current_y = 0
    for line in lines:
        draw.text((0, current_y), line, font=font, fill=(0, 0, 0))
        current_y += line_spacing

    # Convert the image to a numpy array
    img = np.array(img).astype(float) / 255

    # Crop the image to remove any excess whitespace
    background_mask = img.min(axis=2) == 1
    background_cols = background_mask.min(axis=0).astype(bool)
    foreground_col_indices = np.where(~background_cols)[0]
    left_edge = foreground_col_indices.min()
    right_edge = foreground_col_indices.max()

    if trim_vertical:
        background_rows = background_mask.min(axis=1).astype(bool)
        foreground_row_indices = np.where(~background_rows)[0]
        top_edge = foreground_row_indices.min()
        bottom_edge = foreground_row_indices.max()
        cropped_img = img[top_edge:bottom_edge+1, left_edge:right_edge+1, :]
    else:
        cropped_img = img[:, left_edge:right_edge+1, :]

    return cropped_img

class MotifDomainMap:
    # Object for mapping motifs onto a domain map image for a protein of interest

    def __init__(self, total_residues, scaling_factor = 1.0, protein_id = None,
                 protein_label_fontsize = 42, protein_label_offset = 28, legend_fontsize = 36):
        self.total_residues = total_residues
        self.height = round(150 * scaling_factor)
        self.width = round(2000 * scaling_factor)
        self.scaling_factor = scaling_factor
        self.legend_fontsize = legend_fontsize

        self.arr = np.ones(shape=(self.height, self.width, 3), dtype=float)
        midline_thickness = round(10 * scaling_factor)
        h1 = round(130 * scaling_factor)
        h2 = h1 + midline_thickness
        self.arr[h1:h2, :, :] = 0  # set horizontal midline to black

        self.legend_exists = False
        self.tick_placements = []
        self.protein_id = protein_id
        if protein_id:
            self.protein_label_fontsize = protein_label_fontsize
            self.protein_label_offset = protein_label_offset
            self.label_protein_id(protein_id)

    def label_protein_id(self, protein_id, left_offset=0):
        # Place protein label in top left corner

        protein_label = render_text(f"Isoform {protein_id}:", round(self.protein_label_fontsize * self.scaling_factor))
        top = self.protein_label_offset
        bottom = top + protein_label.shape[0]
        left = left_offset
        right = left + protein_label.shape[1]
        self.arr[top:bottom, left:right, :] = protein_label

    def infer_color(self, motif_score, specificity_score=None, bottom_color=None, mid_color=None, top_color=None,
                    color_ranges=None):
        # Interpolate tick color based on score

        if specificity_score is not None:
            bottom_color = np.array([0.0, 1.0, 0.0]) if bottom_color is None else bottom_color
            mid_color = np.array([1.0, 1.0, 1.0]) if mid_color is None else mid_color
            top_color = np.array([1.0, 0.35, 0.35]) if top_color is None else top_color
            color_ranges = (-2.0, 0.0, 2.0) if color_ranges is None else color_ranges
            color_score = specificity_score
        else:
            bottom_color = np.array([0.75, 0.75, 0.75]) if bottom_color is None else bottom_color
            top_color = np.array([0.0, 0.5, 1.0]) if top_color is None else top_color
            color_ranges = (0.0, 1.0) if color_ranges is None else color_ranges
            color_score = motif_score

        if len(color_ranges) == 2:
            scaling_score = (color_score - color_ranges[0]) / (color_ranges[1] - color_ranges[0])
            interpolated_color = interpolate_color(bottom_color, top_color, t=scaling_score)
        elif len(color_ranges) == 3:
            if color_score <= color_ranges[0]:
                interpolated_color = bottom_color
            elif color_score == color_ranges[1]:
                interpolated_color = mid_color
            elif color_score >= color_ranges[2]:
                interpolated_color = top_color
            elif color_score > color_ranges[0] and color_score < color_ranges[1]:
                scaling_score = (color_score - color_ranges[0]) / (color_ranges[1] - color_ranges[0])
                interpolated_color = interpolate_color(bottom_color, mid_color, t=scaling_score)
            elif color_score > color_ranges[1] and color_score < color_ranges[2]:
                scaling_score = (color_score - color_ranges[1]) / (color_ranges[2] - color_ranges[1])
                interpolated_color = interpolate_color(mid_color, top_color, t=scaling_score)
            else:
                print(f"color_score out of range: {color_score}")
        else:
            raise Exception(f"color_ranges requires an ascending series of 2 or 3 values, "
                            f"but {len(color_ranges)} were given")

        return interpolated_color

    def get_tick_dims(self, motif_start, motif_len, min_thickness_ratio=0.005):
        '''
        Calculates the dimensions of the tick to denote the motif of interest.

        Args:
            motif_start (int):           motif starting position in the protein sequence
            motif_len (int):             motif length
            min_thickness_ratio (float): minimum thickness of the tick as a fraction of the total domain map width
        '''

        tick_horizontal_midpoint = round((motif_start / self.total_residues) * self.arr.shape[1])
        motif_width = round(motif_len / self.total_residues)
        min_thickness = round(min_thickness_ratio * self.arr.shape[1])
        if motif_width >= min_thickness:
            delta_width = 0
            w1 = tick_horizontal_midpoint
            w2 = tick_horizontal_midpoint + motif_width
        else:
            delta_width = min_thickness - motif_width
            w1 = tick_horizontal_midpoint - round(delta_width / 2)
            w2 = tick_horizontal_midpoint + motif_width + round(delta_width / 2)

        motif_height = round(30 * self.scaling_factor)
        h1 = self.arr.shape[0] - motif_height
        h2 = self.arr.shape[0]

        return (h1, h2, w1, w2, tick_horizontal_midpoint)

    def add_motif_tick(self, motif_start, score, motif_len, specificity=None, min_thickness_ratio=0.005, tick_outline=0,
                       bottom_color=None, mid_color=None, top_color=None, color_ranges=None, opacity_range=(0,1)):
        '''
        Function for adding a motif tick to the domain map.

        Args:
            motif_start (int):           motif starting position in the protein sequence
            score (float):               motif confidence score
            motif_len (int):             motif length
            specificity (float):         motif specificity score (optional)
            min_thickness_ratio (float): minimum thickness of the tick as a fraction of the total domain map width
            tick_outline (int):          thickness of black outline around tick
            bottom_color (tuple):        base color for lowest score
            mid_color (tuple):           midpoint color
            top_color (tuple):           top color for highest score
            color_ranges (tuple):        score ranges for interpolating tick color
            opacity_range (tuple):       score range for determining opacity; only applies when specficity score exists

        Returns:
            tick_horizontal_midpoint (int): horizontal midpoint of drawn tick
            tick_top_edge (int):            top edge of drawn tick
        '''

        # Get the tick dimensions and interpolate the tick color
        h1, h2, w1, w2, tick_horizontal_midpoint = self.get_tick_dims(motif_start, motif_len, min_thickness_ratio)
        tick_top_edge = h1
        interpolated_color = self.infer_color(score, specificity, bottom_color, mid_color, top_color, color_ranges)

        # Apply the tick onto the image
        if specificity is not None:
            opacity = (score - opacity_range[0]) / (opacity_range[1] - opacity_range[0])
            old_tick_region = self.arr[h1:h2, w1:w2, :]
            new_tick = np.zeros_like(old_tick_region, dtype=float)
            new_tick[:,:,:] = interpolated_color
            blended_tick = ((1-opacity) * old_tick_region) + (opacity * new_tick)
            self.arr[h1:h2, w1:w2, :] = blended_tick
        else:
            self.arr[h1:h2, w1:w2, :] = interpolated_color

        # Apply the tick outline if desired
        if tick_outline > 0:
            if specificity is not None:
                weakest_outline_color = np.array([1.0, 1.0, 1.0])
                strongest_outline_color = np.array([0.0, 0.5, 1.0])
                confidence = (score - opacity_range[0]) / (opacity_range[1] - opacity_range[0])
                tick_outline_color = interpolate_color(weakest_outline_color, strongest_outline_color, t=confidence)
            else:
                tick_outline_color = np.array([0.0, 0.0, 0.0])

            tick_outline = round(self.scaling_factor * tick_outline)
            self.arr[h1:h1+tick_outline, w1-tick_outline:w2+tick_outline] = tick_outline_color # top line
            self.arr[h2-tick_outline:h2, w1-tick_outline:w2+tick_outline] = tick_outline_color # bottom line
            self.arr[h1:h2, w1-tick_outline:w1] = tick_outline_color # left line
            self.arr[h1:h2, w2:w2+tick_outline] = tick_outline_color # right line

        return (tick_horizontal_midpoint, tick_top_edge)

    def tick_num_coords(self, tick_top_edge, tick_horizontal_midpoint, tick_num_label):
        # Helper function that retrieves coords to apply the tick num label

        label_top = tick_top_edge - round(1.25 * tick_num_label.shape[0])
        label_bottom = tick_top_edge - round(0.25 * tick_num_label.shape[0]) - 1
        label_left = tick_horizontal_midpoint - round(tick_num_label.shape[1] / 2)
        label_right = label_left + tick_num_label.shape[1] - 1

        return (label_top, label_bottom, label_left, label_right)

    def get_adjacent_label(self, current_tick_num, sorted_tick_indices, side = "left"):
        # Helper function that retrieves adjacent label assignment coordinates

        adjacent_tick_num = current_tick_num - 1 if side == "left" else current_tick_num + 1
        adjacent_tick_idx = sorted_tick_indices[adjacent_tick_num - 1]
        adjacent_placement = self.tick_placements[adjacent_tick_idx]
        adjacent_horizontal_midpoint, adjacent_top_edge = adjacent_placement[:2]
        adjacent_num_label = render_text(str(adjacent_tick_num), round(42 * self.scaling_factor), use_bold=True)
        adjacent_coords = self.tick_num_coords(adjacent_top_edge, adjacent_horizontal_midpoint, adjacent_num_label)
        adjacent_top, adjacent_bottom, adjacent_left, adjacent_right = adjacent_coords

        return (adjacent_top, adjacent_bottom, adjacent_left, adjacent_right)

    def get_label_coords(self, tick_num, tick_top_edge, tick_horizontal_midpoint, tick_num_label, sorted_tick_indices,
                         arr_right_edge, prev_right = None, next_left = None):
        '''
        Dynamically gets coordinates for where to assign the tick number label

        Args:
            tick_num (int):                   Current tick number
            tick_top_edge (int):              Current tick top edge
            tick_horizontal_midpoint (int):   Current tick horizontal midpoint
            tick_num_label (np.ndarray):      Current tick number as a rasterized label image array
            sorted_tick_indices (np.ndarray): Sorted tick indices
            arr_right_edge (int):             Right edge of the parent image
            prev_right (int|None):            Previous tick right edge; can be optionally given in advance
            next_left (int|None):             Next tick left edge; can be optionally given in advance

        Returns:
            label_coords (tuple):           Tuple of top, bottom, left, and right edge coordinates
        '''

        top, bottom, left, right = self.tick_num_coords(tick_top_edge, tick_horizontal_midpoint, tick_num_label)

        if tick_num > 1 and tick_num < len(sorted_tick_indices):
            # Handle cases where non-edge ticks
            if prev_right is None:
                _, _, _, prev_right = self.get_adjacent_label(tick_num, sorted_tick_indices, side="left")
            if next_left is None:
                _, _, next_left, _ = self.get_adjacent_label(tick_num, sorted_tick_indices, side="right")
            overlap_with_prev = prev_right - left  # positive when overlap exists
            overlap_with_next = right - next_left  # positive when overlap exists

            if overlap_with_prev > 0 and overlap_with_next < 0:
                # Overlaps on the left, but not on the right
                room_to_nudge = -overlap_with_next
                if overlap_with_prev < room_to_nudge:
                    # Sufficient room to fully resolve the overlap
                    left += overlap_with_prev
                    right += overlap_with_prev
                elif room_to_nudge > 0:
                    # Insufficient room, so some leftover overlap will persist
                    left += room_to_nudge
                    right += room_to_nudge

            elif overlap_with_next > 0 and overlap_with_prev < 0:
                # Overlaps on the right, but not on the left
                room_to_nudge = -overlap_with_prev
                if overlap_with_next < room_to_nudge:
                    # Sufficient room to fully resolve the overlap
                    left -= overlap_with_next
                    right -= overlap_with_next
                elif room_to_nudge > 0:
                    # Insufficient room, so some leftover overlap will persist
                    left -= room_to_nudge
                    right -= room_to_nudge

        elif tick_num == 0:
            # First tick; no previous tick to consider
            if next_left is None:
                _, _, next_left, _ = self.get_adjacent_label(tick_num, sorted_tick_indices, side="right")
            overlap_with_next = right - next_left  # positive when overlap exists

            if overlap_with_next > 0 and left > 0:
                # Overlaps on the right, but still has some room on the left
                room_to_nudge = left
                if overlap_with_next < room_to_nudge:
                    # Sufficient room to fully resolve the overlap
                    left -= overlap_with_next
                    right -= overlap_with_next
                elif room_to_nudge > 0:
                    # Insufficient room, so some leftover overlap will persist
                    left -= room_to_nudge
                    right -= room_to_nudge

        elif tick_num == len(sorted_tick_indices):
            # Last tick; no next tick to consider
            if prev_right is None:
                _, _, _, prev_right = self.get_adjacent_label(tick_num, sorted_tick_indices, side="left")
            overlap_with_prev = prev_right - left  # positive when overlap exists

            if overlap_with_prev > 0 and right < arr_right_edge:
                # Overlaps on the left, but not on the right
                room_to_nudge = arr_right_edge - right
                if overlap_with_prev < room_to_nudge:
                    # Sufficient room to fully resolve the overlap
                    left += overlap_with_prev
                    right += overlap_with_prev
                elif room_to_nudge > 0:
                    # Insufficient room, so some leftover overlap will persist
                    left += room_to_nudge
                    right += room_to_nudge

        label_coords = (top, bottom, left, right)

        return label_coords

    def add_tick_numbers(self):
        '''
        Adds numbered labels to the motif ticks and corresponding label lines to the list of legend lines.
        '''

        # Sort the tick midpoints and iterate over them from left to right
        tick_horizontal_midpoints = [placement[0] for placement in self.tick_placements]
        sorted_tick_indices = np.argsort(tick_horizontal_midpoints)
        legend_lines = []

        prev_right = None
        for tick_num, tick_idx in zip(np.arange(1, len(sorted_tick_indices)+1), sorted_tick_indices):
            placement = self.tick_placements[tick_idx]
            tick_horizontal_midpoint, tick_top_edge, start, end, motif_seq, score, specificity = placement

            # Create a numbered label for the motif tick
            tick_num_label = render_text(str(tick_num), round(42 * self.scaling_factor), use_bold=True)

            # Get coordinates for applying the label dynamically, avoiding overlaps
            top, bottom, left, right = self.get_label_coords(tick_num, tick_top_edge, tick_horizontal_midpoint,
                                                             tick_num_label, sorted_tick_indices,
                                                             self.arr.shape[1], prev_right)
            prev_right = right # reset for next round

            # Rasterize the tick number label onto the main image
            expanded_label = np.ones_like(self.arr, dtype=float)
            expanded_label[top:bottom+1, left:right+1, :] = tick_num_label

            label_foreground_mask = np.not_equal(tick_num_label.min(axis=2), 1)
            expanded_foreground_mask = np.zeros(shape=(self.arr.shape[0], self.arr.shape[1]), dtype=bool)
            expanded_foreground_mask[top:bottom+1, left:right+1] = label_foreground_mask

            self.arr[expanded_foreground_mask] = expanded_label[expanded_foreground_mask]

            # Add a line to the legend for this numbered motif
            if specificity is None:
                combined_label = f"Motif #{tick_num}: score = {score:.2f}, range = {start}:{end}, motif = {motif_seq}"
            else:
                combined_label = (f"Motif #{tick_num}: score = {score:.2f}, specificity = {specificity:.2f}, "
                                  f"range = {start}:{end}, motif = {motif_seq}")
            legend_lines.append(combined_label)

        self.legend_lines = legend_lines

    def get_legend_arr(self, legend_fontsize=36):
        '''
        Converts legend_lines into a legend image array.
        '''

        scaled_fontsize = round(legend_fontsize * self.scaling_factor)

        # Separate the legend lines into their constitutive elements for left-alignment
        elements_lines = []
        for legend_line in self.legend_lines:
            motif_num, scores_info = legend_line.split(": ")
            elements_line = [f"{motif_num}: "]
            elements_line.extend(scores_info.split(", "))
            elements_lines.append(elements_line)

        element_counts = [len(elements_line) for elements_line in elements_lines]
        matching_counts = np.all([element_count == element_counts[0] for element_count in element_counts[1:]])
        if matching_counts:
            element_count = element_counts[0]

            # Use dummy character spacer to vertically align columns
            dummy_spacer = render_text("\t", scaled_fontsize, trim_vertical=True)
            dummy_spacer_height = dummy_spacer.shape[0]

            # Get column image arrays
            col_arrs = []
            for element_idx in np.arange(element_count):
                col_elements = [elements_line[element_idx] for elements_line in elements_lines]
                col_elements_str = "\n".join(col_elements)
                col_elements_str = f"\t\n{col_elements_str}"
                col_elements_arr = render_text(col_elements_str, scaled_fontsize, trim_vertical=True)
                col_elements_arr = col_elements_arr[dummy_spacer_height:]
                col_arrs.append(col_elements_arr)

            # Render columns into one image
            col_spacing = round(scaled_fontsize / 2)
            legend_height = max([col_arr.shape[0] for col_arr in col_arrs])
            legend_width = sum([col_arr.shape[1] for col_arr in col_arrs])
            legend_width += (len(col_arrs)-1) * col_spacing
            legend_arr = np.ones(shape=(legend_height, legend_width, 3), dtype=float)

            current_left_edge = 0
            for col_arr in col_arrs:
                legend_arr[:col_arr.shape[0], current_left_edge:current_left_edge+col_arr.shape[1], :] = col_arr
                current_left_edge += col_arr.shape[1]
                current_left_edge += col_spacing

        else:
            # Render legend lines without respect to element alignment
            warnings.warn(f"Could not left-align legend element cols due to varying number of elements per line.")

            for legend_line in self.legend_lines:
                legend_line_arr = render_text(legend_line, round(legend_fontsize * self.scaling_factor), trim_vertical=False)
                legend_line_arrs.append(legend_line_arr)

            # Render the legend as one image
            combined_height = sum([arr.shape[0] for arr in legend_line_arrs])
            widest_line = max([arr.shape[1] for arr in legend_line_arrs])
            legend_arr = np.ones(shape=(combined_height, widest_line, 3), dtype=float)

            current_top_edge = 0
            for legend_line_arr in legend_line_arrs:
                bottom_edge = current_top_edge + legend_line_arr.shape[0]
                right_edge = legend_line_arr.shape[1]
                legend_arr[current_top_edge:bottom_edge, 0:right_edge, :] = legend_line_arr
                current_top_edge = bottom_edge

        self.legend_arr = legend_arr
        self.legend_exists = True

    def apply_corner_legend(self, outline=2, outline_offset=4):
        # Try to apply legend into corner of image

        outlined_within_vertical = (self.legend_arr.shape[0] + (4 * outline)) <= self.arr_with_legend.shape[0]
        outlined_within_horizontal = (self.legend_arr.shape[1] + (4 * outline)) <= self.arr_with_legend.shape[1]

        rendered_legend = False
        if outlined_within_vertical and outlined_within_horizontal:
            top = 0
            bottom = top + self.legend_arr.shape[0] + (4 * outline)
            right = self.arr_with_legend.shape[1]
            left = right - self.legend_arr.shape[1] - (4 * outline)
            top_right_space = self.arr_with_legend[top:bottom, left:right, :]

            if np.all(np.equal(top_right_space, 1)):
                if outline > 0:
                    combined_offset = outline + outline_offset
                    outlined_height = self.legend_arr.shape[0] + (2*combined_offset)
                    outlined_width = self.legend_arr.shape[1] + (2*combined_offset)

                    outlined_legend_arr = np.ones(shape=(outlined_height, outlined_width, 3), dtype=float) * 0.5
                    outlined_legend_arr[outline:-outline, outline:-outline, :] = 1.0
                    outlined_legend_arr[combined_offset:-combined_offset, combined_offset:-combined_offset, :] = self.legend_arr

                    self.arr_with_legend[top:bottom, left:right, :] = outlined_legend_arr

                else:
                    self.arr_with_legend[top:bottom, left:right, :] = self.legend_arr

                rendered_legend = True

        return rendered_legend

    def generate_legend(self, placement = "corner", corner_legend_outline = 0):
        '''
        Renders a version of the main array with the legend applied.

        Args:
            legend_arr (np.ndarray): the legend as an image array
        '''

        self.arr_with_legend = self.arr.copy()
        self.get_legend_arr(self.legend_fontsize)

        rendered_legend = False
        if placement == "corner":
            rendered_legend = self.apply_corner_legend(corner_legend_outline)
            if not rendered_legend:
                # Try again with a smaller font
                self.get_legend_arr(self.legend_fontsize)
                rendered_legend = self.apply_corner_legend(corner_legend_outline)
                if rendered_legend:
                    # Reassign legend font size if downsizing was successful
                    self.legend_fontsize = round(self.legend_fontsize * 0.75)

        # Warn the user if the selected placement is not possible
        if not rendered_legend and placement == "corner":
            print(f"Could not render legend in top right corner due to insufficient space; rendering at bottom.")
        elif not rendered_legend and placement != "bottom":
            print(f"Unrecognized placement argument \"{placement}\"; placing at bottom of image.")

        # Render at bottom if not already rendered in the corner before
        if not rendered_legend:
            # Make array with whitespace for legend at the bottom
            gap = round(self.scaling_factor * 10)
            rendered_height = self.arr.shape[0] + gap + self.legend_arr.shape[0]
            rendered_width = max(self.arr.shape[1], self.legend_arr.shape[1])
            self.arr_with_legend = np.ones(shape=(rendered_height, rendered_width, 3), dtype=float)
            self.arr_with_legend[0:self.arr.shape[0], 0:self.arr.shape[1], :] = self.arr

            # Apply the legend
            top = self.arr.shape[0] + gap
            bottom = top + self.legend_arr.shape[0]
            left = 0
            right = left + self.legend_arr.shape[1]
            self.arr_with_legend[top:bottom, left:right, :] = self.legend_arr
            rendered_legend = True

    def label_motifs(self, placement = "corner"):
        '''
        Adds numbered labels to the motif ticks and corresponding label lines to the list of legend lines.
        '''

        # Add tick numbers to main array and create legend lines for rendering later
        self.add_tick_numbers()

        # Render the legend as an image, then apply it to the main image
        self.get_legend_arr()
        self.generate_legend(placement)

    def add_motif(self, start, seq, score, specificity, motif_len, min_thickness_ratio=0.005, tick_outline=0,
                  bottom_color=None, mid_color=None, top_color=None, color_ranges=None, opacity_range=(0,1),
                  legend_placement="corner"):
        '''
        Main function for adding a motif to the domain map and labelling it appropriately.

        Args:
            start (int):                 motif starting position in the protein sequence
            seq (str):                   motif sequence
            score (float):               motif confidence score
            specificity (float):         motif specificity score (optional)
            motif_len (int):             motif length
            min_thickness_ratio (float): minimum thickness of the tick as a fraction of the total domain map width
            tick_outline (int):          thickness of black outline around tick
            bottom_color (tuple):        base color for lowest score
            mid_color (tuple):           midpoint color
            top_color (tuple):           top color for highest score
            color_ranges (tuple):        score ranges for interpolating tick color
            opacity_range (tuple):       score range for determining opacity; only applies when specficity score exists
        '''

        if score > 0:
            # Place the actual motif tick on the main image array
            placement_info = self.add_motif_tick(start, score, motif_len, specificity, min_thickness_ratio, tick_outline,
                                                 bottom_color, mid_color, top_color, color_ranges, opacity_range)
            tick_horizontal_midpoint, tick_top_edge = placement_info

            # Record the tick placement and motif info, then use for labelling and constructing a legend
            end = start + motif_len - 1
            self.tick_placements.append((tick_horizontal_midpoint, tick_top_edge, start, end, seq, score, specificity))
            self.label_motifs(legend_placement)

    def get_arr(self):
        if self.legend_exists:
            return self.arr_with_legend
        else:
            return self.arr

    def show(self):
        imshow(self.get_arr())
        plt.show()

    def save(self, path):
        imwrite(path, self.get_arr())