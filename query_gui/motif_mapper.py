import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite, imshow
import matplotlib.pyplot as plt
import os
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
def render_text(text, vertical_resolution, use_bold = False, trim_vertical = True):
    """
    Renders a line of text at a specified vertical resolution with antialiasing.

    Args:
        text (str):                    text to render
        vertical_resolution (int):     vertical resolution
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

    # Create a dummy image to calculate text size and position
    dummy_img = Image.new("RGB", (1, 1), (0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    # Get the full text bounding box (considering any characters that extend above or below the typical bounds)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # Draw the text with antialiasing at a calculated position
    img = Image.new("RGB", (text_width, vertical_resolution), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    vertical_position = (vertical_resolution - text_height) // 2 - text_bbox[1]
    draw.text((0, vertical_position), text, font=font, fill=(0, 0, 0))

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

    def add_tick_numbers(self):
        '''
        Adds numbered labels to the motif ticks and corresponding label lines to the list of legend lines.
        '''

        # Sort the tick midpoints and iterate over them from left to right
        tick_horizontal_midpoints = [placement[0] for placement in self.tick_placements]
        sorted_tick_indices = np.argsort(tick_horizontal_midpoints)
        legend_lines = []

        for tick_num, tick_idx in zip(np.arange(1, len(sorted_tick_indices)+1), sorted_tick_indices):
            placement = self.tick_placements[tick_idx]
            tick_horizontal_midpoint, tick_top_edge, start, end, motif_seq, score, specificity = placement

            # Add a numbered label to the motif tick
            tick_num_label = render_text(str(tick_num), round(42 * self.scaling_factor), use_bold=True)
            label_top = tick_top_edge - round(1.25 * tick_num_label.shape[0])
            label_bottom = tick_top_edge - round(0.25 * tick_num_label.shape[0]) - 1
            label_left = tick_horizontal_midpoint - round(tick_num_label.shape[1] / 2)
            label_right = label_left + tick_num_label.shape[1] - 1
            self.arr[label_top:label_bottom+1, label_left:label_right+1, :] = tick_num_label

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

        # Get the legend lines as image arrays
        legend_line_arrs = []
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