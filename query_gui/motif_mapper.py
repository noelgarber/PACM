import numpy as np
from PIL import Image
from tifffile import imwrite, imshow
import matplotlib.pyplot as plt
import os
from Motif_Predictor.load_predictor_config import load_config

predictor_params = load_config(verbose=True)
default_db_path = predictor_params["db_params"]["db_path"]
cwd = os.getcwd()

class MotifDomainMap:
    # Object for mapping motifs onto a domain map image for a protein of interest

    def __init__(self, total_residues, scaling_factor = 1.0, protein_id = None):
        self.total_residues = total_residues
        self.height = round(150 * scaling_factor)
        self.width = round(2000 * scaling_factor)
        self.scaling_factor = scaling_factor

        self.arr = np.ones(shape=(self.height, self.width, 3), dtype=float)
        midline_thickness = round(10 * scaling_factor)
        h1 = round(130 * scaling_factor)
        h2 = h1 + midline_thickness
        self.arr[h1:h2, :, :] = 0  # set horizontal midline to black

        self.text_layers = []
        self.protein_id = protein_id
        if protein_id:
            self.label_protein_id(protein_id)

    def label_protein_id(self, protein_id):
        # Place label text using Matplotlib
        protein_label = f"{protein_id}:"
        label_center_position = (0, round(5 * self.scaling_factor))

        # Use 'Agg' backend for off-screen rendering
        original_backend = plt.get_backend()
        plt.switch_backend('Agg')
        plt.figure(figsize=(self.arr.shape[1] / 100, self.arr.shape[0] / 100), dpi=100)
        plt.imshow(self.arr)
        plt.text(label_center_position[1], label_center_position[0], protein_label, fontsize=20, ha='left', va='top',
                 color='black')
        plt.axis('off')

        # Convert plot to image array
        plt.gca().set_position([0, 0, 1, 1])  # Remove padding
        plt.gca().set_axis_off()  # Hide axes
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Render the canvas and convert to numpy array
        plt.gcf().canvas.draw()  # Force the canvas to render
        self.arr = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        self.arr = self.arr.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        self.arr = self.arr.astype(float) / 255
        plt.close()

        plt.switch_backend(original_backend)

    def interpolate_color(self, color1, color2, t):
        t = np.clip(t, 0, 1)
        interpolated_color = color1 + t * (color2 - color1)
        return interpolated_color

    def add_motif_tick(self, motif_start, motif_score, motif_len, min_thickness_ratio=0.005,
                       bottom_color=None, top_color=None):
        # Place the actual motif tick
        distance_from_left = round((motif_start / self.total_residues) * self.arr.shape[1])
        motif_width = round(motif_len / self.total_residues)
        min_thickness = round(min_thickness_ratio * self.arr.shape[1])
        if motif_width >= min_thickness:
            delta_width = 0
            w1 = distance_from_left
            w2 = distance_from_left + motif_width
        else:
            delta_width = min_thickness - motif_width
            w1 = distance_from_left - round(delta_width / 2)
            w2 = distance_from_left + motif_width + round(delta_width / 2)

        motif_height = round(30 * self.scaling_factor)
        h1 = self.arr.shape[0] - motif_height
        h2 = self.arr.shape[0]

        if bottom_color is None:
            bottom_color = np.array([0.75, 0.75, 0.75])
        if top_color is None:
            top_color = np.array([0.0, 0.5, 1.0])
        interpolated_color = self.interpolate_color(bottom_color, top_color, t=motif_score)

        self.arr[h1:h2, w1:w2, :] = interpolated_color

        return distance_from_left, delta_width

    def add_motif_text(self, motif_start, motif_seq, motif_score, motif_len, distance_from_left, delta_width):
        # Place label text using Matplotlib
        text_arr = np.ones_like(self.arr)
        fontsize = 16 * self.scaling_factor

        position_label = f"range={motif_start}:{motif_start + motif_len - 1}"
        score_label = f"score={motif_score:.2f}"

        seq_center_position = (round(45 * self.scaling_factor), distance_from_left + round(motif_len / 2) - round(delta_width / 2))
        start_center_position = (round(75 * self.scaling_factor), distance_from_left + round(motif_len / 2) - round(delta_width / 2))
        score_center_position = (round(105 * self.scaling_factor), distance_from_left + round(motif_len / 2) - round(delta_width / 2))

        # Use 'Agg' backend for off-screen rendering
        original_backend = plt.get_backend()
        plt.switch_backend('Agg')
        plt.figure(figsize=(text_arr.shape[1] / 100, text_arr.shape[0] / 100), dpi=100)
        plt.imshow(text_arr)
        plt.text(seq_center_position[1], seq_center_position[0], motif_seq, fontsize=fontsize, ha='center', va='center', color='black')
        plt.text(start_center_position[1], start_center_position[0], position_label, fontsize=fontsize, ha='center', va='center', color='black')
        plt.text(score_center_position[1], score_center_position[0], score_label, fontsize=fontsize, ha='center', va='center', color='black')
        plt.axis('off')

        # Convert plot to image array
        plt.gca().set_position([0, 0, 1, 1])  # Remove padding
        plt.gca().set_axis_off()  # Hide axes
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Render the canvas and convert to numpy array
        plt.gcf().canvas.draw()  # Force the canvas to render
        text_arr = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        text_arr = text_arr.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        text_arr = text_arr.astype(float) / 255
        plt.close()

        plt.switch_backend(original_backend)

        self.text_layers.append(text_arr)

    def add_motif(self, motif_start, motif_seq, motif_score, motif_len, min_thickness_ratio = 0.005,
                  bottom_color = None, top_color = None):
        if motif_score > 0:
            # Place the actual motif tick on the main image array
            distance_from_left, delta_width = self.add_motif_tick(motif_start, motif_score, motif_len,
                                                                  min_thickness_ratio, bottom_color, top_color)
            # Create label text using Matplotlib and assign as separate layer; will be rasterized later
            self.add_motif_text(motif_start, motif_seq, motif_score, motif_len, distance_from_left, delta_width)

    def get_bounding_box(self, arr):
        # Get the indices of the non-zero elements
        flattened_arr = arr.min(axis=2) if arr.ndim == 3 else arr
        foreground_mask = np.not_equal(flattened_arr, 1)

        foreground_indices = np.argwhere(foreground_mask)
        if len(foreground_indices) == 0:
            return None

        # Determine the bounding box
        top = np.min(foreground_indices[:, 0])
        bottom = np.max(foreground_indices[:, 0]) + 1 # add 1, as these will be used as end indices
        left = np.min(foreground_indices[:, 1])
        right = np.max(foreground_indices[:, 1]) + 1 # Add 1, as these will be used as end indices

        bounding_box = {"top": top, "bottom": bottom, "left": left, "right": right}

        return bounding_box

    def nudge_text_layer(self, text_layer_idx, nudge_amount, bounding_box):
        # Nudge text layer; nudge_amount nudges right if positive or left if negative

        left = bounding_box["left"]
        right = bounding_box["right"]
        top = bounding_box["top"]
        bottom = bounding_box["bottom"]

        width = right - left  # Calculate the width of the bounding box
        new_layer = np.ones_like(self.text_layers[text_layer_idx])
        new_left = left + nudge_amount
        new_right = right + nudge_amount

        # Check if new bounds are within layer dimensions and adjust if necessary
        leftover = 0
        if new_left < 0:
            leftover = -new_left  # Leftover is the amount the left bound exceeds the left edge
            new_left = 0  # Clamp to left edge
            new_right = new_left + width
        if new_right > new_layer.shape[1]:
            leftover += new_right - new_layer.shape[1]  # Add the amount right bound exceeds the right edge
            new_right = new_layer.shape[1]  # Clamp to right edge
            new_left = new_right - width

        # Update bounding box in case further rounds are needed
        new_bounding_box = {"left": new_left, "right": new_right, "top": top, "bottom": bottom}

        # Nudge the layer within valid bounds
        bounded_snippet = self.text_layers[text_layer_idx][top:bottom, left:right, :]
        new_layer[top:bottom, new_left:new_right, :] = bounded_snippet
        self.text_layers[text_layer_idx] = new_layer

        return (new_bounding_box, leftover)

    def resolve_overlaps(self, min_sep = 5):
        # Check for overlaps and resolve them by nudging layers

        # Get bounding boxes for each text_arr, representing the edges of the non-background text
        horizontal_boundaries = {}
        for i, text_arr in enumerate(self.text_layers):
            text_bounding_box = self.get_bounding_box(text_arr)
            horizontal_boundaries[i] = text_bounding_box

        # Iterate over bounding boxes and their corresponding text_arr layers to nudge if required
        leftovers = {}
        for i, bounding_box_1 in horizontal_boundaries.items():
            left1 = bounding_box_1["left"]
            right1 = bounding_box_1["right"]
            mid1 = left1 + ((right1 - left1) / 2)

            for j, bounding_box_2 in horizontal_boundaries.items():
                if i != j:
                    left2 = bounding_box_2["left"]
                    right2 = bounding_box_2["right"]
                    mid2 = left2 + ((right2 - left2) / 2)

                    if mid1 < mid2:
                        leftmost_idx, rightmost_idx = i, j
                        leftmost_bounding_box, rightmost_bounding_box = bounding_box_1, bounding_box_2
                    else:
                        leftmost_idx, rightmost_idx = j, i
                        leftmost_bounding_box, rightmost_bounding_box = bounding_box_2, bounding_box_1

                    overlap_amount = leftmost_bounding_box["right"] - (rightmost_bounding_box["left"] - min_sep)
                    if overlap_amount > 0:
                        nudge_amount_left = -round(overlap_amount / 2) # negative means leftward direction
                        left_nudge_info = self.nudge_text_layer(leftmost_idx, nudge_amount_left, leftmost_bounding_box)
                        leftmost_bounding_box, leftmost_leftover = left_nudge_info
                        horizontal_boundaries[leftmost_idx] = leftmost_bounding_box # update bounding box

                        nudge_amount_right = overlap_amount - abs(nudge_amount_left) + abs(leftmost_leftover)
                        right_nudge_info = self.nudge_text_layer(j, nudge_amount_right, bounding_box_2)
                        rightmost_bounding_box, rightmost_leftover = right_nudge_info
                        horizontal_boundaries[rightmost_idx] = rightmost_bounding_box # update bounding box

                        final_leftover = rightmost_leftover
                        leftovers[(i,j)] = final_leftover

        return leftovers

    def rasterize(self, nudging_rounds_max, min_sep = 5):
        # Rasterize text layers into main image, correcting overlapping labels first

        # Correct overlapping labels
        for i in np.arange(nudging_rounds_max):
            leftovers = self.resolve_overlaps(min_sep)
            if len(leftovers) == 0:
                break
            elif max(leftovers.values()) == 0:
                break

        # Rasterize non-transparent pixels from text layers into the main image
        for text_layer in self.text_layers:
            mask = np.not_equal(text_layer, 1)
            self.arr[mask] = text_layer[mask]
        
        return leftovers

    def to_image(self):
        im = Image.fromarray(self.arr.astype('uint8'))
        return im

    def show(self):
        imshow(self.arr)
        plt.show()

    def save(self, path):
        imwrite(path, self.arr)