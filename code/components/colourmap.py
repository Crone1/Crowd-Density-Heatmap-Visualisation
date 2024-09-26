# import libraries
import os.path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
# import utilities
from utils.cv2_config import cv2_dict

# read the colourmap customisation configuration variables
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(root_dir, "configs", "colourmap_configs.yaml"), "r") as colmap_config_file:
    colourmap_configs = yaml.load(colmap_config_file, Loader=yaml.FullLoader)
with open(os.path.join(root_dir, "configs", "default_configs.yaml"), "r") as default_config_file:
    data_configs = yaml.load(default_config_file, Loader=yaml.FullLoader)["data"]


class ColourMap:

    def __init__(self, height, width):
        self.final_height = int(height)
        self.final_width = int(width)
        self.inner_width = None
        self.inner_height = None
        self.image = None
        self.mapper = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=data_configs["min_value"], vmax=data_configs["max_value"]),
            cmap=colourmap_configs["background"]["cmap_name"]
        )

    @staticmethod
    def _abbreviate_num(num):
        """
        Function Goal : turn a number into a 3 digit string and use an abbreviation to represent its magnitude

        num : integer - number

        return : string - abbreviated number
        """
        num_digits = len(str(num).rstrip("0").rstrip("."))
        format_string = "{:.3g}"

        if 3 < num_digits < 7:
            return format_string.format(num / 1000) + "K"
        elif 6 < num_digits < 10:
            return format_string.format(num / 1000000) + "M"
        elif 9 < num_digits < 13:
            return format_string.format(num / 1000000000) + "B"
        elif 12 < num_digits < 16:
            return format_string.format(num / 1000000000000) + "T"
        else:
            return format_string.format(num)

    def _draw_heading(self):

        # define heading text
        heading_thickness = int(self.inner_width * colourmap_configs["text"]["heading"]["proportions"]["thickness"])
        heading_size = self.inner_width * colourmap_configs["text"]["heading"]["proportions"]["size"]
        heading_font = cv2_dict[colourmap_configs["text"]["heading"]["font_type"]]
        heading_width, heading_height = cv2.getTextSize(
            data_configs["title"], heading_font, heading_size, thickness=heading_thickness,
        )[0]

        # define position to draw heading text on image
        heading_box_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["heading"])
        heading_start_x_coord = int(self.inner_width / 2 - heading_width / 2)
        heading_start_y_coord = int(heading_box_height / 2 + heading_height / 2)

        # put the text on the image
        cv2.putText(
            self.image,
            data_configs["title"],
            (heading_start_x_coord, heading_start_y_coord),
            heading_font,
            heading_size,
            colourmap_configs["text"]["heading"]["colour"],
            lineType=colourmap_configs["text"]["heading"]["line_type"],
            thickness=heading_thickness,
        )

    def _draw_heading_seperator(self):

        # define position to draw line on image
        heading_box_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["heading"])
        seperator_thickness = int(self.inner_height * colourmap_configs["proportions"]["height"]["seperator"])

        # draw line to separate heading area
        seperator_colour = colourmap_configs["lines"]["seperator"]["colour"]
        self.image[heading_box_height + 1: heading_box_height + 1 + seperator_thickness, :, :] = seperator_colour

    def _draw_spectrum(self):

        # define position to draw spectrum on image
        bottom_gap_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["bottom_gap"])
        spectrum_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["spectrum"])
        y_coord = self.inner_height - bottom_gap_height
        x_coord = int(self.inner_width * colourmap_configs["proportions"]["width"]["gap"])

        # define spectrum values to get colours for
        spectrum_width = int(self.inner_width * colourmap_configs["proportions"]["width"]["spectrum"])
        spectrum_vals = np.linspace(start=data_configs["min_value"], stop=data_configs["max_value"], num=spectrum_width)

        # draw the colourmap spectrum on the image
        for val in spectrum_vals:
            colour = self.mapper.to_rgba(val)[:3][::-1]
            self.image[(y_coord - spectrum_height):y_coord, x_coord, :] = colour
            x_coord = x_coord + 1

    def _draw_spectrum_index_lines(self):

        # define position to draw the lines
        bottom_gap_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["bottom_gap"])
        line_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["spectrum"]) + int(
            self.inner_height * colourmap_configs["proportions"]["height"]["index_ticks"])
        y_coord = self.inner_height - bottom_gap_height

        # define x-coordinates to draw the lines at
        spectrum_width = int(self.inner_width * colourmap_configs["proportions"]["width"]["spectrum"])
        start_x_coord = int(self.inner_width * colourmap_configs["proportions"]["width"]["gap"])
        colourmap_divider = spectrum_width / colourmap_configs["lines"]["index"]["num"]
        line_x_coords = [int(i * colourmap_divider) + start_x_coord for i in
                         range(0, colourmap_configs["lines"]["index"]["num"] + 1)]

        # draw the colourmap index lines
        for x_coord in line_x_coords:
            colour = colourmap_configs["lines"]["index"]["colour"]
            half_thickness = int(self.inner_width * colourmap_configs["lines"]["index"]["width_proportion"]) // 2
            self.image[(y_coord - line_height):y_coord, (x_coord - half_thickness):(x_coord + half_thickness),
            :] = colour

        return line_x_coords, y_coord - line_height

    def _draw_spectrum_index_values(self, x_coords, start_y_coord):

        # define position to draw the indexes
        index_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["index_text"])
        index_gap_height = int(self.inner_height * colourmap_configs["proportions"]["height"]["index_gap"])
        y_coord = start_y_coord - index_gap_height

        # define index values to draw
        index_increment = (data_configs["max_value"] - data_configs["min_value"]) / colourmap_configs["lines"]["index"][
            "num"]
        index_values = [self._abbreviate_num((i * index_increment) + data_configs["min_value"]) for i in
                        range(0, colourmap_configs["lines"]["index"]["num"] + 1)]

        # define text variables
        index_thickness = int(self.inner_width * colourmap_configs["text"]["index"]["proportions"]["thickness"])
        index_size = self.inner_width * colourmap_configs["text"]["index"]["proportions"]["size"]
        index_font = cv2_dict[colourmap_configs["text"]["index"]["font_type"]]

        # draw the colourmap index values
        for index, x_coord in zip(index_values, x_coords):
            index_width, _ = cv2.getTextSize(index, index_font, index_size, thickness=index_thickness)[0]

            # draw the text on the image
            cv2.putText(
                self.image,
                index,
                (x_coord - (index_width // 2), y_coord),
                index_font,
                index_size,
                color=colourmap_configs["text"]["index"]["colour"],
                lineType=cv2_dict[colourmap_configs["text"]["index"]["line_type"]],
                thickness=index_thickness,
            )

    def create(self):
        """
        Function Goal : create the colourmap

        desired_height : integer - the height of the final created colourmap
        desired_width : integer - the width of the final created colourmap

        return : a 3D numpy array of integers - an array that corresponds to the colourmap image that was created
        """

        # define blank colourmap array
        border_width = int(self.final_width * colourmap_configs["lines"]["border"]["width_proportion"])
        self.inner_width = self.final_width - (2 * border_width)
        self.inner_height = self.final_height - (2 * border_width)
        self.image = np.ones((self.inner_height, self.inner_width, 3)) * colourmap_configs["background"]["colour"]

        # draw colourmap
        self._draw_heading()
        self._draw_heading_seperator()
        self._draw_spectrum()
        line_x_coords, y_coord = self._draw_spectrum_index_lines()
        self._draw_spectrum_index_values(line_x_coords, y_coord)

        # put border on colourmap
        self.image = cv2.copyMakeBorder(
            self.image, top=border_width, bottom=border_width, left=border_width, right=border_width,
            borderType=cv2_dict[colourmap_configs["lines"]["border"]["type"]],
            value=colourmap_configs["lines"]["border"]["colour"]
        )

    def plot(self):
        plt.imshow(self.image)
        plt.show()


def main():
    width = input("What width colourmap do you want to create? ")
    height = input("What height colourmap do you want to create? ")

    cmap = ColourMap(height, width)
    cmap.create()
    cmap.plot()


if __name__ == "__main__":
    main()
