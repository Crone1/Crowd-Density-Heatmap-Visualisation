
import argparse
import os.path

import cv2
import yaml

from utils import _exit_if_empty, _exit_if_try_fails


# read in YAML configuration file
with open("configs/default_configs.yaml", "r") as config_file:
    default_configs = yaml.load(config_file, Loader=yaml.FullLoader)

# get variables
default_background_folder = default_configs["default_background_folder"]
default_area_details_folder = default_configs["default_area_details_folder"]
default_base_width = default_configs["default_base_width"]


class DrawingInputs:

    def __init__(self):
        self.background_name = ""
        self.background_folder = default_background_folder
        self.output_folder = default_area_details_folder
        self.base_width = default_base_width

    @staticmethod
    def process_background_image_name(image_path):

        universal_criteria = "the path to the background image points to a valid image file."
        # check it's not empty
        _exit_if_empty(image_path, error="You did not enter a valid path to a background image.", criteria=universal_criteria)

        # check the path exists
        if os.path.exists(image_path):
            background_folder = os.path.dirname(image_path)
            background_name = os.path.basename(image_path)
        elif os.path.exists(os.path.join(default_background_folder, image_path)):
            background_folder = default_background_folder
            background_name = image_path
        else:
            print(
                "\nThe background image name entered does not point to a file that exists"
                " in the current working directory or the default directory."
                " Please re-run this program ensuring the path to the background image points to a valid image file."
            )
            exit(0)

        # check the file is an image file
        _exit_if_try_fails(
            cv2.imread,
            args=[os.path.join(background_folder, background_name)],
            exception=AttributeError,
            error="The file path entered does not point to a valid image file.",
            criteria=universal_criteria
        )

        return background_folder, background_name

    @staticmethod
    def process_output_folder(folder_path):

        universal_criteria = "the path entered points to an existing folder."
        # check it's not empty
        _exit_if_empty(folder_path, error="You did not enter a valid path to a folder.", criteria=universal_criteria)
        # check the path exists
        _exit_if_empty(os.path.exists(folder_path), error="The folder path entered does not exist.", criteria=universal_criteria)
        # check the path is a directory
        _exit_if_empty(os.path.isdir(folder_path), error="The folder path entered does not point to a folder.", criteria=universal_criteria)

        return folder_path

    @staticmethod
    def process_base_width(supplied_base_width):

        universal_criteria = "the width entered for the base of the background image is a valid integer."
        # check it's not empty
        _exit_if_empty(supplied_base_width, error="You did not enter a valid width for the base of the background image.", criteria=universal_criteria)
        # check it's an integer
        _exit_if_try_fails(
            int,
            args=[supplied_base_width],
            exception=ValueError,
            error="You did not enter an integer for the width of the background image.",
            criteria=universal_criteria
        )

        return supplied_base_width

    def get_variables_from_command_line(self):

        parser = argparse.ArgumentParser(description="The variables that make this programme work")

        # background image name
        parser.add_argument(
            dest="background_image_name",
            nargs="?",
            type=str,
            required=True,
            help="The path to the image that will act as the background image for you to draw areas on.",
        )
        # output folder
        parser.add_argument(
            '-of',
            dest="output_folder",
            default=default_area_details_folder,
            nargs="?",
            type=str,
            required=False,
            help="The path to the folder where a file containing the data on the drawn shapes will be output to.",
        )
        # base_width
        parser.add_argument(
            '-bw',
            dest="base_width",
            default=default_base_width,
            nargs="?",
            type=int,
            required=False,
            help="The width that the base of the background image to be scaled to.",
        )

        args = parser.parse_args()

        # process data
        self.background_folder, self.background_name = self.process_background_image_name(args.background_image_name)
        self.output_folder = self.process_output_folder(args.output_folder)
        self.base_width = self.process_base_width(args.base_width)

    def get_variables_from_user(self):

        # background image
        print(
            "\nPlease enter the path to the image that will act as the background image for you to draw areas on."
            "\nThis can be relative to the current working directory"
            " or the default directory: '{}'".format(default_background_folder)
        )
        self.background_folder, self.background_name = self.process_background_image_name(input())

        print("\nTo set the following variables to default values: press 'Enter'.")

        # output folder
        print(
            "\nPlease enter the path to the folder where a file containing the data on the dranw shapes will be output to."
            "\nPress 'Enter' to set this to the default directory: '{}'".format(default_area_details_folder)
        )
        supplied_output_folder = input()
        if supplied_output_folder == "\n":
            self.output_folder = default_area_details_folder
            print("\nThe output folder has been set to the default - '{}'.".format(default_area_details_folder))
        else:
            self.output_folder = self.process_output_folder(supplied_output_folder)

        # base width
        print("\nPlease enter the width that the base of the background image to be scaled to.")
        supplied_base_width = input()
        if supplied_base_width != "\n":
            self.base_width = self.process_base_width(supplied_base_width)
        else:
            self.base_width = default_base_width
            print("\nThe width of image along the x-axis has been set to the default - {}.".format(default_base_width))
