
import argparse
import os.path

import cv2
import yaml

from utils.input_utils import exit_if_empty, exit_if_try_fails


# read in YAML configuration file
with open("configs/default_configs.yaml", "r") as config_file:
    default_configs = yaml.load(config_file, Loader=yaml.FullLoader)

# get variables
default_background_folder = default_configs["default_background_folder"]
default_area_details_folder = default_configs["default_area_details_folder"]
default_base_width = default_configs["default_base_width"]


class DrawingInputs:

    def __init__(self):
        self.background_path = ""
        self.output_folder = default_area_details_folder
        self.base_width = self._process_base_width(default_base_width)

    @staticmethod
    def _process_base_width(supplied_base_width):

        universal_criteria = "the width entered for the base of the background image is a valid integer."
        # check it's not empty
        exit_if_empty(supplied_base_width, error="The configured width for the base of the background image is empty.", criteria=universal_criteria)
        # check it's an integer
        exit_if_try_fails(
            int,
            args=[supplied_base_width],
            exception=ValueError,
            error="The configured width for the base of the background image is not an integer.",
            criteria=universal_criteria
        )

        return int(supplied_base_width)

    @staticmethod
    def _process_background_image_name(image_path):

        universal_criteria = "the path to the background image points to a valid image file."
        # check it's not empty
        exit_if_empty(image_path, error="You did not enter a valid path to a background image.", criteria=universal_criteria)

        # check the path exists
        if os.path.exists(image_path):
            verified_image_path = image_path
        elif os.path.exists(os.path.join(default_background_folder, image_path)):
            verified_image_path = os.path.join(default_background_folder, image_path)
        else:
            print(
                "\nThe background image name entered does not point to a file that exists"
                " in the current working directory or the default directory."
                " Please re-run this program ensuring the path to the background image points to a valid image file."
            )
            exit(0)

        # check the file is an image file
        exit_if_try_fails(
            cv2.imread,
            args=[verified_image_path],
            exception=AttributeError,
            error="The file path entered does not point to a valid image file.",
            criteria=universal_criteria
        )

        return verified_image_path

    @staticmethod
    def _process_output_folder(folder_path):

        universal_criteria = "the path entered points to an existing folder."
        # check it's not empty
        exit_if_empty(folder_path, error="You did not enter a valid path to a folder.", criteria=universal_criteria)
        # check the path exists
        exit_if_empty(os.path.exists(folder_path), error="The folder path entered does not exist.", criteria=universal_criteria)
        # check the path is a directory
        exit_if_empty(os.path.isdir(folder_path), error="The folder path entered does not point to a folder.", criteria=universal_criteria)

        return folder_path

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

        args = parser.parse_args()

        # process data
        self.background_path = self._process_background_image_name(args.background_image_name)
        self.output_folder = self._process_output_folder(args.output_folder)

    def get_variables_from_user(self):

        # background image
        print(
            "\nPlease enter the path to the image that will act as the background image for you to draw areas on."
            "\nThis can be relative to the current working directory"
            " or the default directory: '{}'".format(default_background_folder)
        )
        self.background_path = self._process_background_image_name(input())

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
            self.output_folder = self._process_output_folder(supplied_output_folder)
