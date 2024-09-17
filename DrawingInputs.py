
import argparse
import os.path

import cv2
import yaml

# read in YAML configuration file
with open("configs/drawing_configs.yaml", "r") as variables:
    config_variables = yaml.load(variables, Loader=yaml.FullLoader)

# get variables
default_background_folder = config_variables["default_background_folder"]
default_output_folder = config_variables["default_output_folder"]
default_base_width = config_variables["default_base_width"]


class DrawingInputs:

    def __init__(self):
        self.background_folder = default_background_folder
        self.background_name = ""
        self.output_folder = default_output_folder
        self.base_width = default_base_width


    @staticmethod
    def process_background_image_name(path_supplied):

        # check it's not empty
        if not path_supplied:
            print(
                "\nYou did not enter a valid background image name."
                " Please re-run this program ensuring the path to a background image is not empty."
            )
            exit(0)

        # check the path exists
        if os.path.exists(path_supplied):
            background_folder = os.path.dirname(path_supplied)
            background_name = os.path.basename(path_supplied)
        elif os.path.exists(os.path.join(default_background_folder, path_supplied)):
            background_folder = default_background_folder
            background_name = path_supplied
        else:
            print(
                "\nThe background image name entered does not point to a file that exists"
                " in the current working directory or the default directory."
                " Please re-run this program ensuring the path to the background image exists."
            )
            exit(0)

        # check the file is an image file
        try:
            cv2.imread(os.path.join(background_folder, background_name))
        except AttributeError:
            print(
                "\nThe background image name entered does not point to an valid image file."
                " Please re-run this program ensuring the path to the background image points to an image file.")
            exit(0)

        return background_folder, background_name

    @staticmethod
    def process_output_folder(supplied_output_folder):

        # check it's not empty
        if not supplied_output_folder:
            print(
                "\nYou did not enter a valid folder name."
                "Please re-run this program ensuring the path to the output folder is not empty."
            )
            exit(0)

        # check the path exists
        if not os.path.exists(supplied_output_folder):
            print(
                "\nThe folder path entered does not exist."
                "Please re-run this program ensuring the path to the output folder points to a valid folder."
            )
            exit(0)

        # check the path is a directory
        if not os.path.isdir(supplied_output_folder):
            print(
                "\nThe folder path entered points to a file and not a folder."
                "Please re-run this program ensuring the path to the output folder points to a valid folder."
            )
            exit(0)

        return supplied_output_folder


    @staticmethod
    def process_base_width(supplied_base_width):

        # check it's not empty
        if not supplied_base_width:
            print(
                "\nYou did not enter a valid width for the base of the background image."
                "Please re-run this program ensuring you enter a valid value for the base width of the background image.")
            exit(0)

        # check it's an integer
        try:
            base_width = int(supplied_base_width)
        except ValueError:
            print(
                "\nYou did not enter an integer for the width of the background image."
                "Please re-run this program ensuring you enter an integer for the base width of the background image."
            )
            exit(0)

        return base_width

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
            default=default_output_folder,
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
            "\nPress 'Enter' to set this to the default directory: '{}'".format(default_output_folder)
        )
        supplied_output_folder = input()
        if supplied_output_folder != "\n":
            self.output_folder = self.process_output_folder(supplied_output_folder)
        else:
            output_folder = default_output_folder
            print("\nThe output folder has been set to the default - '{}'.".format(default_output_folder))

        # base width
        print("\nPlease enter the width that the base of the background image to be scaled to.")
        supplied_base_width = input()
        if supplied_base_width != "\n":
            self.base_width = self.process_base_width(supplied_base_width)
        else:
            self.base_width = default_base_width
            print("\nThe width of image along the x-axis has been set to the default - {}.".format(default_base_width))
