
import argparse
import cv2
import json
import yaml
import os
import sys

from drawing_images_program import main as drawing_program

from image import Image

from utils.file_utils import add_extension, get_filename_no_extension, is_file_with_valid_extension
from utils.input_utils import exit_if_empty, exit_if_try_fails


# read the default configuration variables
with open("configs/default_configs.yaml", "r") as defaults_file:
    default_configs = yaml.load(defaults_file, Loader=yaml.FullLoader)

default_csv_folder = default_configs["default_csv_folder"]
default_video_folder = default_configs["default_video_folder"]
default_background_image = default_configs["default_background_image"]
default_output_file = default_configs["default_output_file"]
default_area_details_folder = default_configs["default_area_details_folder"]
default_events_file = default_configs["default_events_file"]
# TODO: remove base_width
default_base_width = default_configs["default_base_width"]
# TODO: remove below once automate colourmap creation
default_colourmap_image = default_configs["default_colourmap_image"]
default_colourmap_name = default_configs["default_colourmap_name"]


class HeatmapInputHandler:

    def __init__(self):
        # TODO: remove base_width
        self.base_width = default_base_width
        if len(sys.argv) > 1:
            self._get_variables_from_command_line()
        else:
            self._get_variables_from_user()

    @staticmethod
    def _get_file_paths(folder_path, file_ext):
        """
        Function Goal : Take a folder path and extract paths to the files with a specific file extension from it

        folder_path : string - the name of a folder full of csvs
        file_ext : string - the extension of the file type we are looking for within the supplied folder

        return : list of strings - list of the paths to the csv files in the folder
        """

        universal_criteria = "the path entered points to a folder containing '.{}' files.".format(file_ext)
        # check it's not empty
        exit_if_empty(folder_path, error="You did not enter a valid path to a folder.", criteria=universal_criteria)
        # check the path exists
        exit_if_empty(os.path.exists(folder_path), error="The folder path entered does not exist.", criteria=universal_criteria)
        # check the path is a directory
        exit_if_empty(os.path.isdir(folder_path), error="The folder path entered does not point to a folder.", criteria=universal_criteria)

        # extract the file paths from the folder
        file_paths = []
        for file_name in sorted(os.listdir(folder_path)):
            full_path = os.path.join(folder_path, file_name)
            if is_file_with_valid_extension(full_path, file_ext):
                file_paths.append(full_path)

        # check the folder contains some CSV files
        exit_if_empty(
            file_paths,
            error="The folder path entered points to a folder that doesn't contain any {} files.".format(file_ext),
            criteria=universal_criteria
        )

        return file_paths

    @staticmethod
    def _process_background_image(image_path):

        universal_criteria = "the path to the background image points to a valid image file."
        # check it's not empty
        exit_if_empty(image_path, error="You did not enter a valid path to a background image.", criteria=universal_criteria)
        # check the path exists
        exit_if_empty(os.path.exists(image_path), error="The file path entered does not exist.", criteria=universal_criteria)
        # check the path is a file
        exit_if_empty(os.path.isfile(image_path), error="The file path entered does not point to a file.", criteria=universal_criteria)
        # check the file is an image file
        exit_if_try_fails(
            cv2.imread,
            args=[image_path],
            exception=AttributeError,
            error="The file path entered does not point to a valid image file.",
            criteria=universal_criteria
        )
        # read the image
        return Image.from_path(image_path)

    @staticmethod
    def process_output_file_name(file_name):

        universal_criteria = "the name for the file the created video will be saved to is valid."
        # check it's not empty
        exit_if_empty(file_name, error="You did not enter a valid file name.", criteria=universal_criteria)

        return add_extension(file_name, "mp4")

    @staticmethod
    def _get_heatmap_area_details(file_path):

        universal_criteria = "the path to the file containing details of the heatmap areas points to a valid json file."
        # check it's not empty
        exit_if_empty(file_path, error="You did not enter a valid path to a json file.", criteria=universal_criteria)
        # check the path exists
        exit_if_empty(os.path.exists(file_path), error="The file path entered does not exist.", criteria=universal_criteria)
        # check the path is a file
        exit_if_empty(os.path.isfile(file_path), error="The file path entered does not point to a file.", criteria=universal_criteria)

        # check it's a json file
        def _load_json(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
        exit_if_try_fails(
            _load_json,
            args=[file_path],
            exception=(AttributeError, ValueError),
            error="The file path entered does not point to a valid json file.",
            criteria=universal_criteria
        )
        # check json file is not empty
        exit_if_empty(_load_json(file_path), error="The file path entered points to an empty json file.", criteria=universal_criteria)

        # extract area details from file
        return _load_json(file_path)

    @staticmethod
    def _get_event_details(file_path):

        universal_criteria = "the path to the file containing details of the events points to a valid text file."
        # check it's not empty
        exit_if_empty(file_path, error="You did not enter a valid path to a text file.", criteria=universal_criteria)
        # check the path exists
        exit_if_empty(os.path.exists(file_path), error="The file path entered does not exist.", criteria=universal_criteria)
        # check the path is a file
        exit_if_empty(os.path.isfile(file_path), error="The file path entered does not point to a file.", criteria=universal_criteria)

        # check it's a readable file
        def _read_file(path):
            with open(path, "r") as file:
                return file.readlines()
        exit_if_try_fails(
            _read_file,
            args=[file_path],
            exception=PermissionError,
            error="The file at the path entered can not be accessed.",
            criteria=universal_criteria
        )

        # check it's the correct format
        def _check_first_token_is_int(path):
            int(_read_file(path)[0].strip().split(" ")[0])
        exit_if_try_fails(
            _check_first_token_is_int,
            args=[file_path],
            exception=ValueError,
            error="The file at the path is not in the correct format.",
            criteria=universal_criteria
        )

        # read the events file
        dictionary_of_events = {}
        for event_line in _read_file(file_path):
            time, *tokens = event_line.strip().split(" ")
            dictionary_of_events[int(time)] = " ".join(tokens)

        return dictionary_of_events

    def _get_variables_from_command_line(self):
        """
        Function Goal: This function is used to read all the variables in from the command line arguments
        """

        parser = argparse.ArgumentParser(description="The variables that make this program work")

        # csv folder
        parser.add_argument(
            '-cf',
            dest="csv_folder",
            default=default_csv_folder,
            nargs="?",
            type=str,
            required=False,
            help="The path to the folder containing the csvs needed to colour the heatmap.",
        )
        # video folder
        parser.add_argument(
            '-vf',
            dest="video_folder",
            default=default_video_folder,
            nargs="?",
            type=str,
            required=False,
            help="The name of the folder containing the video footage taken from the cameras that the data in the csvs was taken from.",
        )
        # background image path
        parser.add_argument(
            '-bi',
            dest="background_image_path",
            default=default_background_image,
            nargs="?",
            type=str,
            required=False,
            help="The path to the background image that the heatmap will be drawn on.",
        )
        # output filename
        parser.add_argument(
            '-of',
            dest="output_file_name",
            default=default_output_file,
            nargs="?",
            type=str,
            required=False,
            help="The name of the file to output the final heatmap video to.",
            )
        # area details file path
        parser.add_argument(
            '-af',
            dest="area_details_file_path",
            default="default",
            nargs="?",
            type=str,
            required=False,
            help="The path to the file containing the data on the heatmap areas drawn on the image. Enter 'draw' to draw the areas now.",
        )
        # events file path
        parser.add_argument(
            '-ef',
            dest="events_file_path",
            default=default_events_file,
            nargs="?",
            type=str,
            required=False,
            help="The path to the file containing details of events which happen at various points of the video. Enter 'none' if there are no events."
        )

        args = parser.parse_args()

        # process data
        self.background_image = self._process_background_image(args.background_image_path)
        self.csv_file_paths = self._get_file_paths(args.csv_folder, "csv")
        self.video_file_paths = self._get_file_paths(args.video_folder, "mp4")
        self.background_image_path = self.process_background_image_path(args.background_image_path)
        self.output_file_name = self.process_output_file_name(args.output_file_name)
        if args.area_details_file_path == "draw":
            self.area_details = drawing_program(self.background_image_path, self.base_width, default_area_details_folder)
        elif args.area_details_file_path == "default":
            output_file_name = add_extension(get_filename_no_extension(self.background_image_path), "json")
            self.area_details = self._get_heatmap_area_details(os.path.join(default_area_details_folder, output_file_name))
        else:
            self.area_details = self._get_heatmap_area_details(args.area_details_file_path)
        if args.events_file_path != "none":
            self.event_details = self._get_event_details(args.events_file_path)

    def _get_variables_from_user(self):

        # background image
        supplied_background_image_path = input("Please enter the path to the background image: ")
        self.background_image = self._process_background_image(supplied_background_image_path)

        # csv folder
        print("\nPlease enter the path to the folder containing the csvs needed to colour the heatmap.")
        supplied_csv_folder = input()
        self.csv_file_paths = self._get_file_paths(supplied_csv_folder, "csv")

        # videos folder
        print("\nPlease enter the name of the folder containing the video footage taken from the cameras that the data in the csvs was taken from.")
        supplied_videos_folder = input()
        self.video_file_paths = self._get_file_paths(supplied_videos_folder, "mp4")

        # background image
        print("\nPlease enter the path to the background image that the heatmap will be drawn on.")
        supplied_background_image_path = input()
        self.background_image_path = self.process_background_image_path(supplied_background_image_path)

        print("\nTo set the following variables to default values: press 'Enter'.")

        # video output file name
        print("\nPlease enter the name of the file to output the final heatmap video to.")
        supplied_output_file_name = input()
        if supplied_output_file_name != "\n":
            self.output_file_name = self.process_output_file_name(supplied_output_file_name)
        else:
            self.output_file_name = default_output_file
            print("\nThe name of the file the final heatmap video will be output to has been set to the default - '{}'.".format(default_output_file))

        # area details
        print(
            "\nPlease enter the path to the file containing the data on the heatmap areas drawn on the image."
            " Enter 'draw' to draw the areas now."
        )
        supplied_area_details_file_path = input()
        if supplied_area_details_file_path == "\n":
            output_file_name = add_extension(get_filename_no_extension(self.background_image_path), "json")
            supplied_area_details_file_path = os.path.join(default_area_details_folder, output_file_name)
            print("\nThe path to the file containing the data on the heatmap areas drawn on the image has been set to the default - '{}'.".format(supplied_area_details_file_path))
        if supplied_area_details_file_path == "draw":
            self.area_details = drawing_program(self.background_image_path, self.base_width, default_area_details_folder)
        else:
            self.area_details = self._get_heatmap_area_details(supplied_area_details_file_path)

        # events file
        print(
            "\nPlease enter the path to the file containing details of events which happen at various points of the video."
            "Enter 'none' if there are no events."
        )
        supplied_events_file_path = input()
        if supplied_events_file_path == "\n":
            supplied_events_file_path = default_events_file
            print("\nThe path to the file containing details of events which happen at various points of the video has been set to the default - '{}'.".format(default_events_file))
        if supplied_events_file_path != "none":
            self.event_details = self._get_event_details(supplied_events_file_path)

        # Let the user know the inputs have all been received
        print("\nOK, Making the video now. Please wait!")
