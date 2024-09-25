
import argparse
import cv2
import json
import yaml
import os
import sys
import pandas as pd

from drawing_images_program import main as drawing_program

from data_models.image import Image

from utils.file_utils import add_extension, is_file_with_valid_extension
from utils.input_utils import exit_if_false, exit_if_try_fails


# read the default configuration variables
with open("configs/default_configs.yaml", "r") as defaults_file:
    default_configs = yaml.load(defaults_file, Loader=yaml.FullLoader)
default_drawing_output_file = default_configs["drawing"]["output_file_path"]
default_video_output_file = default_configs["heatmap"]["output_file_path"]


class HeatmapInputHandler:

    def __init__(self):
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

        universal_criteria = "the path entered points to a folder containing readable '.{}' files.".format(file_ext)
        # check it's not empty
        exit_if_false(folder_path, error="You did not enter a valid path to a folder.", criteria=universal_criteria)
        # check the path exists
        exit_if_false(os.path.exists(folder_path), error="The folder path entered does not exist.", criteria=universal_criteria)
        # check the path is a directory
        exit_if_false(os.path.isdir(folder_path), error="The folder path entered does not point to a folder.", criteria=universal_criteria)

        # extract the file paths from the folder
        file_paths = []
        for file_name in sorted(os.listdir(folder_path)):
            full_path = os.path.join(folder_path, file_name)
            if is_file_with_valid_extension(full_path, file_ext):
                file_paths.append(full_path)

        # check the folder contains some CSV files
        exit_if_false(
            file_paths,
            error="The folder path entered points to a folder that doesn't contain any {} files.".format(file_ext),
            criteria=universal_criteria
        )

        # check the files are readable
        func = {"csv": pd.read_csv, "mp4": cv2.VideoCapture}[file_ext]
        for fpath in file_paths:
            exit_if_try_fails(
                func,
                args=[fpath],
                exception=PermissionError,
                error="The {} file at path '{}' can not be accessed.".format(file_ext, fpath),
                criteria=universal_criteria,
            )

        return file_paths

    @staticmethod
    def _process_background_image(image_path):

        universal_criteria = "the path to the background image points to a valid image file."
        # check it's not empty
        exit_if_false(image_path, error="You did not enter a valid path to a background image.", criteria=universal_criteria)
        # check the path exists
        exit_if_false(os.path.exists(image_path), error="The file path entered does not exist.", criteria=universal_criteria)
        # check the path is a file
        exit_if_false(os.path.isfile(image_path), error="The file path entered does not point to a file.", criteria=universal_criteria)
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
    def _process_output_file_name(file_name):

        universal_criteria = "the name for the file the created video will be saved to is valid."
        # check it's not empty
        exit_if_false(file_name, error="You did not enter a valid file name.", criteria=universal_criteria)

        return add_extension(file_name, "mp4")

    @staticmethod
    def _get_heatmap_area_details(file_path):

        universal_criteria = "the path to the file containing details of the heatmap areas points to a valid json file."
        # check it's not empty
        exit_if_false(file_path, error="You did not enter a valid path to a json file.", criteria=universal_criteria)
        # check the path exists
        exit_if_false(os.path.exists(file_path), error="The file path entered does not exist.", criteria=universal_criteria)
        # check the path is a file
        exit_if_false(os.path.isfile(file_path), error="The file path entered does not point to a file.", criteria=universal_criteria)

        # check it's a json file
        def _load_json(path):
            with open(path, "r") as file:
                return json.load(file)
        exit_if_try_fails(
            _load_json,
            args=[file_path],
            exception=(AttributeError, ValueError),
            error="The file path entered does not point to a valid json file.",
            criteria=universal_criteria
        )
        # check json file is not empty
        exit_if_false(_load_json(file_path), error="The file path entered points to an empty json file.", criteria=universal_criteria)
        # TODO: check coordinates align with the expected coordinates in the output video

        # extract area details from file
        return _load_json(file_path)

    @staticmethod
    def _get_event_details(file_path):

        universal_criteria = "the path to the file containing details of the events points to a valid text file."
        # check it's not empty
        exit_if_false(file_path, error="You did not enter a valid path to a text file.", criteria=universal_criteria)
        # check the path exists
        exit_if_false(os.path.exists(file_path), error="The file path entered does not exist.", criteria=universal_criteria)
        # check the path is a file
        exit_if_false(os.path.isfile(file_path), error="The file path entered does not point to a file.", criteria=universal_criteria)

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

        parser = argparse.ArgumentParser(description="Create a heatmap video of the changes in value across different areas drawn onto a background image.")

        # background image path
        parser.add_argument(
            '-bi',
            dest="background_image_path",
            nargs="?",
            type=str,
            required=True,
            help="The path to the background image.",
        )
        # csv folder
        parser.add_argument(
            '-cf',
            dest="csv_folder_path",
            nargs="?",
            type=str,
            required=True,
            help="The path to the folder containing the CSV data used to colour the heatmap.",
        )
        # output video path
        parser.add_argument(
            '-of',
            dest="video_output_file_path",
            default=default_video_output_file,
            nargs="?",
            type=str,
            required=False,
            help="The path to the file where the heatmap video will be output to.",
            )
        # area details file path
        parser.add_argument(
            '-af',
            dest="area_details_file_path",
            default="draw",
            nargs="?",
            type=str,
            required=False,
            help="The path to the file containing details of the heatmap areas.",
        )
        # events file path
        parser.add_argument(
            '-ef',
            dest="events_file_path",
            default="none",
            nargs="?",
            type=str,
            required=False,
            help="The path to the file containing details of events which happen during the video."
        )
        # video folder
        parser.add_argument(
            '-vf',
            dest="video_folder_path",
            default="none",
            nargs="?",
            type=str,
            required=False,
            help="The path of the folder containing the video footage which accompanies the CSV data.",
        )

        args = parser.parse_args()

        # process data
        self.background_image = self._process_background_image(args.background_image_path)
        self.csv_file_paths = self._get_file_paths(args.csv_folder_path, "csv")
        self.video_output_file_path = self._process_output_file_name(args.video_output_file_path)
        if args.area_details_file_path == "draw":
            self.area_details = drawing_program(self.background_image, default_drawing_output_file)
            print(f"We have output the details of these drawn areas to '{default_drawing_output_file}' so you don't have to draw them again next time!")
        else:
            self.area_details = self._get_heatmap_area_details(args.area_details_file_path)
        if args.events_file_path != "none":
            self.event_details = self._get_event_details(args.events_file_path)
        if args.video_folder_path != "none":
            self.video_file_paths = self._get_file_paths(args.video_folder_path, "mp4")

    def _get_variables_from_user(self):

        # background image
        supplied_background_image_path = input("Please enter the path to the background image: ")
        self.background_image = self._process_background_image(supplied_background_image_path)

        # csv folder
        supplied_csv_folder_path = input("Please enter the path to the folder containing the CSV data used to colour the heatmap: ")
        self.csv_file_paths = self._get_file_paths(supplied_csv_folder_path, "csv")

        # video output file name
        video_output_file_path = input("Please enter the path to the file where the heatmap video will be output to (Press 'Enter' for default): ")
        if video_output_file_path == "":
            self.video_output_file_path = default_video_output_file
            print("\nThe final heatmap video will be output to the default file - '{}'.".format(default_video_output_file))
        else:
            self.video_output_file_path = self._process_output_file_name(video_output_file_path)

        # area details
        supplied_area_details_file_path = input("Please enter the path to the file containing details of the heatmap areas (Press 'Enter' to draw them): ")
        if supplied_area_details_file_path == "draw":
            self.area_details = drawing_program(self.background_image, default_drawing_output_file)
            print(f"We have output the details of these drawn areas to '{default_drawing_output_file}' so you don't have to draw them again next time!")
        else:
            self.area_details = self._get_heatmap_area_details(supplied_area_details_file_path)

        # events file
        supplied_events_file_path = input("Please enter the path to the file containing details of events which happen during the video (Press 'Enter' to skip): ")
        if supplied_events_file_path != "":
            self.event_details = self._get_event_details(supplied_events_file_path)

        # videos folder
        supplied_videos_folder_path = input("Please enter the path of the folder containing the video footage which accompanies the CSV data (Press 'Enter' to skip): ")
        if supplied_videos_folder_path != "":
            self.video_file_paths = self._get_file_paths(supplied_videos_folder_path, "mp4")

        # Let the user know the inputs have all been received
        print("\nThanks for the inputs! Making the video now, please wait!")

    def validate(self):
        # check number of CSVs == number of areas
        exit_if_false(
            len(self.csv_file_paths) == len(self.area_details),
            error="The number of areas you drew and the number of csvs you supplied do not match.",
            criteria="to draw the same amount of areas on the image as csvs are in the supplied folder."
        )
        # check number of videos == number of CSVs
        exit_if_false(
            len(self.video_file_paths) == len(self.csv_file_paths),
            error="The number of videos in the folder supplied does not match the number of csvs supplied.",
            criteria="the number of videos in the supplied folder is the same as the number of csvs in the supplied folder."
        )
