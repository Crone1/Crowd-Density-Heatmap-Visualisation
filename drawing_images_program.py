#!/usr/bin/env python

"""
This program needs the path to a YAML/JSON Configuraton file
It returns a file containing a list of dictionaries containing the details of the coordinates of the shapes drawn when this program is run.
"""

# imports
import numpy as np
import argparse
import json
import yaml
import cv2
import sys
import os

'''
# read in JSON configuration file
with open("Configuration files/Configurate drawing variables.json", "r") as variables:
    config_variables = json.load(variables)
'''

# read in YAML configuration file
with open("Configuration files/Configurate drawing variables.yaml", "r") as variables:
    config_variables = yaml.load(variables, Loader=yaml.FullLoader)

# get variables
default_output_filename = config_variables["default_output_filename"]
default_basewidth = config_variables["default_basewidth"]

color_of_drawing = config_variables["color_of_drawing"]
mode = config_variables["default_mode"]
proportion_of_basewidth_to_make_line_thickness = config_variables["proportion_of_basewidth_to_make_line_thickness"]

# variables needed to set up drawing
drawing = False
drawing_poly = False
start_x, start_y = -1, -1
shapes = []
poly_points = []
img_hist = []


def touch_up_file_name(name):

    """
    Function Goal : add ".json" to the end of the output file name if it isn't already there

    name : string - name of the output file

    return : string - name of the output video ending in ".json"
    """

    if name[-5:] != ".json":
        return name + ".json"

    else:
        return name


def get_variables_from_command_line():

    parser = argparse.ArgumentParser(description="The variables that make this programme work")

    # number of csvs
    parser.add_argument(dest="num_csv", nargs="?", type=int,
                        help="The number of areas to draw using this program - This value must match the number of csv files in the folder of csv files you supply to the visualisation program.")

    # background image name
    parser.add_argument(dest="background_image_name", nargs="?", type=str,
                        help="The path to the image that will act as the background image that you will draw the different areas on.")

    # output filename
    parser.add_argument('-of', dest="output_filename", default=default_output_filename, nargs="?", type=str, required=False,
                        help="The name of/path to the file that you want the data on the shapes you drew to be output to.")

    # basewidth
    parser.add_argument('-bw', dest="basewidth", default=default_basewidth, nargs="?", type=int, required=False,
                        help="The width that you want the base of the background image to be scaled to.")

    args = parser.parse_args()

    # background
    if args.background_image_name:
        try:
            if cv2.imread(args.background_image_name).all() == None:
                pass

        except AttributeError:
            print("\nThe name entered does not corrospond to a valid name of an image in this directory, please run this program again with a valid path to an image.")
            exit(0)

    else:
        print("\nYou did not enter a path to a background image. Please re-run this program and add a valid path to a background image as a command line argument.")
        exit(0)

    if not args.basewidth:
        print("\nYou did not enter an integer for the width of the base of the background image. Please re-run this program and fix the command line arguments to include this integer.")
        exit(0)

    # output filename
    if args.output_filename:
        output_filename = touch_up_file_name(args.output_filename)

    else:
        print("\nYou did not enter the name of a file for the shapes to be output to. Please re-run this program and fix the command line arguments to include this.")
        exit(0)

    return args.num_csv, args.background_image_name, args.basewidth, output_filename


def get_variables_from_user():

    # csv files
    print("\nHello, How many areas do you want to draw using this program?\n - This value must match the number of csv files in the folder of csv files you supply to the visualisation program.")
    read_in = input()
    try:
        num_csv = int(read_in)

    except ValueError:
        print("\nThis value must be an integer, please run this programme again entering a valid number of csvs.")
        exit(0)

    # background image
    print("\nPlease enter the name of the background image you would like to highlight the camera areas on.")
    background_image_name = input()
    if background_image_name:
        try:
            if cv2.imread(background_image_name).all() == None:
                pass

        except AttributeError:
            print("\nThe name entered does not corrospond to a valid name of an image in this directory, please run this program again with a valid path to an image.")
            exit(0)

    else:
        print("\nYou did not enter a path to a background image. Please re-run this program and ensure to enter a valid path to a background image.")
        exit(0)


    print("\nTo set the following variables to default values: press 'Enter'.")

    # filename
    print("\nPlease enter what name you would like the output file to be called.")
    x = input()
    if x:
        output_filename = touch_up_file_name(x)

    else:
        output_filename = default_output_filename
        print("\nThe output filename has been set to the default - '{}'.".format(default_output_filename))

    # basewidth
    print("\nPlease enter the number that you would like the width of the base of the background image to be when scaled.")
    read_in = input()
    if read_in:
        try:
            basewidth = int(read_in)

        except ValueError:
            print("\nThis value must be an integer, please run this programme again with a valid width.")
            exit(0)

    else:
        basewidth = default_basewidth
        print("\nThe width of image along the x-axis has been set to the default - {}.".format(default_basewidth))

    return num_csv, background_image_name, basewidth, output_filename


def reshape_background_image(img_name, basewidth):

    """
    Function Goal : reshape the image so that it still maintains the same proportion but that its width is the basewidth

    img_name : string - the name of the file containing the image you want to reshape
    basewidth : integer - the width that you want the image to be
    
    return : 3D numpy array of integers - this array represents the image in its reshaped form in a way that python can deal with.
    """

    img = cv2.imread(img_name)

    height, width = img.shape[:2]

    '''if height < width:
        # change image from landscape to portrait
        img2 = np.zeros((width, height, 3), np.uint8)
        cv2.transpose(img, img2)
        cv2.flip(img2, 1, img2)

        # calculate the desired hight of the image based on the proportion of the original image and the desired width
        width_percent = (basewidth / float(img2.shape[1]))
        height_size = int(img2.shape[0] * width_percent)


        reshaped_img = cv2.resize(img2, (basewidth, height_size))
        rotated = True

    else:'''

    width_percent = (basewidth / float(img.shape[1]))
    height_size = int(img.shape[0] * width_percent)

    reshaped_img = cv2.resize(img, (basewidth, height_size))

    return reshaped_img


def print_how_to_use_image_drawer(num_csv):

    """
    Function Goal : This function contains a series of print statements that explain to the user how to use the program and draw the shapes on the image

    num_csv : integer - the number of csvs that were given to the program

    return : None

    keyboard shortcuts to control the program:
        "c": circle
        "p": polygon
        "r": rectangle
        "Escape": Exit program
        "Enter" (in poly mode): Finish polygon
        "Backspace": Undo
    """

    print("\nOnly draw one shape per area that the sensor covers.\n")
    if num_csv == 1:
        print("You only supplied one csv so please only draw 1 area.\n")

    else:
        print("Please draw {} areas.".format(num_csv, num_csv))
        print("Please draw the areas in the order that corrosponds to the order the csvs are in the folder you will supply to the visualisation progam.\n")

    print(" To:\t\t\t\t Press:")
    print(" Draw rectangle:\t\t 'r'\n Draw multi cornered polygon:\t 'p'\n Draw circle:\t\t\t 'c'")
    print(" Finish drawing polygon:\t 'Enter'")
    print(" Undo the last drawn shape:\t 'Backspace'\n Finished drawing:\t\t 'Esc'\n")

    if mode == "rect":
        print("The default mode is rectangle.")

    if mode == "circle":
        print("The default mode is circle.")

    if mode == "poly":
        print("The default mode is polygon.")


def keyboard_callbacks(key, line_thickness):

    global mode, drawing, poly_points, tmp_img

    # Modes
    if key == ord("c"):
        mode = "circle"

    elif key == ord("r"):
        mode = "rect"

    elif key == ord("p"):
        mode = "poly"

    elif key == 13:  # Enter
        # Save polygon
        if mode == "poly":
            drawing = False

            # Update image
            tmp_img = np.copy(img_hist[-1])
            cv2.polylines(tmp_img, np.int32([poly_points]), True,
                          color_of_drawing, line_thickness)
            img_hist.append(np.copy(tmp_img))

            # Save parameters
            shapes.append({
                "type": "poly",
                "points": poly_points,
            })

            poly_points = []

    elif key == 8:  # Backspace
        # Undo
        if mode == "poly" and drawing:
            if len(poly_points) > 0:
                poly_points.pop()

        elif len(img_hist) > 1:
            img_hist.pop()
            shapes.pop()
            tmp_img = np.copy(img_hist[-1])


def dist_between_2_points(point1_x, point1_y, point2_x, point2_y):

    """
    Function Goal : get the distance between 2 points

    point1_x : integer - the x coordinate for point 1
    point1_y : integer - the y coordinate for point 1
    point2_x : integer - the x coordinate for point 2
    point2_y : integer - the y coordinate for point 2

    return : intager - the distance between point1 and point2
    """

    return np.hypot(point1_x - point2_x, point1_y - point2_y)


def mouse_callbacks(event, x, y, flags, param, line_thickness):

    global start_x, start_y, drawing, mode, img, tmp_img, color_of_drawing, poly_points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

        if mode == "poly":
            poly_points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            tmp_img = np.copy(img_hist[-1])

            # Draw temporary shape that follows the cursor
            if mode == "rect":
                cv2.rectangle(tmp_img, (start_x, start_y),
                              (x, y), color_of_drawing, line_thickness)

            elif mode == "circle":
                cv2.circle(tmp_img, (start_x, start_y), int(
                    dist_between_2_points(start_x, start_y, x, y)), color_of_drawing, line_thickness)

            elif mode == "poly":
                cv2.polylines(tmp_img, np.int32(
                    [poly_points + [(x, y)]]), True, color_of_drawing, line_thickness)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # Finish drawing shape
        if mode == "rect" and (start_x != x or start_y != y):
            cv2.rectangle(tmp_img, (start_x, start_y), (x, y), color_of_drawing, line_thickness)
            # save shape as json
            shapes.append({
                "type": "rectangle",
                "start": (start_x, start_y),
                "end": (x, y),
            })

        elif mode == "circle":
            radius = int(dist_between_2_points(start_x, start_y, x, y))
            cv2.circle(tmp_img, (start_x, start_y), radius, color_of_drawing, line_thickness)
            shapes.append({
                "type": "circle",
                "centre": (start_x, start_y),
                "radius": radius,
            })

        elif mode == "poly":
            # ie dont cancel drawing
            drawing = True
            return

        img_hist.append(np.copy(tmp_img))


def draw_on_image(image, thickness):

    global img, tmp_img

    img = image
    img_hist.append(img)
    tmp_img = np.copy(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', lambda event, x, y, flags, param: mouse_callbacks(event, x, y, flags, param, line_thickness=thickness))

    while True:
        cv2.imshow('image', tmp_img if drawing else img_hist[-1])

        # key press handling
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Escape
            break

        else:
            keyboard_callbacks(key, thickness)

    cv2.destroyAllWindows()

    return shapes

def handle_inputs():

    """
    Function Goal : call the functions that input the variables that are needed to make the program work and pass these variables to the main() function

    return : None
    """


    if len(sys.argv) > 1:
        num_csv, background_image_name, basewidth, output_filename = get_variables_from_command_line()

    else:
        num_csv, background_image_name, basewidth, output_filename = get_variables_from_user()

    main(num_csv, background_image_name, basewidth, output_filename)


def main(num_csv, background_image_name, basewidth, output_filename=""):

    background = reshape_background_image(background_image_name, basewidth)
    line_thickness = int(proportion_of_basewidth_to_make_line_thickness * basewidth)

    print_how_to_use_image_drawer(num_csv)
    shapes = draw_on_image(background, line_thickness)

    if len(shapes) == num_csv:
        if output_filename:
            with open(output_filename, 'w') as file_of_shapes:
                string_of_shapes = json.dumps(shapes)
                file_of_shapes.write(string_of_shapes)

            print("\nThe shapes were written to the file under the name '" + str(output_filename) + "'.")

        else:
            return shapes

    else:
        print("\nYou supplied {} csvs but drew {} shapes. Please re-run the programme drawing the same amount of areas on the image as csvs you supply.".format(num_csv, len(shapes)))
        exit(0)


if __name__ == '__main__':
    handle_inputs()
