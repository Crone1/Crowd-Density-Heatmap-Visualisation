#!/usr/bin/env python

"""
This program needs the path to a YAML/JSON Configuraton file
It returns a file containing a list of dictionaries containing the details of the coordinates of the shapes drawn when this program is run.
"""

# imports
import numpy as np
import json
import yaml
import cv2
import sys
import os

from DrawingInputs import DrawingInputs

'''
# read in JSON configuration file
with open("configs/drawing_configs.json", "r") as variables:
    config_variables = json.load(variables)
'''

# read in YAML configuration file
with open("configs/drawing_configs.yaml", "r") as variables:
    config_variables = yaml.load(variables, Loader=yaml.FullLoader)

color_of_drawing = config_variables["color_of_drawing"]
mode = config_variables["default_mode"]
proportion_of_base_width_to_make_line_thickness = config_variables["proportion_of_basewidth_to_make_line_thickness"]

# variables needed to set up drawing
drawing = False
drawing_poly = False
start_x, start_y = -1, -1
shapes = []
poly_points = []
img_hist = []


def reshape_background_image(img_name, base_width):

    """
    Function Goal : reshape the image so that it still maintains the same proportion but that its width is the basewidth

    img_name : string - the name of the file containing the image you want to reshape
    base_width : integer - the width that you want the image to be
    
    return : 3D numpy array of integers - this array represents the image in its reshaped form in a way that python can deal with.
    """

    img = cv2.imread(img_name)

    height, width = img.shape[:2]

    '''if height < width:
        # change image from landscape to portrait
        img2 = np.zeros((width, height, 3), np.uint8)
        cv2.transpose(img, img2)
        cv2.flip(img2, 1, img2)

        # calculate the desired height of the image based on the proportion of the original image and the desired width
        width_percent = (base_width / float(img2.shape[1]))
        height_size = int(img2.shape[0] * width_percent)

        reshaped_img = cv2.resize(img2, (base_width, height_size))
        rotated = True

    else:'''

    width_percent = (base_width / float(img.shape[1]))
    height_size = int(img.shape[0] * width_percent)

    reshaped_img = cv2.resize(img, (base_width, height_size))

    return reshaped_img


def print_how_to_use_image_drawer():

    """
    Function Goal : This function contains a series of print statements that explain to the user how to use the program and draw the shapes on the image

    return : None

    keyboard shortcuts to control the program:
        "c": circle
        "p": polygon
        "r": rectangle
        "Escape": Exit program
        "Enter" (in poly mode): Finish polygon
        "Backspace": Undo
    """

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

    input_handler = DrawingInputs()

    if len(sys.argv) > 1:
        input_handler.get_variables_from_command_line()
    else:
        input_handler.get_variables_from_user()

    main(input_handler.background_folder, input_handler.background_name, input_handler.base_width, input_handler.output_folder)


def main(background_folder, background_name, base_width, output_folder):

    background_path = os.path.join(background_folder, background_name)
    background = reshape_background_image(background_path, base_width)
    line_thickness = int(proportion_of_base_width_to_make_line_thickness * base_width)

    print_how_to_use_image_drawer()
    shapes = draw_on_image(background, line_thickness)

    # output to file
    output_filename = os.path.splitext(os.path.basename(background_name))[0]
    output_path = os.path.join(output_folder, output_filename + ".json")
    with open(output_path, 'w') as file_of_shapes:
        string_of_shapes = json.dumps(shapes)
        file_of_shapes.write(string_of_shapes)

        print("\nThe shapes were written to the file under the name '" + str(output_path) + "'.")


if __name__ == '__main__':
    handle_inputs()
