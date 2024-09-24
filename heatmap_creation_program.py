#!/usr/bin/env python

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import time
import cv2

# import helper classes
from components.colourmap import ColourMap
from data_models.image import Image
from data_models.shape import Shape
from input_handlers.heatmap_inputs import HeatmapInputHandler
from input_output.video_reader import VideoReaderQueue

# import utilities
from utils.maths_utils import get_slope, get_equation_of_line, get_distance, get_ratio_interval_point, convert_cartesian_to_polar, convert_polar_to_cartesian
from configs.cv2_config import cv2_dict


# read configurations
with open("configs/video_resolutions.yaml", "r") as resolution_file:
    resolution_configs = yaml.load(resolution_file, Loader=yaml.FullLoader)
with open("configs/default_configs.yaml", "r") as default_config_file:
    default_configs = yaml.load(default_config_file, Loader=yaml.FullLoader)
data_configs = default_configs["data"]
video_configs = default_configs["video"]


# read the heatmap customisation configuration variables
with open("configs/heatmap_configs.yaml", "r") as heatmap_config_file:
    heatmap_configs = yaml.load(heatmap_config_file, Loader=yaml.FullLoader)
border_configs = heatmap_configs["borders"]
font_configs = heatmap_configs["fonts"]
arrow_configs = heatmap_configs["arrows"]
background_configs = heatmap_configs["background"]
event_box_configs = heatmap_configs["events_box"]


# lists for timing everything
define_heatmap_times = []
define_event_box_times = []
define_timer_times = []
central_merge_times = []
video_list = []
resize_list = []
concat_outside_list = []
bar_list = []
video_2_frames_list = []
merge_list = []
slice_list = []
concat_inside_list = []
border_list = []
read_in_list = []
merge_heatmap = []
colmap_creation = []


def create_area_masks(list_of_area_details, img_shape):
    """
    Function Goal : Iterate over the dictionaries, call the function "create_array_of_shapes" and put the created arrays and their centres in a list

    list_of_area_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    img_shape : tuple (int, int, int) - the size of the background image that the masks are to be drawn on

    return : list of shape objects
    """
    area_shapes = [Shape.from_dict(area_info_dict) for area_info_dict in list_of_area_details]
    for shape in area_shapes:
        shape.create_masks(img_shape, outline_thickness=background_configs["outline_thickness"])
    return area_shapes


def read_csvs_into_dataframes(csv_file_paths):
    """
    Function Goal : Read each csv into a DataFrame with 2 columns, Second and Sensor value, and add the DataFrame to a list

    csv_file_paths : list of strings - a list of paths to the csvs input

    return : list of DataFrames
    """
    # read the data
    raw_dfs = [pd.read_csv(csv_path, names=data_configs["columns"]) for csv_path in csv_file_paths]

    # process the data
    processed_dfs = []
    for raw_df in raw_dfs:
        # turn minute to second
        raw_df["Second"] = raw_df["Minute"].astype(float) * 60
        df = raw_df.drop(columns=["Minute"])
        # set second as index
        processed_dfs.append(df.sort_values("Second").set_index("Second"))

    return processed_dfs


def process_csv_dataframes(list_of_dfs):
    """
    Function Goal : Take the list of DataFrames and join them together column wise on their second column
                    Then populate the values where a particular column doesn't contain any value for sensor value for this Second
                    Populate these values by just using the value for the previous Second

    list_of_dfs : list of DataFrames - a list containing DataFrames with 2 columns, Second and sensor value

    return : DataFrame - a DataFrame containing a Second column with one Second per row and then with columns of the sensor values
                         from each csv at that second
                         (Second, df1 sensor value, df2 Crowd density, .... ect.)
    """
    if len(list_of_dfs) == 1:
        return list_of_dfs[0]

    # join the dataframes together
    joined_df = pd.concat(list_of_dfs, axis=0, join="outer").sort_index()

    # fill the non-leading and non-trailing "Nan" values
    filled_df = joined_df.ffill(limit_area="inside")

    # resample the dataframe to one row for each second - avg values with more rows per second
    filled_df.index = pd.to_datetime(filled_df.index, unit="s")
    resampled_df = filled_df.resample('1s').mean()

    return resampled_df.ffill(limit_area="inside")


def add_colour_to_area_masks_and_merge(sensor_values, shape_objects, mapper):
    """
    Function Goal : Be given a row from the DataFrame which is a row of the sensor values for 1 frame worth of video for each differnt shape in the list of shapes
                    and to turn the mask for each shape the colour output when the sensor value for its respective csv is put into the mapper

    sensor_values : pd.Series - a row from the DataFrame which gives a sensor reading for each csv input
    shape_objects : list of shape objects - a list containing objects whereby the masks for each shape is accessible
    mapper : the heatmap mapper that maps a number to a colour

    return : list of shape objects - a list containing objects whereby the masks for each shape is accessible
    """
    default_colour = np.reshape(background_configs["colour_when_nan"], (1, 1, 3)) / 255
    outline_colour = heatmap_configs["borders"]["areas"]["colour"]
    coloured_shape_objs = []
    for val, shape in zip(sensor_values, shape_objects):
        area_colour = default_colour if np.isnan(val) else mapper.to_rgba(val)[:3][::-1]
        shape.change_colour(fill_colour=area_colour, outline_colour=outline_colour)
        shape.create_merged_mask()
        coloured_shape_objs.append(shape)
    return coloured_shape_objs


def join_shapes_to_background(shape_objects, background_array):

    # join shapes together
    shapes_canvas = np.zeros(background_array.shape)
    empty = [0, 0, 0]
    for shape in shape_objects:
        # if canvas is blank, fill this value with the shape
        shapes_canvas = np.where(
            (shape.merged_mask != empty) & (shapes_canvas == empty),
            shape.merged_mask,
            shapes_canvas
        )
        # if the canvas is filled, fill with the mean values
        shapes_canvas = np.where(
            (shape.merged_mask != empty) & (shapes_canvas != empty),
            np.mean([shape.merged_mask, shapes_canvas], axis=0),
            shapes_canvas
        )

    # overlay the joined shapes onto the background image
    background_with_areas = cv2.addWeighted(
        src1=np.where(shapes_canvas != empty, shapes_canvas, background_array).astype(np.uint8),
        alpha=background_configs["transparency_alpha"],
        src2=background_array,
        beta=1 - background_configs["transparency_alpha"],
        gamma=background_configs["transparency_gamma"],
    )

    return background_with_areas


def label_areas_on_background(background_with_areas, list_of_area_centres, names):
    """
    Function Goal : Take a list of arrays corresponding to the images of the areas on the background and merge these arrays so that this array corresponds to one image
                    of all the different areas and then add this image to the background to create one image

    background_with_areas : 3D numpy array of integers - array of the background image with the shapes overlaid
    list_of_area_centres : a list of tuples of integers [(int, int), (int, int), ...] -  list of the centre points of each area
    names : list of strings [str, str, ...] - list of the names of the areas

    return : 3D numpy array of integers - this array corresponds to the image with each area labeled
    """
    # define text variables
    label_font = cv2_dict[font_configs["areas"]["type"]]
    label_size = background_with_areas.shape[1] * font_configs["areas"]["proportions"]["size"]
    label_thickness = int(background_with_areas.shape[1] * font_configs["areas"]["proportions"]["thickness"])

    # draw the labels on each area
    for name, (centre_x, centre_y) in zip(names, list_of_area_centres):

        text_width, text_height = cv2.getTextSize(name, label_font, label_size, thickness=label_thickness)[0]

        # define where to start drawing the text on the image
        start_x_coord = int(centre_x - text_width/2)
        start_y_coord = int(centre_y + text_height/2)

        # put the text on the image
        cv2.putText(
            background_with_areas,
            name,
            (start_x_coord, start_y_coord),
            label_font,
            label_size,
            color=font_configs["areas"]["colour"],
            lineType=cv2_dict[font_configs["areas"]["line_type"]],
            thickness=label_thickness,
        )

    return background_with_areas


def create_event_text_box(second, events_dict, final_width, final_height, event_duration):
    """
    Function Goal : Create the event text box for the top of the visualisation

    second : integer - the second that the particular frame is produced at
    events_dict : dictionary of integer to string {integer : string, integer : string, ... etc.} - This is a dictionary of different integers representing particular
                                                                                                          seconds in the video mapped to an event that happend at that
                                                                                                          second. The string contains the text to be displayed in the text
                                                                                                          box at the top of the image.
    final_width : integer - the width of the text box along the x-axis
    final_height : integer - the height of the text box on the y-axis
    event_duration : integer - the number of frames either side of the event to display the text for that event

    return : 3D numpy array of integers - an array corresponding to the text box containing the text about the event
    """

    # define blank text box
    border_width = int(final_width * border_configs["event_box"]["width_proportion"])
    x_width = final_width - (2 * border_width)
    y_height = final_height - (2 * border_width)
    text_box = np.ones((y_height, x_width, 3))

    # get text for event box
    potential_seconds = list(range(second - event_duration, second + 1))
    found_seconds = [sec for sec in potential_seconds if sec in events_dict]
    if found_seconds:
        sec_to_display = max(found_seconds)
        text = events_dict[sec_to_display]

        # define text variables
        event_thickness = int(x_width * font_configs["event_box"]["proportions"]["thickness"])
        event_size = int(x_width * font_configs["event_box"]["proportions"]["size"])
        event_font = cv2_dict[font_configs["event_box"]["type"]]

        # define position to draw text
        text_width, text_height = cv2.getTextSize(text, event_font, event_size, thickness=event_thickness)[0]
        while text_width > x_width:
            print("WARNING: Event name associated with second '{}' is too long for text box. Will be truncated.".format(sec_to_display))
            # TODO: Add newline instead of truncating
            text = text[:len(text) // 2]
            text_width, text_height = cv2.getTextSize(text, event_font, event_size, thickness=event_thickness)[0]
        start_y_coord = int(y_height/2 + text_height/2)
        start_x_coord = int(x_width/2 - text_width/2)

        # draw the text on the image
        cv2.putText(
            text_box,
            text,
            (start_x_coord, start_y_coord),
            event_font,
            event_size,
            color=font_configs["event_box"]["colour"],
            lineType=cv2_dict[font_configs["event_box"]["line_type"]],
            thickness=event_thickness,
        )

    # put border on the text box
    bordered_text_box = cv2.copyMakeBorder(
        text_box, top=border_width, bottom=border_width, left=border_width, right=border_width,
        borderType=cv2_dict[border_configs["event_box"]["type"]], value=border_configs["event_box"]["colour"],
    )

    return bordered_text_box


def create_timer(second, final_width, final_height):
    """
    Function Goal : Take an integer second and create an array corresponding to an image that is a particular width and height that contains the second fed in

    second : integer - the second that the particular frame is produced at
    final_width : integer - the width along the x-axis to make the array
    final_height : integer - the height along the y-axis to make the array

    return : a 3D numpy array of integers - this array corresponds to the image of a particular width and height that contains the integer second given
    """

    # define blank timer box
    border_width = int(final_width * border_configs["timer"]["width_proportion"])
    x_width = final_width - (2 * border_width)
    y_height = final_height - (2 * border_width)
    timer = np.ones((y_height, x_width, 3))

    # get text for timer
    text = '%05d' % second

    # define text variables
    timer_thickness = int(x_width * font_configs["timer"]["proportions"]["thickness"])
    timer_size = x_width * font_configs["timer"]["proportions"]["size"]
    timer_font = cv2_dict[font_configs["timer"]["type"]]

    # define position to draw text
    text_width, text_height = cv2.getTextSize(text, timer_font, timer_size, thickness=timer_thickness)[0]
    start_y = int(y_height/2 + text_height/2)
    start_x = int(x_width/2 - text_width/2)

    # draw the text on the image
    cv2.putText(
        timer,
        text,
        (start_x, start_y),
        timer_font,
        timer_size,
        color=font_configs["timer"]["colour"],
        lineType=cv2_dict[font_configs["timer"]["line_type"]],
        thickness=timer_thickness,
    )

    # put border on the image
    bordered_timer = cv2.copyMakeBorder(
        timer, top=border_width, bottom=border_width, left=border_width, right=border_width,
        borderType=cv2_dict[border_configs["timer"]["type"]], value=border_configs["timer"]["colour"])

    return bordered_timer


def fig_to_img(fig):
    """
    Function Goal : Turn a matplotlib figure into a BGRA image

    adapted from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure

    fig : matplotlib figure

    return : 3D numpy array of integers - the image drawn from the matplotlib figure
    """

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)

    # ARGB -> BGR
    return buf[:, :, ::-1][:, :, :3]/255


def create_bar_plot(sensor_values, final_width, final_height, names, bar_colours):
    """
    Function Goal : take a row of sensor values and make a bar plot from these integers

    sensor_values : pd.Series of integers - a row from a DataFrame containing sensor values used to colour the bar plot
    final_width : integer - the width on the x-axis to make the bar plot
    final_height : integer - the height on the y-axis to make the bar plot
    names : list of strings [str, str, ...etc.] - a list containing the names of the cameras to put on the bar plot
    bar_colours: list of 3x1 numpy arrays of integers - these arrays represent the RGB colour for each bar

    return : an 3D numpy array of integers - an array corresponding to an image of the bar plot
    """

    # create bar plot figure
    fig = plt.figure()
    plt.subplot(
        title="The {} in the different areas.".format(data_configs["title"]),
        xlabel="Areas",
        ylim=(data_configs["min_value"], data_configs["max_value"]),
        ylabel=data_configs["title"],
    )
    plt.bar(names, sensor_values, color=bar_colours)

    # turn the figure to an image array
    img = fig_to_img(fig)
    plt.close()

    # resize the image to the desired size
    border_width = int(final_width * border_configs["bar_plot"]["proportions"]["width"])
    x_width = final_width - (2 * border_width)
    y_height = final_height - (2 * border_width)
    image = cv2.resize(img, (x_width, y_height))

    # put border on the image
    bordered_image = cv2.copyMakeBorder(
        image, top=border_width, bottom=border_width, left=border_width, right=border_width,
        borderType=cv2_dict[border_configs["bar_plot"]["type"]], value=border_configs["bar_plot"]["colour"]
    )

    return bordered_image


def generate_all_points_on_outside_of_shape(corners):
    """
    Function Goal : To take the corners of the shape drawn and to use these to generate all the points that are on the outside permimeter of the shape drawn

    corners : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of the points on the corners of the shape that was drawn

    return : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of points that surround the whole area
    """

    points = []
    start_x, start_y = corners[0]
    for x, y in corners[1:] + corners[:1]:

        if start_x == x:
            # line goes along x-axis (horizontal)

            for new_y in range(min(start_y, y), max(start_y, y)):
                points.append((x, new_y))

        elif start_y == y:
            # line goes along the y-axis (vertical)

            for new_x in range(min(start_x, x), max(start_x, x)):
                points.append((new_x, y))

        else:
            # line is slanted diagonally either upwards from left to right or downwards from left to right
            slope = get_slope((start_x, start_y), (x, y))

            length_on_x_axis = max(start_x, x) - min(start_x, x)

            for i in range(arrow_configs["points_on_line"]):

                x_val = min(start_x, x) + (i * length_on_x_axis) / arrow_configs["points_on_line"]

                new_y = (slope * x_val) - (slope * start_x) + start_y

                points.append((int(x_val), int(new_y)))

        start_x = x
        start_y = y

    return points


def get_closest_point(corners, camera_point, centre):
    """
    Function Goal : get a list of all the points on the outside of shape and find the closest point on the outside to the point given

    corners : list of tuples of interest [(int, int), (int, int), ...etc.] - a list of the corner points of the rectangle/polygon
    point : tuple of integers (int, int) - a point

    return : the closest point on the outside of the shape to the point given
    """

    closest = ""
    closest_dist_to_line = np.inf
    list_of_points_on_outside = generate_all_points_on_outside_of_shape(corners)

    cam_point = int(camera_point[0]), int(camera_point[1])
    centre_point = int(centre[0]), int(centre[1])

    coefficients_of_x_nd_y, constant = get_equation_of_line(centre_point, cam_point)

    for point in list_of_points_on_outside:

        if (centre[0] < camera_point[0] and centre[0] < point[0] < camera_point[0]) or (centre[0] > camera_point[0] and centre[0] > point[0] > camera_point[0]):

            intercept_of_points = np.array(coefficients_of_x_nd_y).dot(np.array([point[0], point[1]]))

            if np.abs(constant - intercept_of_points) < closest_dist_to_line:
                closest = point
                closest_dist_to_line = np.abs(constant - intercept_of_points)

    return closest


def draw_arrows_from_cameras_to_shapes(image, list_of_shapes_details, list_of_camera_image_midpoints, width_to_move, height_to_move):
    """
    Function Goal : Take an array corresponding to the image, change some values so that when it is turned to an image, arrows are drawn between the shapes on the image
                    and the boxes containing the camera footage

    image : 3D numpy array of integers - the array that corresponds to one frame of the video
    list_of_shapes_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    list_of_camera_image_midpoints : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of points. These points are the coordinates of the midpoints of the edges of the boxes
                                     containing camera footage
    width_to_move : integer - the value to add to the x value of the points on the shapes to account for the fact I added in the camera footage on the LHS and RHS
    height_to_move : integer - the value to add to the y value of the points on the shapes to account for the fact I added in the colourmap and text box on the
                               top and bottom of the image

    return : 3D numpy array of integers - the array that corresponds to one frame of the video that includes the arrows draw on the image
    """

    for i in range(len(list_of_camera_image_midpoints)):
        shape_details = list_of_shapes_details[i]

        if shape_details["type"] == "rectangle":
            start = tuple(shape_details["start"])
            end = tuple(shape_details["end"])

            moved_start = (start[0] + width_to_move, start[1] + height_to_move)
            moved_end = (end[0] + width_to_move, end[1] + height_to_move)

            moved_centre = ((moved_start[0] + moved_end[0])/2, (moved_start[1] + moved_end[1])/2)

            start_x_nd_end_y = (moved_start[0], moved_end[1])
            end_x_nd_start_y = (moved_end[0], moved_start[1])

            moved_corners = [moved_start, start_x_nd_end_y, moved_end, end_x_nd_start_y]

            closest = get_closest_point(moved_corners, list_of_camera_image_midpoints[i], moved_centre)

        elif shape_details["type"] == "poly":
            corners = shape_details["points"]

            moved_corners = []
            for corner in corners:
                moved_corners.append((corner[0] + width_to_move, corner[1] + height_to_move))

            moved_centre = np.mean(pd.DataFrame(moved_corners), axis=0).astype(int)

            closest = get_closest_point(moved_corners, list_of_camera_image_midpoints[i], moved_centre)

        elif shape_details["type"] == "circle":
            centre = tuple(shape_details["centre"])
            radius = shape_details["radius"]

            moved_centre = (centre[0] + width_to_move, centre[1] + height_to_move)

            distance = get_distance(moved_centre, list_of_camera_image_midpoints[i])

            closest = get_ratio_interval_point(moved_centre, list_of_camera_image_midpoints[i], radius, distance-radius)

        # draw line for the arrow between the edge of the area to the camera footage frame
        cv2.line(image, list_of_camera_image_midpoints[i], closest, arrow_configs["line_colour"], thickness=arrow_configs["line_thickness"],
                 lineType=arrow_configs["line_type"])

        # calculate the angle that the arrow head needs to be
        point_relative_to_point_on_shape = (list_of_camera_image_midpoints[i][0] - closest[0], list_of_camera_image_midpoints[i][1] - closest[1])

        rho, pi = convert_cartesian_to_polar(point_relative_to_point_on_shape)

        new_angles = [pi - arrow_configs["head_angle"], pi + arrow_configs["head_angle"]]

        # draw the lines for the arrow head
        for angle in new_angles:
            x, y = convert_polar_to_cartesian(arrow_configs["head_length"], angle)

            cv2.line(image, (closest[0] + x, closest[1] + y), closest, arrow_configs["line_colour"],
                     thickness=arrow_configs["line_thickness"], lineType=arrow_configs["line_type"])

    return image


def get_list_of_camera_image_midpoints(first_x, distance_between_first_nd_second, num_videos_on_lhs, num_videos_on_rhs, total_height):
    """
    Function goal : create a list of points which correspond to the coordinates of the midpoints of the edges of the boxes containing camera footage

    first_x : integer - the distance along the x-axis between the most left point and the edge of the first set of camera footage videos
    distance_between_first_nd_second : integer - the distance along the x-axis between the edge of the first set of camera footage videos and the start of the second
                                                 set of camera footage video
    num_videos_on_rhs : integer - the number of camera footage videos on the right hand side of the middle heatmap image
    total_height : integer - the total height along the y-axis of the whole video

    return : list of tuples of integers [(int, int), (int, int), ...etc.] - a list of points. These points are the coordinates of the midpoints of the edges of the boxes
                                                                            containing camera footage
    """

    left_token_height = int(total_height/(2 * (num_videos_on_lhs + 1)))
    right_token_height = int(total_height/(2 * num_videos_on_rhs))

    list_of_camera_image_midpoints = []
    for i in range(num_videos_on_lhs + num_videos_on_rhs):
        if i < num_videos_on_lhs:
            # videos on the left hand side
            x = first_x
            y = ((2 * i) + 1) * left_token_height

        elif i >= num_videos_on_lhs and i < num_videos_on_lhs + num_videos_on_rhs:
            #  videos on the right hand side
            x = first_x + distance_between_first_nd_second
            y = ((2 * (i - num_videos_on_lhs)) + 1) * right_token_height

        # add the point to a list
        list_of_camera_image_midpoints.append((x, y))

    return list_of_camera_image_midpoints


def merge_lhs_and_rhs_frames(frames, num_videos_on_lhs, bar_plot_image):
    """
    Function goal : Take the list of the arrays corresponding to a frame from the different videos and separate these into the images that will go on the LHS and the RHS,
                    then merge all the LHS and RHS images, put a border around them and then add the bar plot to the LHS image.

    frames : A list of 3D numpy arrays [Array, Array, etc...] - A list of arrays of the different images from the different videos that corresponds to 1 frame of video
                                                                The length of this list is the number of videos that will feature on the frame
    num_videos_on_lhs : integer - the number of images on the left hand side of the main heatmap image
    bar_plot_image : 3D numpy array of integers - this array represents the image of the bar plot

    return : 3D numpy arrays of integers => Array, Array - the left array is an array corresponding to the image that will go to the left of the main heatmap image
                                                           the right array is an array corresponding to the image that will go to the right of the main heatmap image
    """

    start_slice = time.time()
    lhs_list = frames[:num_videos_on_lhs]
    rhs_list = frames[num_videos_on_lhs:]
    slice_list.append(time.time() - start_slice)

    start_concat_inside = time.time()
    rhs_img = np.concatenate(rhs_list)
    lhs_img = np.concatenate(lhs_list)
    concat_inside_list.append(time.time() - start_concat_inside)

    start_border = time.time()
    bordered_rhs_image = cv2.copyMakeBorder(rhs_img, top=border_configs["cameras"]["width"], bottom=border_configs["cameras"]["width"], left=border_configs["cameras"]["width"],
                                            right=border_configs["cameras"]["width"], borderType=cv2_dict[border_configs["cameras"]["type"]], value=border_configs["cameras"]["colour"])
    bordered_lhs_image = cv2.copyMakeBorder(lhs_img, top=border_configs["bar_plot"]["width"], bottom=border_configs["bar_plot"]["width"], left=border_configs["bar_plot"]["width"],
                                            right=border_configs["bar_plot"]["width"], borderType=cv2_dict[border_configs["bar_plot"]["type"]], value=border_configs["bar_plot"]["colour"])
    border_list.append(time.time() - start_border)

    try:
        bordered_lhs_image = np.concatenate((bordered_lhs_image, bar_plot_image))

    except ValueError:
        print("Error, line 1393")

    return bordered_lhs_image, bordered_rhs_image


def video_to_frames(read_videos, width, total_height):
    """
    Function Goal : read in one frame from the videos and resize them to a specific width and all to the same height.

    read_videos : list of video read ins - a list of the variables for each different video to read in the next frame from
    width : integer - the width to resize the images to

    return : a List of 3D numpy arrays of images and an integer => [Array, array, etc.], integer - a list of the resized frames and the height of each frame
    """

    height_of_frame = int(total_height/int((len(read_videos)/2)))

    # read in the images and resize them
    resized_frames = []
    for i in range(len(read_videos)):
        # for each video

        #start_read = time.time()

        try:
            start_read = time.time()
            success, image = read_videos[i].read()

            if success == False:
                raise StopIteration

            #image = next(read_videos[i])

            read_in_list.append(time.time() - start_read)

        except StopIteration:
            image = np.zeros((1, 1, 3))

        img = cv2.resize(image, (width, height_of_frame))/255

        if np.array_equal(image, np.zeros((1, 1, 3))):

            text_when_video_finishes = "No Video"

            text_width, text_height = cv2.getTextSize(text_when_video_finishes, cv2_dict[font_configs["cameras"]["type"]], font_configs["cameras"]["size"],
                                                      font_configs["cameras"]["thickness"])[0]

            height = int(total_height/2 - text_height/2)
            start_x = int(width/2 - text_width/2)

            cv2.putText(img, text_when_video_finishes, (start_x, height), cv2_dict[font_configs["cameras"]["type"]], font_configs["cameras"]["size"],
                        font_configs["cameras"]["colour"], lineType=cv2_dict[font_configs["cameras"]["line_type"]], thickness=font_configs["cameras"]["thickness"])

        resized_frames.append(img)

    return resized_frames, height_of_frame


def turn_all_the_different_images_into_one_image(main_heatmap_component, df_of_row,
                                                 width_of_left_and_right_images, names, read_videos,
                                                 num_videos_on_lhs, list_of_shapes_details, list_of_colours):
    """
    Function Goal : Get the images of the background, the shapes, the colourmap, the frame number, the camera footage videos and the bar plot image and merge these images
                    together to form one singular image

    list_of_coloured_shapes : a list 3D numpy arrays of integers - the list containing the coloured masks of the shapes drawn on the background
    background : 3D numpy Arrays of integers - an array that corresponds to the background image of the shapes
    colourmap_image : 3D numpy array of integers - an array that corresponds to the image of the colourmap
    df_of_row : DataFrame of integers - this is a single row from a DataFrame that contains a second and the sensor values corresponding to that particular second
                                        in the larger DataFrame
    timer_width : the width along the x-axis to make the array
    timer_height : the height along the y-axis to make the array
    width_of_left_and_right_images : the width along the x-axis to make the camera footage videos on either side of the main heatmap image
    dictionary_of_events : dictionary of integer to string {integer : string, integer : string, ... etc.} - This is a dictionary of different integers representing particular
                                                                                                          seconds in the video mapped to an event that happend at that
                                                                                                          second. The string contains the text to be displayed in the text
                                                                                                          box at the top of the image.
    event_duration_frame : integer - the number of frames either side of the event to display the text for that event
    names : list of strings [str, str, ...etc.] - a list containing the names of the cameras to put on the bar plot
    list_of_area_centres : a list of tuples of integers [(int, int), (int, int), ...etc.] -  a list containing the centre points of each of the shapes
    read_videos : list of the camera footage videos read ins - a list of the variables for each different video to read in the next frame from
    num_videos_on_lhs : integer - the number of camera footage videos on the left hand side of the middle heatmap image
    list_of-shapes_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    height_of_text_box : the height of the text box on the y-axis

    return : 3D numpy array of integers - an array corresponding to the image which is made up of all the different images to be featured in the video merged together
    """


    start_video = time.time()

    # if there are camera footage videos to put on the sides of the heatmap video
    if read_videos:

        start_video_2_frames = time.time()

        # get a list of 1 frame from each video and reshape them
        frames, height_of_frame = video_to_frames(read_videos, width_of_left_and_right_images, main_heatmap_component.shape[0])

        video_2_frames_list.append(time.time() - start_video_2_frames)

        start_bar = time.time()

        # create bar plot
        bar_plot_image = create_bar_plot(sensor_values, camera_video_width, height_of_frame, names, list_of_colours)

        bar_list.append(time.time() - start_bar)

        start_merge = time.time()

        # crate the left and right images from the frames of the videos
        lhs_img, rhs_img = merge_lhs_and_rhs_frames(frames, num_videos_on_lhs, bar_plot_image)

        merge_list.append(time.time() - start_merge)

        start_resize = time.time()

        # resize the left and right images
        lhs_img = cv2.resize(lhs_img, (lhs_img.shape[1], main_heatmap_component.shape[0]))
        rhs_img = cv2.resize(rhs_img, (rhs_img.shape[1], main_heatmap_component.shape[0]))

        resize_list.append(time.time() - start_resize)

        start_concat = time.time()

        # merge the lhs images, the main heatmap image and the rhs images
        fully_merged_image = np.concatenate((lhs_img, main_heatmap_component, rhs_img), axis=1)

        # draw arrows on the images joining the camera footage videos with their respective area on the heatmap
        list_of_camera_image_midpoints = get_list_of_camera_image_midpoints(
            width_of_left_and_right_images, shapes_nd_background_image.shape[1],
            num_videos_on_lhs, len(frames) - num_videos_on_lhs, fully_merged_image.shape[0]
        )

        final_image = draw_arrows_from_cameras_to_shapes(
            fully_merged_image, list_of_shapes_details, list_of_camera_image_midpoints, width_of_left_and_right_images, height_of_text_box
        )

        concat_outside_list.append(time.time() - start_concat)

    else:
        # create bar plot
        bar_plot_image = create_bar_plot(sensor_values, camera_video_width, main_heatmap_component.shape[0] - (2 * border_configs["bar_plot"]["width"]), names, list_of_colours)

        # merge the main heatmap image and the bar plot
        final_image = np.concatenate((bar_plot_image, main_heatmap_component), axis=1)

    video_list.append(time.time() - start_video)

    return final_image


def main():

    # get input variables
    inputs = HeatmapInputHandler()
    inputs.validate()
    background_image = inputs.background_image
    csv_file_paths = inputs.csv_file_paths
    video_file_paths = inputs.video_file_paths
    area_details = inputs.area_details
    event_details = inputs.event_details
    video_output_file_path = inputs.video_output_file_path

    start_time = time.time()

    # resize background image and add border
    video_width, video_height = resolution_configs[default_configs["video"]["resolution"]]
    background_image.resize(
        int(default_configs["video"]["proportions"]["width"]["background"] * video_width),
        int(default_configs["video"]["proportions"]["height"]["background"] * video_height),
    )

    # dissect the area details
    shape_objects = create_area_masks(area_details, background_image.shape)

    # turn the CSV data into a dataframe
    start_joined_df_time = time.time()
    list_of_dfs = read_csvs_into_dataframes(csv_file_paths)
    joined_df = process_csv_dataframes(list_of_dfs)
    joined_df_time = time.time() - start_joined_df_time

    # create the colourmap image
    start_colmap_time = time.time()
    colourmap_width = int(video_width*video_configs["proportions"]["width"]["colourmap"])
    colourmap_height = int(video_height * video_configs["proportions"]["height"]["colourmap"])
    cmap = ColourMap(colourmap_height, colourmap_width)
    cmap.create()
    colmap_creation_time = time.time() - start_colmap_time

    # create the writer to write the image to the video
    writer = cv2.VideoWriter(
        filename=video_output_file_path,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=video_configs["frame_rate"],
        frameSize=(video_width, video_height),
        isColor=True,
    )

    if not video_output_file_path:
        read_videos = []
        num_of_images_on_lhs = 0
    else:
        # find out how many images will be on the lhs
        num_of_images_on_lhs = len(video_file_paths) // 2
        # create video caps
        read_videos = []
        for vid in video_file_paths:
            #v = VideoReaderQueue(vid, queue_size=32)
            #img, num_of_frames = v.load_video()
            img = cv2.VideoCapture(vid)
            read_videos.append(img)

    # width of camera images beside heatmap
    width_of_left_and_right_images = int(base_width * video_proportion_configs["width"]["cameras"])

    # get the height of the video
    if event_details:
        height_of_video = (background.shape[0] + (2 * border_configs["background"]["width"])) + border_colmap.shape[0] + (height_of_text_box + (2 * border_configs["event_box"]["width"]))
    else:
        height_of_video = (background.shape[0] + (2 * border_configs["background"]["width"])) + border_colmap.shape[0]
    # get the width of the video
    if read_videos:
        width_of_video = base_width + (2 * border_configs["background"]["width"]) + (2 * width_of_left_and_right_images) + (2 * border_configs["cameras"]["width"]) + (2 * border_configs["bar_plot"]["width"])
    else:
        width_of_video = base_width + (2 * border_configs["background"]["width"]) + width_of_left_and_right_images  + (2 * border_configs["bar_plot"]["width"])


    # iterate through each row in the dataframe
    print("time before iteration through rows = {}".format(time.time() - start_time))
    new_start_time = time.time()
    for i, (timestamp, sensor_vals) in enumerate(joined_df.iterrows()):
        print("Row {}, {:.3g}% Done".format(str(i), (100 * i/len(joined_df))))

        # define central heatmap image
        define_heatmap_start_time = time.time()
        coloured_shape_objects = add_colour_to_area_masks_and_merge(sensor_vals, shape_objects, cmap.mapper)
        background_with_areas = join_shapes_to_background(coloured_shape_objects, background_image.image)
        shape_centres = [shape.centre for shape in coloured_shape_objects]
        csv_names = [os.path.basename(path)[:-3] for path in csv_file_paths]
        heatmap = label_areas_on_background(background_with_areas, shape_centres, csv_names)
        define_heatmap_times.append(time.time() - define_heatmap_start_time)

        # define event text box
        define_event_box_start_time = time.time()
        event_box_height = int(video_height * video_configs["proportions"]["height"]["events_box"])
        event_duration = int(video_configs["frame_rate"] * event_box_configs["text_duration"])
        event_box = create_event_text_box(int(timestamp.timestamp()), event_details, heatmap.shape[1], event_box_height, event_duration)
        define_event_box_times.append(time.time() - define_event_box_start_time)

        # define timer
        define_timer_start_time = time.time()
        timer_width = int(video_width * video_configs["proportions"]["width"]["timer"])
        timer = create_timer(int(timestamp.timestamp()), timer_width, colourmap_height)
        define_timer_times.append(time.time() - define_timer_start_time)

        # merge central heatmap components
        central_merge_start_time = time.time()
        top_component = np.concatenate((event_box, heatmap), axis=0)
        # merge colourmap and timer
        bottom_component = np.concatenate((cmap.image, timer), axis=1)
        # merge colormap & timer with the heatmap & event box
        main_heatmap_component = np.concatenate((top_component, bottom_component), axis=0)
        central_merge_times.append(time.time() - central_merge_start_time)

        # merge these images with the image of the colourmap and of the second
        start_turn_images_to_one = time.time()
        merged_image = turn_all_the_different_images_into_one_image(
            main_heatmap_component, df_row, cctv_width, csv_names, cctv_video_list, num_videos_on_lhs,
        )
        turn_images_to_one_times.append(time.time() - start_turn_images_to_one)

        # write the images to the video
        final_frame = Image.from_array(merged_image)
        final_frame.write_to_video(writer)

        print(time.time() - new_start_time)
        new_start_time = time.time()

    writer.release()
    print("The video was written to the file with the name '" + video_output_file_path + "'.")

    print("---- BEFORE LOOPING ----")
    print("colourmap = {}".format(colmap_creation_time))
    print("joined_df = {}".format(joined_df_time))
    print("read_videos_time = {}".format(read_videos_time))
    print("---- IN LOOP ----")
    print("define heatmap = {}".format(sum(define_heatmap_times)))
    print("define event box = {}".format(sum(define_event_box_times)))
    print("define timer = {}".format(sum(define_timer_times)))
    print("merge central heatmap = {}".format(sum(central_merge_times)))

    print("video stuff = {}".format(sum(video_list)))
    print("video_2_frames = {}".format(sum(video_2_frames_list)))
    print("read in videos = {}".format(sum(read_in_list)))
    print("merge = {}".format(sum(merge_list)))
    print("concat outside = {}".format(sum(concat_outside_list)))
    print("resize = {}".format(sum(resize_list)))
    print("bar = {}".format(sum(bar_list)))
    print("concat inside = {}".format(sum(concat_inside_list)))
    print("slice = {}".format(sum(slice_list)))
    print("border = {}".format(sum(border_list)))

    print("Time taken = {}".format(time.time() - start_time))


if __name__ == '__main__':
    main()
