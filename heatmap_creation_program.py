#!/usr/bin/env python

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import time
import cv2
from tqdm.auto import tqdm
import os

# import helper classes
from components.colourmap import ColourMap
from data_models.image import Image
from data_models.shape import Shape
from input_handlers.heatmap_inputs import HeatmapInputHandler
from input_output.video_reader import VideoReader

# import utilities
from utils.maths_utils import convert_cartesian_to_polar, convert_polar_to_cartesian
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
bg_area_configs = heatmap_configs["background_areas"]
event_box_configs = heatmap_configs["events_box"]
camera_configs = heatmap_configs["cameras"]


# lists for timing everything
define_heatmap_times = []
define_event_box_times = []
define_timer_times = []
central_merge_times = []
read_frame_times = []
define_bar_plot_times = []
side_merge_times = []
draw_arrows_times = []
loop_times = []


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
        # average duplicate values
        processed_df = df.groupby('Second').mean()
        # make second index a datetime object
        processed_df.index = pd.to_datetime(processed_df.index, unit="s")
        processed_dfs.append(processed_df)

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
    joined_df = pd.concat(list_of_dfs, axis=1, join="outer").sort_index()

    # fill the non-leading and non-trailing "Nan" values
    filled_df = joined_df.ffill(limit_area="inside")

    # resample the dataframe to one row for each second - avg values with more rows per second
    resampled_df = filled_df.resample('1s').mean()

    return resampled_df.ffill(limit_area="inside")


def uint_to_float(array, scale=False):
    if array.dtype == np.uint8:
        if scale:
            array = cv2.normalize(array, np.zeros(array.shape), 0, 1, cv2.NORM_MINMAX)
        else:
            array = array / 255.0
    return array


def create_area_masks(list_of_area_details, img_shape):
    """
    Function Goal : Iterate over the dictionaries, call the function "create_array_of_shapes" and put the created arrays and their centres in a list

    list_of_area_details : list of dictionaries - a list of dictionaries containing the details needed to identify the shapes and their coordinates on the image
    img_shape : tuple (int, int, int) - the size of the background image that the masks are to be drawn on

    return : list of shape objects
    """
    area_shapes = [Shape.from_dict(area_info_dict) for area_info_dict in list_of_area_details]
    for shape in area_shapes:
        shape.create_masks(img_shape, outline_thickness=bg_area_configs["outline_thickness"])
    return area_shapes


def add_colour_to_area_masks_and_merge(sensor_values, shape_objects, mapper):
    """
    Function Goal : Be given a row from the DataFrame which is a row of the sensor values for 1 frame worth of video for each differnt shape in the list of shapes
                    and to turn the mask for each shape the colour output when the sensor value for its respective csv is put into the mapper

    sensor_values : pd.Series - a row from the DataFrame which gives a sensor reading for each csv input
    shape_objects : list of shape objects - a list containing objects whereby the masks for each shape is accessible
    mapper : the heatmap mapper that maps a number to a colour

    return : list of shape objects - a list containing objects whereby the masks for each shape is accessible
    """
    default_colour = np.array(bg_area_configs["colour_when_nan"]) / 255
    outline_colour = heatmap_configs["borders"]["areas"]["colour"]
    coloured_shape_objs = []
    for val, shape in zip(sensor_values, shape_objects):
        area_colour = default_colour if np.isnan(val) else np.array(mapper.to_rgba(val)[:3][::-1])
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
        src1=np.where(shapes_canvas != empty, shapes_canvas, background_array),
        alpha=bg_area_configs["transparency_alpha"],
        src2=background_array,
        beta=1 - bg_area_configs["transparency_alpha"],
        gamma=bg_area_configs["transparency_gamma"],
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
        # TODO: write 'bg_area_configs["text_when_nan"]' in area value is NaN
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
        event_size = x_width * font_configs["event_box"]["proportions"]["size"]
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
        borderType=cv2_dict[border_configs["timer"]["type"]], value=border_configs["timer"]["colour"],
    )

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
    return uint_to_float(buf[:, :, ::-1][:, :, :3])


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
    # TODO: make bar plot bars the same colour as heatmap areas
    plt.bar(names, sensor_values, color=bar_colours)

    # turn the figure to an image array
    img = fig_to_img(fig)
    plt.close()

    # resize the image to the desired size
    border_width = int(final_width * border_configs["cameras"]["width_proportion"])
    x_width = final_width - (2 * border_width)
    y_height = final_height - (2 * border_width)
    image = cv2.resize(img, (x_width, y_height))

    # put border on the image
    bordered_image = cv2.copyMakeBorder(
        image, top=border_width, bottom=border_width, left=border_width, right=border_width,
        borderType=cv2_dict[border_configs["cameras"]["type"]], value=border_configs["cameras"]["colour"]
    )

    return bordered_image


def read_camera_frames(video_objects, second):
    """
    Function Goal : read in one frame from each video.

    video_objects : list of video reader objects - list of objects which allow us to read frames from each video
    second : integer - the second we want the frames to correspond to

    return : a List of 3D numpy arrays of images => [Array, array, ...] - list of the read-in frames
    """
    frames = []
    for video_obj in video_objects:
        try:
            frame = uint_to_float(video_obj.get_frame(video_obj.frame_rate * second))
        except ValueError:
            frame = np.zeros((1, 1, 3))
        frames.append(frame)
    return frames


def add_colour_and_text_if_empty(frame):

    # if the frame isn't empty, don't add text
    if not (frame == np.zeros((1, 1, 3))).all():
        return frame

    # add colour to frame
    coloured_frame = np.where(frame == np.zeros((1, 1, 3)), camera_configs["colour_when_finished"], frame) / 255

    # get text to write
    text = camera_configs["text_when_finished"]

    # define text variables
    height, width, _ = coloured_frame.shape
    frame_thickness = int(width * font_configs["cameras"]["proportions"]["thickness"])
    frame_size = width * font_configs["cameras"]["proportions"]["size"]
    frame_font = cv2_dict[font_configs["cameras"]["type"]]

    # define position to draw text
    text_width, text_height = cv2.getTextSize(text, frame_font, frame_size, thickness=frame_thickness)[0]
    start_y = int(height / 2 + text_height / 2)
    start_x = int(width / 2 - text_width / 2)

    # draw the text on the image
    cv2.putText(
        coloured_frame,
        text,
        (start_x, start_y),
        frame_font,
        frame_size,
        color=font_configs["cameras"]["colour"],
        lineType=cv2_dict[font_configs["cameras"]["line_type"]],
        thickness=frame_thickness,
    )

    return coloured_frame


def get_lhs_and_rhs_frames(video_frames, final_width, total_height):
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

    # decide which side each frame goes on
    num_videos_on_rhs = len(video_frames) - (len(video_frames) // 2 if video_frames else 0)
    lhs_frames = video_frames[:-num_videos_on_rhs]
    rhs_frames = video_frames[-num_videos_on_rhs:]

    # set the height to set the frames
    border_width = int(final_width * border_configs["cameras"]["width_proportion"])
    lhs_y_height = (total_height // (len(lhs_frames) + 1)) - (2 * border_width)
    rhs_y_height = (total_height // len(rhs_frames)) - (2 * border_width)

    # resize the images
    x_width = final_width - (2 * border_width)
    resized_lhs_frames = [cv2.resize(frame, (x_width, lhs_y_height)) for frame in lhs_frames]
    resized_rhs_frames = [cv2.resize(frame, (x_width, rhs_y_height)) for frame in rhs_frames]

    # add text to empty frames
    lhs_frames_with_txt = [add_colour_and_text_if_empty(frame) for frame in resized_lhs_frames]
    rhs_frames_with_txt = [add_colour_and_text_if_empty(frame) for frame in resized_rhs_frames]

    # add border to images
    def border_image(image, width, btype, colour):
        return cv2.copyMakeBorder(image, top=width, bottom=width, left=width, right=width, borderType=btype, value=colour)
    border_type = cv2_dict[border_configs["cameras"]["type"]]
    border_colour = border_configs["cameras"]["colour"]
    bordered_lhs_frames = [border_image(frame, border_width, border_type, border_colour) for frame in lhs_frames_with_txt]
    bordered_rhs_frames = [border_image(frame, border_width, border_type, border_colour) for frame in rhs_frames_with_txt]

    return bordered_lhs_frames, bordered_rhs_frames, bordered_lhs_frames[0].shape[0]


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

        else:
            raise ValueError(f"Encountered unexpected shape type: {shape_details["type"]}.")

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

        else:
            #  videos on the right hand side
            x = first_x + distance_between_first_nd_second
            y = ((2 * (i - num_videos_on_lhs)) + 1) * right_token_height

        # add the point to a list
        list_of_camera_image_midpoints.append((x, y))

    return list_of_camera_image_midpoints


def write_to_video(image, writer, expected_shape):
    """
    Function Goal : write the image to a video so that it is one frame of the video

    image : 3D np.array - array representing the RGB values of the image we want to write
    writer : writer object - object that allows writing to a specific video
    expected_shape : tuple of integers (int, int, int) - expected image shape before writing

    return : None
    """
    # sort the shape
    if image.shape != expected_shape:
        # TODO: Fix if int(width * proportion) rounds the shape down so expected shape is 1 off
        height, width, depth = image.shape
        exp_height, exp_width, exp_depth = expected_shape
        if ((exp_height - height) <= 1) or ((exp_width - width) <= 1):
            image = cv2.resize(image, (exp_width, exp_height))
            assert image.shape == expected_shape
        else:
            raise ValueError(f"Cannot write frame with shape '{image.shape}'. Expecting shape '{expected_shape}'")
    # sort the type of the image
    image = image if image.dtype == np.uint8 else np.uint8(image * 255)
    # write the image
    writer.write(image)


def write_to_folder(image, folder_name, file_name):
    """
    Function Goal : write the image to a folder

    image : 3D np.array - array representing the RGB values of the image we want to write
    folder_name : string - the name of the folder where the image file will be written
    file_name : string - the name of the file in the folder to write the image to

    return : None
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.isdir(folder_name):
        raise ValueError("The supplied 'folder_name' is not a directory.")
    cv2.imwrite(os.path.join(folder_name, file_name), image * 255)


def main():

    # get input variables
    inputs = HeatmapInputHandler()
    inputs.validate()
    background_image = inputs.background_image
    csv_file_paths = inputs.csv_file_paths
    camera_video_file_paths = inputs.video_file_paths
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
    background_image.image = uint_to_float(background_image.image)

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

    # create video reader object for reading CCTV videos
    camera_video_objects = [VideoReader(path) for path in camera_video_file_paths]

    # create the writer to write the image to the video
    writer = cv2.VideoWriter(
        filename=video_output_file_path,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=video_configs["frame_rate"],
        frameSize=(video_width, video_height),
        isColor=True,
    )

    # iterate through each row in the dataframe
    before_iteration_time = new_start_time = time.time()
    try:
        for i, (timestamp, sensor_vals) in tqdm(enumerate(joined_df.iterrows()), total=len(joined_df)):

            # define central heatmap image
            define_heatmap_start_time = time.time()
            coloured_shape_objects = add_colour_to_area_masks_and_merge(sensor_vals, shape_objects, cmap.mapper)
            background_with_areas = join_shapes_to_background(coloured_shape_objects, background_image.image)
            shape_centres = [shape.centre for shape in coloured_shape_objects]
            csv_names = [path[-6:-4] for path in csv_file_paths]
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
            bottom_component = np.concatenate((cmap.image, timer), axis=1)
            main_heatmap_component = np.concatenate((top_component, bottom_component), axis=0)
            central_merge_times.append(time.time() - central_merge_start_time)

            # read the camera video frames
            read_frame_start_time = time.time()
            camera_video_width = int(video_width * video_configs["proportions"]["width"]["cameras"])
            camera_frames = read_camera_frames(camera_video_objects, int(timestamp.timestamp()))
            lhs_cam_frames, rhs_cam_frames, lhs_cam_height = get_lhs_and_rhs_frames(camera_frames, camera_video_width, main_heatmap_component.shape[0])
            read_frame_times.append(time.time() - read_frame_start_time)

            # define bar plot
            define_bar_plot_start_time = time.time()
            area_colours = [shape.fill_colour for shape in coloured_shape_objects]
            bar_plot = create_bar_plot(sensor_vals, camera_video_width, lhs_cam_height, csv_names, area_colours)
            define_bar_plot_times.append(time.time() - define_bar_plot_start_time)

            # merge side components
            side_merge_start_time = time.time()
            lhs_component = np.concatenate((np.concatenate(lhs_cam_frames, axis=0), bar_plot), axis=0)
            rhs_component = np.concatenate(rhs_cam_frames, axis=0)
            all_components = np.concatenate((lhs_component, main_heatmap_component, rhs_component), axis=1)
            side_merge_times.append(time.time() - side_merge_start_time)

            """
            # draw arrows on the images joining the camera footage videos with their respective area on the heatmap
            draw_arrows_start_time = time.time()
            list_of_camera_image_midpoints = get_list_of_camera_image_midpoints(
                camera_video_width, shapes_nd_background_image.shape[1],
                num_videos_on_lhs, n_frames - num_videos_on_lhs, fully_merged_image.shape[0]
            )
            final_image = draw_arrows_from_cameras_to_shapes(
                fully_merged_image, list_of_shapes_details, list_of_camera_image_midpoints, camera_video_width, height_of_text_box
            )
            draw_arrows_times.append(time.time() - draw_arrows_start_time)
            """

            # write the images to the video
            # write_to_folder(all_components, "./vid_images", f"{str(int(timestamp.timestamp()))}.png")
            write_to_video(all_components, writer, expected_shape=(video_height, video_width, 3))

            loop_times.append(time.time() - new_start_time)
            new_start_time = time.time()

    except Exception as e:
        raise e

    finally:
        # release the camera video objects
        for obj in camera_video_objects:
            obj.release()
        # release the output video object
        writer.release()
        print("The video was written to the file with the name '" + video_output_file_path + "'.")
        # print timings
        print("---- BEFORE LOOPING ----")
        print("colourmap = {}".format(colmap_creation_time))
        print("joined_df = {}".format(joined_df_time))
        print("time before iteration = {}".format(before_iteration_time - start_time))
        print("---- IN LOOP ----")
        print("define heatmap = {}".format(sum(define_heatmap_times)))
        print("define event box = {}".format(sum(define_event_box_times)))
        print("define timer = {}".format(sum(define_timer_times)))
        print("merge central heatmap = {}".format(sum(central_merge_times)))
        print("read camera frames = {}".format(sum(read_frame_times)))
        print("define bar plot = {}".format(sum(define_bar_plot_times)))
        print("merge side heatmap = {}".format(sum(side_merge_times)))
        print("draw arrows = {}".format(sum(draw_arrows_times)))
        print("---- TOTAL ----")
        print("Avg loop time = {}".format(np.mean(loop_times)))
        print("Time taken = {}".format(time.time() - start_time))


if __name__ == '__main__':
    main()
