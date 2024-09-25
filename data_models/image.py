
import numpy as np
import cv2
import os


class Image:

    def __init__(self, path=None, array=None):
        if path is None and array is None:
            raise ValueError("Either 'path' or 'array' must be provided when initialising an image.")
        self.image_path = path
        self.image = cv2.imread(self.image_path) if path else array
        self.shape = self.image.shape
        self.rotated = False

    @classmethod
    def from_path(cls, path):
        return cls(path=path)

    @classmethod
    def from_array(cls, array):
        return cls(array=array)

    def resize(self, desired_width, desired_height):
        """
        Function Goal : reshape the image to a desired width and height

        desired_width: integer - the width you want the image to be
        desired_height: integer - the height you want the image to be

        return : None
        """
        # get image orientation in line with desired width and height
        height, width, _ = self.image.shape
        if (desired_height <= desired_width) and not (height <= width):
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.rotated = True
        elif (desired_width <= desired_height) and not (width <= height):
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.rotated = True
        # scale to correct resolution
        self.image = cv2.resize(self.image, (desired_width, desired_height))
        self.shape = self.image.shape

    def write_to_video(self, writer, expected_shape):
        """
        Function Goal : write the image to a video so that it is one frame of the video

        writer : writer object - object that allows writing to a specific video
        expected_shape : tuple of integers (int, int, int) - expected image shape before writing

        return : None
        """
        # sort the shape
        if self.shape != expected_shape:
            # TODO: Fix if int(width * proportion) rounds the shape down so expected shape is 1 off
            height, width, depth = self.shape
            exp_height, exp_width, exp_depth = expected_shape
            if ((exp_height - height) <= 1) or ((exp_width - width) <= 1):
                self.resize(exp_width, exp_height)
                assert self.shape == expected_shape
            else:
                raise ValueError(f"Cannot write frame with shape '{self.shape}'. Expecting shape '{expected_shape}'")
        # sort the type of the image
        image = self.image if self.image.dtype == np.uint8 else np.uint8(self.image * 255)
        # write the image
        writer.write(image)

    def write_to_folder(self, folder_name, file_name):
        """
        Function Goal : write the image to a folder

        folder_name : string - the name of the folder that you want to write the images to

        return : None
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if not os.path.isdir(folder_name):
            raise ValueError("The supplied 'folder_name' is not a directory.")
        cv2.imwrite(os.path.join(folder_name, file_name), self.image * 255)
