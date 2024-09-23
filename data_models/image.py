
import numpy as np
import cv2


class Image:

    def __init__(self, path=None, array=None):
        if path is None and array is None:
            raise ValueError("Either 'path' or 'array' must be provided when initialising an image.")
        self.image_path = path
        self.image = cv2.imread(self.image_path) if path else array
        self.shape = self.image.shape
        self.rotated = False

    def from_path(self, path):
        self.__init__(path=path)

    def from_array(self, array):
        self.__init__(array=array)

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

    def write_to_video(self, writer):
        """
        Function Goal : write the image to a video so that it is one frame of the video

        writer : all the details to do with writing the images to the video

        return : None
        """
        writer.write(np.uint8(self.image * 255))

    def write_to_folder(self, folder_name):
        """
        Function Goal : write the image to a folder

        folder_name : string - the name of the folder that you want to write the images to

        return : None
        """
        cv2.imwrite(folder_name, self.image*255)
