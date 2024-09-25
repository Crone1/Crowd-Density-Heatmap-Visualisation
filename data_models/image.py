
import cv2


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
