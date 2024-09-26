
# import libraries
import cv2
import numpy as np

# import utilities
from utils.maths_utils import generate_points_on_shape_boundary, get_closest_point_between_points, \
    get_distance_to_point, get_ratio_interval_point


class Shape:
    def __init__(self, shape_type):
        self.type = shape_type
        self.centre = None
        self.filled_mask = None
        self.fill_colour = None
        self.outline_mask = None
        self.outline_colour = None
        self.merged_mask = None

    @staticmethod
    def from_dict(data):
        shape_type = data.get("type").lower()
        if shape_type == "rectangle":
            return Rectangle(data["start_point"], data["end_point"])
        elif shape_type == "circle":
            return Circle(data["centre"], data["radius"])
        elif shape_type == "polygon":
            return Polygon(data["points"])
        else:
            raise ValueError(f"Unknown shape type: {shape_type}. Valid types are 'polygon', 'circle', and 'rectangle'.")

    def _calculate_centre(self):
        raise NotImplementedError("Subclasses must implement this method")

    def create_masks(self, img_size, filled_val, outline_val, outline_thickness):
        raise NotImplementedError("Subclasses must implement this method")

    def adjust(self, x_offset, y_offset):
        raise NotImplementedError("Subclasses must implement this method")

    def get_closest_point(self, point):
        raise NotImplementedError("Subclasses must implement this method")

    def change_colour(self, fill_colour, outline_colour):
        if (self.filled_mask is None) or (self.outline_mask is None):
            raise ValueError("Cannot colour masks when they are empty. Please run 'create_masks()' first.")
        self.filled_mask = np.where(self.filled_mask == self.fill_colour, fill_colour, self.filled_mask)
        self.outline_mask = np.where(self.outline_mask == self.outline_colour, outline_colour, self.outline_mask)
        self.fill_colour = fill_colour
        self.outline_colour = outline_colour

    def create_merged_mask(self):
        if (self.filled_mask is None) or (self.outline_mask is None):
            raise ValueError("Cannot merge masks when masks are empty. Please run 'create_masks()' first.")
        self.merged_mask = np.where(self.outline_mask == self.outline_colour, self.outline_mask, self.filled_mask)


class Rectangle(Shape):
    def __init__(self, start_point, end_point):
        super().__init__('rectangle')
        self.start_point = start_point
        self.end_point = end_point
        self.centre = self._calculate_centre()

    def _calculate_centre(self):
        x_centre = (self.start_point[0] + self.end_point[0]) / 2
        y_centre = (self.start_point[1] + self.end_point[1]) / 2
        return (x_centre, y_centre)

    def create_masks(self, img_size, fill_colour=[1, 1, 1], outline_colour=[2, 2, 2], outline_thickness=1):
        # warn about merge issue
        if fill_colour == outline_colour:
            print("WARNING: The shape filling colour and outline colour are the same. May cause issues when merging.")
        # filled mask
        filled_canvas = np.zeros(img_size)
        cv2.rectangle(filled_canvas, self.start_point, self.end_point, color=fill_colour, thickness=cv2.FILLED)
        self.filled_mask = filled_canvas
        self.fill_colour = fill_colour
        # outline mask
        outline_canvas = np.zeros(img_size)
        cv2.rectangle(outline_canvas, self.start_point, self.end_point, color=outline_colour, thickness=outline_thickness)
        self.outline_mask = outline_canvas
        self.outline_colour = outline_colour

    def adjust(self, x_offset, y_offset):
        """Returns a new rectangle with points adjusted by an x and y offset."""
        adjusted_start = (self.start_point[0] + x_offset, self.start_point[1] + y_offset)
        adjusted_end = (self.end_point[0] + x_offset, self.end_point[1] + y_offset)
        return Rectangle(adjusted_start, adjusted_end)

    def get_closest_point(self, point):
        start_x_end_y = (self.start_point[0], self.end_point[1])
        end_x_start_y = (self.end_point[0], self.start_point[1])
        corners = [self.start_point, start_x_end_y, self.end_point, end_x_start_y]
        boundary_points = generate_points_on_shape_boundary(corners)
        return get_closest_point_between_points(boundary_points, point, self.centre)


class Circle(Shape):
    def __init__(self, centre, radius):
        super().__init__('circle')
        self.centre = centre
        self.radius = radius

    def create_masks(self, img_size, fill_colour=[1, 1, 1], outline_colour=[2, 2, 2], outline_thickness=1):
        # warn about merge issue
        if fill_colour == outline_colour:
            print("WARNING: The shape filling colour and outline colour are the same. May cause issues when merging.")
        # filled mask
        filled_canvas = np.zeros(img_size)
        cv2.circle(filled_canvas, self.centre, self.radius, color=fill_colour, thickness=cv2.FILLED)
        self.filled_mask = filled_canvas
        self.fill_colour = fill_colour
        # outline mask
        outline_canvas = np.zeros(img_size)
        cv2.circle(outline_canvas, self.centre, self.radius, color=outline_colour, thickness=outline_thickness)
        self.outline_mask = outline_canvas
        self.outline_colour = outline_colour

    def adjust(self, x_offset, y_offset):
        """Returns a new circle with its centre adjusted by an x and y offset."""
        adjusted_centre = (self.centre[0] + x_offset, self.centre[1] + y_offset)
        return Circle(adjusted_centre, self.radius)

    def get_closest_point(self, point):
        distance = get_distance_to_point(self.centre, point)
        return get_ratio_interval_point(self.centre, point, self.radius, distance - self.radius)


class Polygon(Shape):
    def __init__(self, points):
        super().__init__('polygon')
        self.points = points
        self.centre = self._calculate_centre()

    def _calculate_centre(self):
        return tuple(np.mean(self.points, axis=0).astype(int))

    def create_masks(self, img_size, fill_colour=[1, 1, 1], outline_colour=[2, 2, 2], outline_thickness=1):
        # warn about merge issue
        if fill_colour == outline_colour:
            print("WARNING: The shape filling colour and outline colour are the same. May cause issues when merging.")
        # filled mask
        filled_canvas = np.zeros(img_size)
        cv2.fillPoly(filled_canvas, pts=np.int32([self.points]), color=fill_colour)
        self.filled_mask = filled_canvas
        self.fill_colour = fill_colour
        # outline mask
        outline_canvas = np.zeros(img_size)
        cv2.polylines(outline_canvas, pts=np.int32([self.points]), isClosed=True, color=outline_colour, thickness=outline_thickness)
        self.outline_mask = outline_canvas
        self.outline_colour = outline_colour

    def adjust(self, x_offset, y_offset):
        """Returns a new polygon with points adjusted by an x and y offset."""
        adjusted_points = [(x + x_offset, y + y_offset) for x, y in self.points]
        return Polygon(adjusted_points)

    def get_closest_point(self, point):
        all_boundary_points = generate_points_on_shape_boundary(self.points)
        return get_closest_point_between_points(all_boundary_points, point, self.centre)
