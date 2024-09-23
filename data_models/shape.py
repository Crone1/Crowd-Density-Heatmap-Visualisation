
import numpy as np
import cv2


class Shape:
    def __init__(self, shape_type):
        self.shape_type = shape_type
        self.centre = None
        self.filled_mask = None
        self.outline_mask = None

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

    def create_masks(self, img_size, filled_val=0.5, outline_val=1, outline_thickness=1):
        # filled mask
        filled_canvas = np.zeros(img_size)
        cv2.rectangle(filled_canvas, self.start_point, self.end_point, color=[filled_val] * 3, thickness=cv2.FILLED)
        self.filled_mask = filled_canvas
        # outline mask
        outline_canvas = np.zeros(img_size)
        cv2.rectangle(outline_canvas, self.start_point, self.end_point, color=[outline_val] * 3, thickness=outline_thickness)
        self.outline_mask = outline_canvas


class Circle(Shape):
    def __init__(self, centre, radius):
        super().__init__('circle')
        self.centre = centre
        self.radius = radius

    def create_masks(self, img_size, filled_val=0.5, outline_val=1, outline_thickness=1):
        # filled mask
        filled_canvas = np.zeros(img_size)
        cv2.circle(filled_canvas, self.centre, self.radius, color=[filled_val] * 3, thickness=cv2.FILLED)
        self.filled_mask = filled_canvas
        # outline mask
        outline_canvas = np.zeros(img_size)
        cv2.circle(outline_canvas, self.centre, self.radius, color=[outline_val] * 3, thickness=outline_thickness)
        self.outline_mask = outline_canvas


class Polygon(Shape):
    def __init__(self, points):
        super().__init__('polygon')
        self.points = points
        self.centre = self._calculate_centre()

    def _calculate_centre(self):
        return tuple(np.mean(self.points, axis=0).astype(int))

    def create_masks(self, img_size, filled_val=0.5, outline_val=1, outline_thickness=1):
        # filled mask
        filled_canvas = np.zeros(img_size)
        cv2.fillPoly(filled_canvas, pts=np.int32([self.points]), color=[filled_val] * 3)
        self.filled_mask = filled_canvas
        # outline mask
        outline_canvas = np.zeros(img_size)
        cv2.polylines(outline_canvas, pts=np.int32([self.points]), isClosed=True, color=[outline_val] * 3, thickness=outline_thickness)
        self.outline_mask = outline_canvas
