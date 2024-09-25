
import numpy as np


def get_slope(p1, p2):
    """
    Function Goal : Get the slope of a line going through 2 points

    p1 : tuple of integers (int, int) - a point
    p2 : tuple of integers (int, int) - a point

    return : integer - the slope of the line going through the 2 points
    """
    return (p2[1] - p1[1])/(p2[0] - p1[0])


def get_equation_of_line(p1, p2):
    """
    Functon Goal : This function gets the coefficients and the constants of the equation of a line passing through 2 points

    p1 : a tuple of integers (int, int) - a point on the cartesian plane
    p2 : a tuple of integers (int, int) - a point on the cartesian plane

    return : a tuple of a list of integers and an integer ([int, int], int) - a list containing the coefficients for the x and y values in the equation of the line and the constant in the equation of the line
    """
    slope = get_slope(p1, p2)
    x1, y1 = p1

    coefficients = [-slope, 1]
    constant = y1 - (slope * x1)
    return coefficients, constant


def get_distance(p1, p2):
    """
    Function Goal : get the distance between 2 points

    p1 : tuple of integers (int, int) - a point
    p2 : tuple of integers (int, int) - a point

    return : integer - th distance between p1 and p2
    """
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_distance_to_line(line_start, line_end, point):
    """
    Function Goal : get the distance between a point and a line between two points

    line_start : np.Array integers np.Array([int, int]) - start point for the line
    line_end : np.Array of integers np.Array(int, int]) - end point for the line
    point: np.Array of integers np.Array([int, int]) - a point

    return : float - the distance between the point and the line
    """
    return np.linalg.norm(np.cross(line_end-line_start, line_start-point))/np.linalg.norm(line_end-line_start)


def get_ratio_interval_point(p1, p2, a, b):
    """
    Function Goal : find the point that divides 2 points into the ratio a:b where dist(p1, point):dist(point, p2) is proportional to a:b

    p1 : tuple of integers (int, int) - a point
    p2 : tuple of integers (int, int) - a point
    a : integer - ratio a
    b : integer - ratio b

    return : tuple of integers (int, int) - the point that divides the 2 points into the ration a:b
    """
    x = ((b * p1[0]) + (a * p2[0]))/(a + b)
    y = ((b * p1[1]) + (a * p2[1]))/(a + b)
    return int(x), int(y)


def convert_cartesian_to_polar(point):
    """
    Function Goal : Turn a point into polar coordinates based on its location with respect to the origin (0,0)

    point : a tuple of integers (int, int) - a point on the x-y plane

    return : float, float - the distance between the point and the origin and the angle between a line from the point to the origin and the x axis
    """
    rho = np.sqrt(point[0]**2 + point[1]**2)
    phi = np.arctan2(point[1], point[0])
    return rho, phi


def convert_polar_to_cartesian(rho, phi):
    """
    Function Goal : Turn the polar coordinates into cartesian coordinates

    rho : float - The distance from the output point to the origin (0,0)
    phi : float - The angle between a line from the output point to the origin and a line along the x-axis

    return : a tuple of integers (int, int) - a point on the x-y plane
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return int(x), int(y)


def generate_points_on_shape_boundary(boundary_points):
    """
    Function Goal : Use the boundary corner points to generate points on the perimeter of the shape

    boundary_points : list of tuples of integers [(int, int), (int, int), ...] - a list of the points representing the corners of the boundary shape

    return : set of tuples of integers [(int, int), (int, int), ...etc.] - a list of points that surround the whole area
    """

    gen_points = set()
    for i in range(1, len(boundary_points)):
        prev_x, prev_y = boundary_points[i]
        x, y = boundary_points[0] if i + 1 >= len(boundary_points) else boundary_points[i + 1]

        # if is horizontal line between previous
        if prev_x == x:
            for new_y in range(min(prev_y, y), max(prev_y, y)):
                gen_points.add((x, new_y))

        # if is vertical line between previous
        elif prev_y == y:
            for new_x in range(min(prev_x, x), max(prev_x, x)):
                gen_points.add((new_x, y))

        # if is sloped line between previous
        else:
            slope = get_slope(p1=(prev_x, prev_y), p2=(x, y))
            length_on_x_axis = max(prev_x, x) - min(prev_x, x)
            length_on_y_axis = max(prev_y, y) - min(prev_y, y)
            for j in range(max(length_on_x_axis, length_on_y_axis) + 1):
                if length_on_x_axis > length_on_y_axis:
                    x_val = min(prev_x, x) + j
                    new_y = int(prev_y - (slope * prev_x) + (slope * x_val))
                    point = (x_val, new_y)
                else:
                    y_val = min(prev_y, y) + j
                    new_x = int(prev_x + ((y_val - prev_y)/slope))
                    point = (new_x, y_val)
                gen_points.add(point)

    return gen_points


def get_closest_point_between_points(points_list, point1, point2):
    """
    Function Goal : Find the point within a list of points which falls between two points and is closest to the first point

    points_list : list of tuples of integers [(int, int), (int, int), ...] - list of points to find the closest point within
    point1 : tuple of integers (int, int) - a point we want to be close to
    point2 : tuple of integers (int, int) - a point

    return : the point between point1 & point2 which is closest to point1
    """
    closest = None
    closest_dist = np.inf
    for boundary_point in points_list:
        start_x, start_y = point1
        end_x, end_y = point2
        x, y = boundary_point
        x_is_between = (start_x < x < end_x) or (end_x < x < start_x)
        y_is_between = (start_y < y < end_y) or (end_y < y < start_y)
        if x_is_between and y_is_between:
            dist = get_distance_to_line(np.array(point1), np.array(point2), np.array(boundary_point))
            if dist < closest_dist:
                closest = boundary_point
                closest_dist = dist

    return closest
