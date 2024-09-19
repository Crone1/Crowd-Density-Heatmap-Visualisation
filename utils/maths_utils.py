
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
