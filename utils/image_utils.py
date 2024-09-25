
import cv2
import numpy as np


def uint_to_float(array, scale=False):
    if array.dtype == np.uint8:
        if scale:
            array = cv2.normalize(array, np.zeros(array.shape), 0, 1, cv2.NORM_MINMAX)
        else:
            array = array / 255.0
    return array


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
