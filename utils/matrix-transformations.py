
import cv2
import numpy as np


def order_points(pts):
    """
    Helper function for four_point_transform.
    Check pyimagesearch blog for an explanation on the matter
    """
    # Order: top-left, top-right, bottom-right and top-left
    rect = np.zeros((4, 2), dtype=np.float32)
    # top-left will have smallest sum, while bottom-right
    # will have the largest one
    _sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(_sum)]
    rect[2] = pts[np.argmax(_sum)]
    # top-right will have smallest difference, while
    # bottom-left will have the largest one
    _diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(_diff)]
    rect[3] = pts[np.argmax(_diff)]
    return rect