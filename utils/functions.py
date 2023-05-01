
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

def four_point_transform(img, pts):
    """Returns 'bird view' of image"""
    rect = order_points(pts)
    tl, tr, br, bl = rect
    # width of new image will be the max difference between
    # bottom-right - bottom-left or top-right - top-left
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    width = int(round(max(widthA, widthB)))
    # Same goes for height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    height = int(round(max(heightA, heightB)))
    # construct destination for 'birds eye view'
    dst = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32)
    # compute perspective transform and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def resize(img, new_width):
    """Resizes image to new_width while maintaining its ratio"""
    height, width = img.shape[:2]
    ratio = height / width
    return cv2.resize(img, (new_width, int(ratio * new_width)))

def find_corners(img):
    """Finds harris corners"""
    corners = cv2.cornerHarris(img, 5, 3, 0.1)
    corners = cv2.dilate(corners, None)
    corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)[1]
    corners = corners.astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        corners, connectivity=4)
    # For some reason, stats yielded better results for
    # corner detection than centroids. This might have
    # something to do with sub-pixel accuracy. 
    # Check issue #10130 on opencv
    return stats


def contoured_bbox(img):
    """Returns bbox of contoured image"""
    contours, hierarchy = cv2.findContours(img, 1, 2)
    # Largest object is whole image,
    # second largest object is the ROI
    sorted_cntr = sorted(contours, key=lambda cntr: cv2.contourArea(cntr))
    return cv2.boundingRect(sorted_cntr[-2])


def preprocess_input(img):
    """Preprocess image to match model's input shape for shape detection"""
    img = cv2.resize(img, (32, 32))
    # Expand for channel_last and batch size, respectively
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32) / 255
