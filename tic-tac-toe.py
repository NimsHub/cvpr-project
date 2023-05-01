import os
import sys
import cv2
import argparse
import numpy as np

from keras.models import load_model

from utils import imutils
from utils import detections
from alphabeta import Tic, get_enemy, determine

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('cam', type=int,
                        help='USB camera for video streaming')
    parser.add_argument('--model', '-m', type=str, default='data/model.h5',
                        help='model file (.h5) to detect Xs and Os')

    return parser.parse_args()

def find_sheet_paper(frame, thresh, add_margin=True):
    """Detect the coords of the sheet of paper the game will be played on"""
    stats = detections.find_corners(thresh)
    # First point is center of coordinate system, so ignore it
    # We only want sheet of paper's corners
    corners = stats[1:, :2]
    corners = imutils.order_points(corners)
    # Get bird view of sheet of paper
    paper = imutils.four_point_transform(frame, corners)
    if add_margin:
        paper = paper[10:-10, 10:-10]
    return paper, corners

def find_shape(cell):
    """Is shape and X or an O?"""
    mapper = {0: None, 1: 'X', 2: 'O'}
    cell = detections.preprocess_input(cell)
    idx = np.argmax(model.predict(cell))
    return mapper[idx]
