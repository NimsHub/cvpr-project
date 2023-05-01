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
