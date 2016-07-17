import argparse as ap
import cv2
import nest as sim
import numpy as np
import pickle
import time

def parse_args():
    """
    Defines the valid commandline options and the variables they are linked to.

    Returns:

        An object which contains the variables which correspond to the
        commandline options.
    """
    dflt_move=4
    parser = ap.ArgumentParser(description='SNN feature detector')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='A directory where the features are stored as images')
    parser.add_argument('--target-name', type=str, required=True,
                        help='The name of the already edge-filtered image to\
                              be recognized')
    parser.add_argument('--filter', choices=['canny', 'sobel'],
                        default='none', help='Sets the edge filter to be used.\
                        Defaults to \'none\'')
    parser.add_argument('--plot-weights', action='store_true',
                        help='Plots the learned feature weights and exits')
    parser.add_argument('--refrac-s1', type=float, default=.1, metavar='MS',
                        help='The refractory period of neurons in the S1 layer in ms')
    parser.add_argument('--refrac-c1', type=float, default=.1, metavar='MS',
                        help='The refractory period of neurons in the C1 layer in ms')
    parser.add_argument('--no-c1', action='store_true',
                        help='Disables the creation of C1 layers')
    parser.add_argument('--reconstruct-s1-img', action='store_true',
                        help='If set, draws a reconstruction of the recognized\
                        features from S1')
    parser.add_argument('--reconstruct-c1-img', action='store_true',
                        help='If set, draws a reconstruction of the recognized\
                        features from C1')
    parser.add_argument('--plot-spikes', action='store_true',
                        help='Plot the spike trains of all layers')
    parser.add_argument('--delta-i', metavar='vert', default=dflt_move, type=int,
                        help='The vertical distance between the basic recognizers')
    parser.add_argument('--delta-j', metavar='horiz', default=dflt_move, type=int,
                        help='The horizontal distance between the basic feature\
                        recognizers')
    parser.add_argument('--scales', default=[1, 0.71, 0.5, 0.35, 0.25], nargs='+',
                        type=float,
                        help='A list of image scales for which to create\
                        layers. Defaults to [1, 0.71, 0.5, 0.35, 0.25]')
    parser.add_argument('--sim-time', default=100, type=float, help='Simulation time')
    args = parser.parse_args()
    print(args)
    return args

def read_and_prepare_img(target_name, filter_type):
    """
    Reads the input image and performs the edge detector of the passed
    commandline arguments on it

    Arguments:

        `target_name`: The name of the image to be read

        `filter_type`: The filter to be applied to the target image. Can be one
                       of 'canny', 'sobel' or 'none', if the image is to be
                       used as-is.

    Returns:

        An image containing the edges of the target image 
    """
    target_img = cv2.imread(target_name, cv2.CV_8U)
    # Optionally resize the image to 300 pixels (or less) in height
    blurred_img = cv2.GaussianBlur(target_img, (5, 5), 1.4)
    filtered_img = None
    if filter_type == 'none':
        return target_img
    if filter_type == 'canny':
        filtered_img = cv2.Canny(blurred_img, 70, 210)
    else:
        dx = cv2.Sobel(blurred_img, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(blurred_img, cv2.CV_32F, 0, 1)
        edge_detected = cv2.sqrt(dx * dx + dy * dy)
        filtered_img = cv2.convertScaleAbs(edge_detected)
    return filtered_img
