import argparse as ap

def parse_args():
    dflt_move=4
    parser = ap.ArgumentParser(description='Invariance layer experiment')
    parser.add_argument('--plot-weights', action='store_true',
                        help='Plots the learned feature weights and exits')
    parser.add_argument('-f', '--feature-dir', type=str, required=True,
                        help='A directory where the features are stored as images')
    parser.add_argument('-t', '--target-name', type=str, required=True,
                        help='The name of the already edge-filtered image to\
                            be recognized')
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
    #parser.add_argument('-o', '--plot_img', type=str, required=True)
    #parser.add_argument('--plot_img', type=str, default='spikes_vert_line.png')
    parser.add_argument('--delta-i', metavar='vert', default=dflt_move, type=int,
                        help='The vertical distance between the basic recognizers')
    parser.add_argument('--delta-j', metavar='horiz', default=dflt_move, type=int,
                        help='The horizontal distance between the basic feature\
                        recognizers')
    args = parser.parse_args()
    print(args)
    return args

