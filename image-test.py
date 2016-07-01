#!/bin/ipython
import numpy as np
import cv2
import sys
import pyNN.nest as sim
import pathlib as plb
import pyNN.utility.plotting as plt
import time

import common as cm
import network as nw
import visualization as vis

# TODO: - Filter input images with a Canny filter

args = cm.parse_args()

weights_dict, feature_imgs_dict = nw.train_weights(args.feature_dir)

if args.plot_weights:
    vis.plot_weights(weights_dict)
    sys.exit(0)

# The training part is done. Go on with the "actual" simulation
sim.setup(threads=4)

target_img = cv2.imread(args.target_name, cv2.CV_8U)
layer_collection = {}
layer_collection['S1'] = nw.create_S1_layers(target_img, weights_dict,
                                             [1, 0.71, 0.5, 0.35, 0.25],
                                             args)
if not args.no_c1:
    layer_collection['C1'] = nw.create_C1_layers(layer_collection['S1'],
                                                 args.refrac_c1)


for layer_dict in layer_collection.values():
    for layers in layer_dict.values():
        for layer in layers:
            layer.population.record('spikes')

print('========= Start simulation =========')
start_time = time.clock()
sim.run(100)
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took:', end_time - start_time)

if args.reconstruct_s1_img:
    print('Reconstructing S1 features')
    vis_img = np.zeros(target_img.shape)
    vis_parts = vis.visualization_parts(target_img.shape,
                                        layer_collection['S1'],
                                        feature_imgs_dict,
                                        args.delta_i, args.delta_j)
    for size, img_pairs in vis_parts.items():
        for img, feature_label in img_pairs:
            vis_img += img
    cv2.imwrite('{}_S1_reconstruction.png'.format(plb.Path(args.target_name).stem),
                                               vis_img)

if args.reconstruct_c1_img:
    print('Reconstructing C1 features')
    # Create the RGB canvas to draw colored rectangles for the features
    canvas = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
    # Create the colored squares for the features in a map
    colored_squares_dict = {} # feature name -> colored square
    # A set of predefined colors
    colors = {'red': (255, 0, 0),
              'green': (0, 255, 0),
              'blue': (0, 0, 255),
              'yellow': (255, 255, 0),
              'purple': (255, 0, 255)}
    color_iterator = colors.__iter__()
    for feature_name, feature_img in feature_imgs_dict.items():
        color_name = color_iterator.__next__()
        f_n, f_m = feature_img.shape
        bf_n = 6 * args.delta_i + f_m   # "big" f_n
        bf_m = 6 * args.delta_j + f_m   # "big" f_m
        # Create a square to cover all pixels of a C1 neuron
        square = np.zeros((bf_n, bf_m, 3))
        cv2.rectangle(square, (0, 0), (bf_n - 1, bf_m - 1), colors[color_name])
        print('feature name and color: ', feature_name, color_name)
        colored_squares_dict[feature_name] = square
    vis_parts = vis.visualization_parts(target_img.shape, layer_collection['C1'],
                                        colored_squares_dict,
                                        6 * args.delta_i,
                                        6 * args.delta_j, canvas)
    for size, img_pairs in vis_parts.items():
        for img, feature_label in img_pairs:
            cv2.imwrite('reconstruction_components/{}_{}_C1_{}_reconstruction.png'.\
                        format(plb.Path(args.target_name).stem, size, feature_label),
                        img)

# Plot the spike trains of both neuron layers
for layer_name, layer_dict in layer_collection.items():
    for size, layers in layer_dict.items():
        spike_panels = []
        for layer in layers:
            out_data = layer.population.get_data().segments[0]
            spike_panels.append(plt.Panel(out_data.spiketrains,# xlabel='Time (ms)',
                                          xticks=True, yticks=True,
                                          xlabel='{}, {} scale layer'.format(\
                                                    layer.population.label, size)))
        plt.Figure(*spike_panels).save('plots/{}_{}_{}_scale.png'.format(\
                                                layer_name,
                                                plb.Path(args.target_name).stem,
                                                size))

sim.end()
