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

target_img = cm.read_and_prepare_img(args.target_name, args)
cv2.imwrite('{}_{}_edges.png'.format(plb.Path(args.target_name).stem,
                                     args.filter), target_img)

layer_collection = {} # layer name -> list of S1 layers
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
print('Simulation took', end_time - start_time, 's')

if args.reconstruct_s1_img:
    vis.reconstruct_S1_features(target_img, layer_collection, feature_imgs_dict,
                                args)

if args.reconstruct_c1_img:
    vis.reconstruct_C1_features(target_img, layer_collection, feature_imgs_dict,
                                args)

# Plot the spike trains of both neuron layers
if args.plot_spikes:
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
