#!/bin/ipython
import numpy as np
import cv2
import sys
import pyNN.nest as sim
import pathlib as plb
import time

import common as cm
import network as nw
import visualization as vis
import time

args = cm.parse_args()

#gabor_edges = cm.get_gabor_edges(cv2.imread(args.target_name, cv2.CV_8UC1))
#
#for edge_name, img in gabor_edges.items():
#    filename = 'edges/{}_gabor_{}.png'.format(plb.Path(args.target_name).stem,
#                                              edge_name)
#    if not plb.Path(filename).exists():
#        cv2.imwrite(filename, img)

sim.setup(threads=4)

layer_collection = {}

target_img = cv2.imread(args.target_name, cv2.CV_8UC1)
print('Create input layers')
t1 = time.clock()
layer_collection['input'] =\
    nw.create_gabor_input_layers_for_scales(target_img, args.scales)
print('Input layer creation took {} s'.format(time.clock() - t1))

print('Create S1 layers')
t1 = time.clock()
layer_collection['S1'] = nw.create_gabor_S1_layers(layer_collection['input'])
nw.create_cross_layer_inhibition(layer_collection['S1'])
print('S1 creation took {} s'.format(time.clock() - t1))

print('Create C1 layers')
t1 = time.clock()
layer_collection['C1'] = nw.create_C1_layers(layer_collection['S1'],
                                             args.refrac_c1)
nw.create_local_inhibition(layer_collection['C1'])
print('C1 creation took {} s'.format(time.clock() - t1))

for layer_name in ['S1', 'C1']:
    if layer_name in layer_collection:
        for layers in layer_collection[layer_name].values():
            for layer in layers:
                layer.population.record('spikes')

print('========= Start simulation =========')
start_time = time.clock()
sim.run(args.sim_time)
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')

for layer_name in ['S1', 'C1']:
    if layer_name in layer_collection:
        for layers in layer_collection[layer_name].values():
            for layer in layers:
                layer.update_spike_counts()

feature_names = ['slash', 'horiz_slash', 'horiz_backslash', 'backslash']
feature_imgs_dict = dict([(name, np.zeros((1,1))) for name in feature_names])

t1 = time.clock()
vis.reconstruct_C1_features(target_img, layer_collection, feature_imgs_dict,
                            args)
print('C1 visualization took {} s'.format(time.clock() - t1))

t1 = time.clock()
if args.plot_spikes:
    print('Plotting spikes')
    vis.plot_spikes(layer_collection, args)
    print('Plotting spiketrains took {} s'.format(time.clock() - t1))

sim.end()
