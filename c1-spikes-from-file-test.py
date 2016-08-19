#!/bin/ipython
import numpy as np
import cv2
import sys
import pyNN.nest as sim
import pathlib as plb
import time
import pickle

import common as cm
import network as nw
import visualization as vis
import time

args = cm.parse_args()

sim.setup(threads=4)

layer_collection = {}

target_img = cv2.imread(args.target_name, cv2.CV_8UC1)

# Read the gabor features for reconstruction
feature_imgs_dict = {} # feature string -> image
for filepath in plb.Path('features_gabor').iterdir():
    feature_imgs_dict[filepath.stem] = cv2.imread(filepath.as_posix(),
                                                  cv2.CV_8UC1)

print('Create C1 layers')
t1 = time.clock()
dumpfile = open(args.c1_dumpfile, 'rb')
ddict = pickle.load(dumpfile)
layer_collection['C1'] = {}
for size, layers_as_dicts in ddict.items():
    layer_list = []
    for layer_as_dict in layers_as_dicts:
        n, m = layer_as_dict['shape']
        spiketrains = layer_as_dict['segment'].spiketrains
        dimensionless_sts = [[s for s in st] for st in spiketrains]
        new_layer = nw.Layer(sim.Population(n * m,
                        sim.SpikeSourceArray(spike_times=dimensionless_sts)), (n, m))
        new_layer.population.label = layer_as_dict['label']
        layer_list.append(new_layer)
    layer_collection['C1'][size] = layer_list
print('C1 creation took {} s'.format(time.clock() - t1))

print('Creating S2 layers')
t1 = time.clock()
layer_collection['S2'] = nw.create_S2_layers(layer_collection['C1'], args)
print('S2 creation took {} s'.format(time.clock() - t1))
#create_S2_inhibition(layer_collection['S2'])

for layer_name in ['C1']:
    if layer_name in layer_collection:
        for layers in layer_collection[layer_name].values():
            for layer in layers:
                layer.population.record('spikes')
for layer in layer_collection['S2'].values():
    layer.population.record(['spikes', 'v'])

print('========= Start simulation =========')
start_time = time.clock()
sim.run(args.sim_time)
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')

final_weights = nw.update_shared_weights(layer_collection['S2'])
reconstruction = vis.reconstruct_S2_features(final_weights, feature_imgs_dict)

cv2.imwrite('S2_reconstructions/{}.png'.format(plb.Path(args.target_name).stem),
            reconstruction)

t1 = time.clock()
print('Plotting spikes')
vis.plot_spikes(layer_collection, args)
print('Plotting spiketrains took {} s'.format(time.clock() - t1))

sim.end()
