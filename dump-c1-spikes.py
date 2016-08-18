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
print('Create S1 layers')
t1 = time.clock()
layer_collection['S1'] =\
    nw.create_gabor_input_layers_for_scales(target_img, args.scales)
nw.create_cross_layer_inhibition(layer_collection['S1'])
print('S1 layer creation took {} s'.format(time.clock() - t1))

print('Create C1 layers')
t1 = time.clock()
layer_collection['C1'] = nw.create_C1_layers(layer_collection['S1'],
                                             args.refrac_c1)
nw.create_local_inhibition(layer_collection['C1'])
print('C1 creation took {} s'.format(time.clock() - t1))

for layer_name in ['C1']:
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

print('Dumping spikes for all scales and layers')
ddict = {}
filename = 'C1_spike_data/{}'.format(plb.Path(args.target_name).stem)
for size, layers in layer_collection['C1'].items():
    ddict[size] = [{'segment': layer.population.get_data().segments[0],
                    'shape': layer.shape,
                    'label': layer.population.label } for layer in layers]
    filename += '_{}'.format(size)
dumpfile = open('{}_{}ms.bin'.format(filename, args.sim_time), 'wb')
pickle.dump(ddict, dumpfile, protocol=4)
dumpfile.close()

sim.end()
