#!/bin/ipython
import numpy as np
import cv2
import sys
import pyNN.nest as sim
import pathlib as plb
import time
import pickle
import argparse as ap

import network as nw
import visualization as vis

parser = ap.ArgumentParser('./dump-single-c1-spikes.py --')
parser.add_argument('--refrac-c1', type=float, default=.1, metavar='0.1',
                    help='The refractory period of neurons in the C1 layer in\
                    ms')
parser.add_argument('--sim-time', default=50, type=float, help='Simulation time',
                    metavar='50')
parser.add_argument('--scales', default=[1.0, 0.71, 0.5, 0.35, 0.25],
                    nargs='+', type=float,
                    help='A list of image scales for which to create\
                    layers. Defaults to [1, 0.71, 0.5, 0.35, 0.25]')
parser.add_argument('--target-name', type=str,
                    help='The name of the already edge-filtered image to be\
                    recognized')
args = parser.parse_args()

target_path = plb.Path(args.target_name)
target_img = cv2.imread(target_path.as_posix(), cv2.CV_8UC1)

sim.setup(threads=4)

layer_collection = {}

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

for layers in layer_collection['C1'].values():
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
filename = 'C1_spike_data/' + target_path.stem
for size, layers in layer_collection['C1'].items():
    ddict[size] = [{'segment': layer.population.get_data().segments[0],
                    'shape': layer.shape,
                    'label': layer.population.label } for layer in layers]
    filename += '_{}'.format(size)
dumpfile = open('{}_{}ms_norefrac.bin'.format(filename, args.sim_time), 'wb')
pickle.dump(ddict, dumpfile, protocol=4)
dumpfile.close()

sim.end()
