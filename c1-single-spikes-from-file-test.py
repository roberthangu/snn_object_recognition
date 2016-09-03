#!/bin/ipython
import numpy as np
import cv2
import sys
import pyNN.nest as sim
import pathlib as plb
import time
import pickle
import argparse as ap

import common as cm
import network as nw
import visualization as vis
import time

parser = ap.ArgumentParser('./c1-spikes-from-file-test.py --')
parser.add_argument('--c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains')
parser.add_argument('--plot-c1-spikes', action='store_true',
                    help='Plot the spike trains of the C1 layers')
parser.add_argument('--plot-s2-spikes', action='store_true',
                    help='Plot the spike trains of the S2 layers')
parser.add_argument('--refrac-s2', type=float, default=.1, metavar='MS',
                    help='The refractory period of neurons in the S2 layer in ms')
parser.add_argument('--sim-time', default=50, type=float, metavar='50',
                     help='Simulation time')
parser.add_argument('--target-name', type=str,
                    help='The name of the already edge-filtered image to be\
                    recognized')
args = parser.parse_args()

sim.setup(threads=4)

layer_collection = {}

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
                        sim.SpikeSourceArray(spike_times=dimensionless_sts),
                        label=layer_as_dict['label']), (n, m))
        layer_list.append(new_layer)
    layer_collection['C1'][size] = layer_list
print('C1 creation took {} s'.format(time.clock() - t1))

print('Creating S2 layers')
t1 = time.clock()
layer_collection['S2'] = nw.create_S2_layers(layer_collection['C1'], args)
print('S2 creation took {} s'.format(time.clock() - t1))

for layers in layer_collection['C1'].values():
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

t1 = time.clock()
if args.plot_c1_spikes:
    print('Plotting C1 spikes')
    vis.plot_C1_spikes(layer_collection['C1'], plb.Path(args.target_name).stem)
    print('Plotting spiketrains took {} s'.format(time.clock() - t1))

if args.plot_s2_spikes:
    print('Plotting S2 spikes')
    vis.plot_S2_spikes(layer_collection['S2'], plb.Path(args.target_name).stem)
    print('Plotting spiketrains took {} s'.format(time.clock() - t1))

sim.end()
