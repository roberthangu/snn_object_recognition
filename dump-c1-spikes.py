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

try:
    from mpi4py import MPI
except ImportError:
    raise Exception("Trying to gather data without MPI installed. If you are\
    not running a distributed simulation, this is a bug in PyNN.")

parser = ap.ArgumentParser('./dump-c1-spikes.py --')
parser.add_argument('--dataset-label', type=str, required=True,
                    help='The name of the dataset which was used for\
                    training')
parser.add_argument('--image-count', type=int, required=True,
                    help='The number of images to read from the training\
                    directory')
parser.add_argument('--training-dir', type=str, required=True,
                    help='The directory with the training images')
parser.add_argument('--refrac-c1', type=float, default=.1, metavar='0.1',
                    help='The refractory period of neurons in the C1 layer in\
                    ms')
parser.add_argument('--sim-time', default=50, type=float, help='Simulation time',
                    metavar='50')
parser.add_argument('--scales', default=[1.0, 0.71, 0.5, 0.35],
                    nargs='+', type=float,
                    help='A list of image scales for which to create\
                    layers. Defaults to [1, 0.71, 0.5, 0.35, 0.25]')
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

MPI_ROOT = 0

def is_root():
    return MPI.COMM_WORLD.rank == MPI_ROOT 

training_path = plb.Path(args.training_dir)
imgs = [(filename.stem, cv2.imread(filename.as_posix(), cv2.CV_8UC1))\
            for filename in training_path.iterdir()]

sim.setup()

layer_collection = {}

print('Create S1 layers')
t1 = time.clock()
layer_collection['S1'] =\
    nw.create_empty_input_layers_for_scales(imgs[0][1], args.scales)
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

if is_root():
    print('========= Start simulation =========')
    start_time = time.clock()
    count = 0
for filename, target_img in imgs[0:args.image_count]:
    if is_root():
        t1 = time.clock()
        print('Simulating for', filename, 'number', count)
        count += 1
    nw.set_i_offsets_for_all_scales_to(layer_collection['S1'], target_img)
    sim.run(args.sim_time)
    if is_root():
        print('Took', time.clock() - t1, 'seconds')
        end_time = time.clock()
        print('========= Stop  simulation =========')
        print('Simulation took', end_time - start_time, 's')

if is_root():
    print('Dumping spikes for all scales and layers')
ddict = {}
filename = 'C1_spike_data/' + args.dataset_label
for size, layers in layer_collection['C1'].items():
    ddict[size] = [{'segment': layer.population.get_data().segments[0],
                    'shape': layer.shape,
                    'label': layer.population.label } for layer in layers]
    filename += '_{}'.format(size)

if is_root():
    dumpfile = open('{}_{}ms_{}_images.bin'.format(filename, args.sim_time,
                                                   args.image_count), 'wb')
    pickle.dump(ddict, dumpfile, protocol=4)
    dumpfile.close()

sim.end()
