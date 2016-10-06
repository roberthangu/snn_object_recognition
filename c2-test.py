#!/bin/ipython
import numpy as np
import cv2
import pyNN.nest as sim
import pathlib as plb
import time
import pickle
import argparse as ap
from sklearn import svm, metrics

import common as cm
import network as nw
import visualization as vis

parser = ap.ArgumentParser('./c1-spikes-from-file-test.py --')
parser.add_argument('--training-c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains for\
                         training')
parser.add_argument('--validation-c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains for\
                         validation')
parser.add_argument('--dataset-label', type=str, required=True,
                    help='The name of the dataset which was used for\
                    training')
parser.add_argument('--training-image-count', type=int, required=True,
                    help='The number of iterations for the images from the\
                         training dataset')
parser.add_argument('--validation-image-count', type=int, required=True,
                    help='The number of iterations for the images from the\
                         validation dataset')
parser.add_argument('--training-labels', type=str, required=True,
                    help='Text file which contains the labels of the training\
                          dataset')
parser.add_argument('--validation-labels', type=str, required=True,
                    help='Text file which contains the labels of the validation\
                          dataset')
parser.add_argument('--plot-c1-spikes', action='store_true',
                    help='Plot the spike trains of the C1 layers')
parser.add_argument('--plot-c2-spikes', action='store_true',
                    help='Plot the spike trains of the C2 layers')
parser.add_argument('--plot-s2-spikes', action='store_true',
                    help='Plot the spike trains of the S2 layers')
parser.add_argument('--sim-time', default=50, type=float, metavar='50',
                     help='Simulation time')
parser.add_argument('--threads', default=1, type=int)
parser.add_argument('--weights-from', type=str, required=True,
                    help='Dumpfile of the S2 weight array')
args = parser.parse_args()

sim.setup(threads=args.threads, min_delay=.1)

layer_collection = {}

print('Create C1 layers')
t1 = time.clock()
dumpfile = open(args.training_c1_dumpfile, 'rb')
ddict = pickle.load(dumpfile)
layer_collection['C1'] = {}
sizes = ''
for size, layers_as_dicts in ddict.items():
    sizes += '_{}'.format(str(size))
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

weights_file = open(args.weights_from, 'rb')
epoch_weights_list = pickle.load(weights_file)
weights_file.close()
epoch = epoch_weights_list[-1][0]
weights_dict_list = epoch_weights_list[-1][1]
f_s = int(np.sqrt(list(weights_dict_list[0].values())[0].shape[0]))
s2_prototype_cells = len(weights_dict_list)

layer_collection['S2'] = nw.create_S2_layers(layer_collection['C1'], f_s,
                                             s2_prototype_cells, stdp=False)
# Set the S2 weights to those from the file
print('Setting S2 weights to epoch', epoch)
for prototype in range(s2_prototype_cells):
    nw.set_s2_weights(layer_collection['S2'], prototype,
                      weights_dict_list=weights_dict_list)
print('S2 creation took {} s'.format(time.clock() - t1))

dataset_label = '{}_fs{}_{}imgs_{}ms_scales{}'.format(args.dataset_label, f_s,
                        args.training_image_count + args.validation_image_count,
                        int(args.sim_time), sizes)

print('Creating C2 layers')
t1 = time.clock()
layer_collection['C2'] = nw.create_C2_layers(layer_collection['S2'],
                                             s2_prototype_cells)
print('C2 creation took {} s'.format(time.clock() - t1))

if args.plot_c1_spikes:
    for layers in layer_collection['C1'].values():
        for layer in layers:
            layer.population.record('spikes')
if args.plot_s2_spikes:
    for layer_list in layer_collection['S2'].values():
        for layer in layer_list:
            layer.population.record(['spikes', 'v'])
for pop in layer_collection['C2']:
    pop.record('spikes')

if args.plot_c1_spikes:
    c1_plots_dir_path = plb.Path('plots/C1/' + dataset_label)
    if not c1_plots_dir_path.exists():
        c1_plots_dir_path.mkdir(parents=True)
if args.plot_s2_spikes:
    s2_plots_dataset_dir = plb.Path('plots/S2/' + dataset_label)
    for i in range(s2_prototype_cells):
        s2_plots_dir_path = s2_plots_dataset_dir / str(i)
        if not s2_plots_dir_path.exists():
            s2_plots_dir_path.mkdir(parents=True)
if args.plot_c2_spikes:
    c2_plots_dir_path = plb.Path('plots/C2/' + dataset_label)
    if not c2_plots_dir_path.exists():
        c2_plots_dir_path.mkdir(parents=True)
training_labels = open(args.training_labels, 'r').read().splitlines()
validation_labels = open(args.validation_labels, 'r').read().splitlines()
training_samples = []
validation_samples = []

print('>>>>>>>>> Extracting data samples for fitting <<<<<<<<<')
print('========= Start simulation =========')
start_time = time.clock()
for i in range(args.training_image_count):
    print('Simulating for training image number', i)
    sim.run(args.sim_time)
    if args.plot_c1_spikes:
        vis.plot_C1_spikes(layer_collection['C1'],
                           '{}_image_{}'.format(dataset_label, i),
                           out_dir_name=c1_plots_dir_path.as_posix())
    if args.plot_s2_spikes:
        vis.plot_S2_spikes(layer_collection['S2'],
                       '{}_image_{}'.format(dataset_label, i),
                       s2_prototype_cells,
                       out_dir_name=s2_plots_dataset_dir.as_posix())
    spikes =\
        [list(layer_collection['C2'][prot].get_spike_counts().values())[0]\
            for prot in range(s2_prototype_cells)]
    training_samples.append(spikes)
    for prot in range(s2_prototype_cells):
        layer_collection['C2'][prot].get_data(clear=True)
    if args.plot_c2_spikes:
        vis.plot_C2_spikes(layer_collection['C2'], i, args.sim_time,
                           '{}_epoch_{}_image_{}'.format(dataset_label, epoch, i),
                           out_dir_name=c2_plots_dir_path.as_posix())
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')

print('Setting C1 spike trains to the validation dataset')
dumpfile = open(args.validation_c1_dumpfile, 'rb')
ddict = pickle.load(dumpfile)
for size, layers_as_dicts in ddict.items():
    for layer_as_dict in layers_as_dicts:
        spiketrains = layer_as_dict['segment'].spiketrains
        dimensionless_sts = [[s for s in st] for st in spiketrains]
        the_layer_iter = filter(lambda layer: layer.population.label\
                        == layer_as_dict['label'], layer_collection['C1'][size])
        the_layer_iter.__next__().population.set(spike_times=dimensionless_sts)

print('>>>>>>>>> Extracting data samples for validation <<<<<<<<<')
print('========= Start simulation =========')
for i in range(args.validation_image_count):
    print('Simulating for validation image number', i)
    sim.run(args.sim_time)
    spikes =\
        [list(layer_collection['C2'][prot].get_spike_counts().values())[0]\
            for prot in range(s2_prototype_cells)]
    validation_samples.append(spikes)
    for prot in range(s2_prototype_cells):
        layer_collection['C2'][prot].get_data(clear=True)
print('========= Stop  simulation =========')

print('Fitting SVM model onto the training samples')

clf = svm.SVC(kernel='linear')
clf.fit(training_samples, training_labels)

print('Predicting the categories of the validation samples')
predicted_labels = clf.predict(validation_samples)
print('Prediction is')
print(metrics.classification_report(validation_labels, predicted_labels))
print('Confusion matrix is')
print(metrics.confusion_matrix(validation_labels, predicted_labels))

sim.end()
