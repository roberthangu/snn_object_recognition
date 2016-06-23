#!/bin/ipython
import cv2
import numpy as np
import pyNN.utility.plotting as plt
import sys
import pathlib as plb
import argparse as ap
import pyNN.nest as sim
import rosbag

dflt_move=4
parser = ap.ArgumentParser(description='Invariance layer experiment')
parser.add_argument('--plot_weights', action='store_true')
parser.add_argument('-f', '--feature_dir', type=str, required=True)
parser.add_argument('-t', '--target_name', type=str, required=True)
#parser.add_argument('-o', '--plot_img', type=str, required=True)
#parser.add_argument('--plot_img', type=str, default='spikes_vert_line.png')
parser.add_argument('--delta_i', metavar='vert', default=dflt_move, type=int,
                    help='The vertical distance between the basic recognizers')
parser.add_argument('--delta_j', metavar='horiz', default=dflt_move, type=int,
                    help='The horizontal distance between the basic recognizers')
args = parser.parse_args()
print(args)

class Layer:
    # TODO: In the future the layer class may also store other information, like
    #       references to the layer before and after it. Think about a nice way
    #       to represent the layers in this pipeline.
    """
    Represents a layer in the network architecture.

    Attributes:

        `population`: The pyNN neuron population of the layer

        `shape`:      The shape of the layer as a tuple
    """

    def __init__(self, population, shape):
        self.population = population
        self.shape = shape

class Stream:
    def __init__(self, events, shape):
        self.events = events
        self.shape = shape

def resize_stream(stream, size):
    # no interpolation so far
    resized_shape = np.ceil(np.multiply(stream.shape, size)).astype(int)
    resized_events = np.copy(stream.events)
    for event in resized_events:
        event.x = int(np.floor(event.x * size))
        event.y = int(np.floor(event.y * size))
    return Stream(resized_events, resized_shape)

def read_stream(filename):
    bag = rosbag.Bag(filename)
    allEvents = []
    initial_time = None
    for topic, msg, t in bag.read_messages(topics=['/dvs/events']):
        if not initial_time and msg.events:
            # we want the first event to happen at 1ms
            initial_time = int(msg.events[0].ts.to_sec() * 1000) - 1
        for event in msg.events:
            event.ts = int(event.ts.to_sec() * 1000) - initial_time
        allEvents = np.append(allEvents, msg.events)
        shape = [msg.width, msg.height]
    bag.close()

    return Stream(allEvents, shape)

def create_spike_source_layer_from_stream(stream):
    nNeurons = stream.shape[0] * stream.shape[1]
    spike_times = []

    for neuron in range(nNeurons):
        spike_times.append([])
    for event in stream.events:
        nIdx = event.x * stream.shape[0] + event.y
        spike_times[nIdx].append(event.ts)

    spike_source_layer = sim.Population(size=len(spike_times),
                                   cellclass=sim.SpikeSourceArray(spike_times=spike_times))
    return Layer(spike_source_layer, stream.shape)

def create_spike_source_layer_from(source_np_array):
    """
    For a given image returns a layer of spike source neurons to encode the
    image intensities in spikes. Acts as an encoder. The size of the spike
    source layer is the number of pixels in the image.
    """
    reshaped_array = source_np_array.ravel()
    rates = []
    for rate in reshaped_array:
        rates.append(int(rate / 4))
    spike_source_layer = sim.Population(size=len(rates),
                                   cellclass=sim.SpikeSourcePoisson(rate=rates))
    return Layer(spike_source_layer, source_np_array.shape)

def recognizer_weights_from(feature_np_array):
    """
    Builds a network from the firing rates of the given feature_np_array for the
    input neurons and learns the weights to recognize the image through STDP.
    """
    in_p = create_spike_source_layer_from(feature_np_array).population
    out_p = sim.Population(1, sim.IF_curr_exp(i_offset=5))
    synapse = sim.STDPMechanism(weight=-0.2,
              timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                  A_plus=0.01,
                                                  A_minus=0.005),
              weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4))
    proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
    sim.run(500)
    return proj.get('weight', 'array')

def connect_layers(input_layer, output_population, weights, i_s, j_s, i_e, j_e,
                   k_out):
    """
    Connects a small recognizer output layer to a big spike source generator
    layer.
    """
    m = input_layer.shape[1]
    view_elements = []
    i = i_s
    while i < i_e:
        j = j_s
        while j < j_e:
            view_elements.append(m * i + j)
            j += 1
        i += 1

    sim.Projection(input_layer.population[view_elements],
                   output_population[[k_out]],
                   sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=weights))

def number_of_neurons_in(f_n, f_m, t_n, t_m, delta_i, delta_j):
    n = int((t_n - f_n) / delta_i) + ((t_n - f_n) % delta_i > 0) + 1
    m = int((t_m - f_m) / delta_j) + ((t_m - f_m) % delta_j > 0) + 1
    return (n, m)

def create_output_layer(input_layer, weights_tuple, delta_i, delta_j,
                        layer_name):
    """
    Builds a layer which creates an output layer which connects to the
    input_layer according to the given parameters.
    """
    weights = weights_tuple[0]
    f_n, f_m = weights_tuple[1]
    t_n, t_m = input_layer.shape
    # Determine how many output neurons can be connected to the input layer
    # according to the deltas
    overfull_n = (t_n - f_n) % delta_i > 0 # True for vertical overflow
    overfull_m = (t_m - f_m) % delta_j > 0 # True for horizontal overflow
    n, m = number_of_neurons_in(f_n, f_m, t_n, t_m, delta_i, delta_j)
    total_output_neurons = n * m
    print('Number of output neurons {} for size {}x{}'.format(\
                                            total_output_neurons, t_n, t_m))
    output_population = sim.Population(total_output_neurons, sim.IF_curr_exp(),
                                       label=layer_name)

    # Go through the lines of the image and connect input neurons to the
    # output layer according to delta_i and delta_j.
    k_out = 0
    i = 0
    while i + f_n <= t_n:
        j = 0
        while j + f_m <= t_m:
            connect_layers(input_layer, output_population, weights,
                           i, j, i + f_n, j + f_m, k_out)
            k_out += 1
            j += delta_j
        if overfull_m:
            connect_layers(input_layer, output_population, weights,
                           i, t_m - f_m, i + f_n, t_m, k_out)
            k_out += 1
        i += delta_i
    if overfull_n:
        j = 0
        while j + f_m <= t_m:
            connect_layers(input_layer, output_population, weights,
                           t_n - f_n, j, t_n, j + f_m, k_out)
            k_out += 1
            j += delta_j
        if overfull_m:
            connect_layers(input_layer, output_population, weights,
                           t_n - f_n, t_m - f_m, t_n, t_m, k_out)
            k_out += 1
    return Layer(output_population, (n, m))

def create_scale_invariance_layers_for(input_layer, weights_dict):
    """
    Takes an input spike source layer and a dict of weight arrays and creates an
    output layer for each separate feature.

    Returns a list of layers.
    """
    invariance_layers = [] # list of layers
    for layer_name, weights_tuple in weights_dict.items():
        invariance_layers.append(create_output_layer(input_layer, weights_tuple,
                                                     args.delta_i, args.delta_j,
                                                     layer_name))
    return invariance_layers

def create_corner_layer_for(input_layers):
    shape = input_layers[0].shape
    total_output_neurons = np.prod(shape)

    output_population = sim.Population(total_output_neurons, sim.IF_curr_exp(),
                                       label='corner')
    for layer in input_layers:
        sim.Projection(layer.population,
                       output_population,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=0.5, delay=0.5))

    return Layer(output_population, shape)


def copy_to_visualization(pos, ratio, feature_img, visualization_img,
                          f_n, f_m, t_n, t_m, n, m):
#    feature_img = np.array([\
#        [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [0,    0,    0,    0,    255,  255,  0,    0,    0,    0],
#        [0,    0,    0,    255,  255,  255,  255,  0,    0,    0],
#        [0,    0,    0,    255,  255,  255,  255,  0,    0,    0],
#        [0,    0,    0,    0,    255,  255,  0,    0,    0,    0],
#        [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [0,    0,    0,    0,    0,    0,    0,    0,    0,    0]])
    p_i = int(pos / m)
    p_j = pos % m
    delta_i = args.delta_i
    delta_j = args.delta_j
    start_i = delta_i * p_i
    start_j = delta_j * p_j
    if p_i == n - 1:
        start_i = n - f_n
    if p_j == m - 1:
        start_j = m - f_m
    rationated_feature_img = ratio * feature_img
    for i in range(f_n):
        for j in range(f_m):
            visualization_img[start_i + i][start_j + j] +=\
                rationated_feature_img[i][j]

def plot_weights(weights_dict):
    weight_panels = []
    for name, (weights, shape) in weights_dict.items():
        weight_panels.append(plt.Panel(weights.reshape(10, -1), cmap='gray',
                                       xlabel='{} Connection weights'.format(name),
                                       xticks=True, yticks=True))

    plt.Figure(*weight_panels).save('plots/weights_plot_blurred.png')


sim.setup()
# Take care of the weights of the basic feature recognizers
weights_dict = {}       # feature name string -> (weights, shape)
feature_imgs_dict = {}  # feature name string -> feature image
for training_img in plb.Path(args.feature_dir).iterdir():
    feature_np_array = cv2.imread(training_img.as_posix(), cv2.CV_8U)
    feature_imgs_dict[training_img.stem] = feature_np_array
    weights = recognizer_weights_from(feature_np_array)
    weights_dict[training_img.stem] = (weights, feature_np_array.shape)
sim.end()

if args.plot_weights:
    plot_weights(weights_dict)
    sys.exit(0)

# The training part is done. Go on with the "actual" simulation
sim.setup(threads=4)

file_extension = plb.Path(args.target_name).suffix

if file_extension == '.bag':
    target_stream = read_stream(args.target_name)
else:
    target_img = cv2.imread(args.target_name, cv2.CV_8U)

S1_layers = {} # input size -> list of S1 feature layers
C1_layers = {} # input size -> list of C1 layers
CORNER_layers = {} # input size -> list of Corner layers
for size in [0.4]:
    if file_extension == '.bag':
        resized_target_stream = resize_stream(target_stream, size)
        input_layer = create_spike_source_layer_from_stream(resized_target_stream)
    else:
        resized_target_np_array = cv2.resize(src=target_img, dsize=None,
                                             fx=size, fy=size,
                                             interpolation=cv2.INTER_AREA)
        print('resized target shape: ', resized_target_np_array.shape)

        # Create S1 layers for the current size
        input_layer = create_spike_source_layer_from(resized_target_np_array)

    print('input population size: ', input_layer.population.size)
    current_invariance_layers = create_scale_invariance_layers_for(\
                                                  input_layer, weights_dict)
    S1_layers[size] = current_invariance_layers

    # # Create C1 layers for the current size
    # C1_layers[size] = []
    # C1_subsampling_shape = (7, 7)
    # neuron_number = C1_subsampling_shape[0] * C1_subsampling_shape[1]
    # move_i, move_j = (6, 6)
    # C1_weight = 5
    # weights_tuple = (C1_weight * np.ones((neuron_number, 1)),
    #                  C1_subsampling_shape)
    # for S1_layer in current_invariance_layers:
    #     print('creating C1 output layer')
    #     C1_output_layer = create_output_layer(S1_layer, weights_tuple,
    #                            move_i, move_j, S1_layer.population.label)
    #     print('created layer')
    #     C1_layers[size].append(C1_output_layer)


    CORNER_layers[size] = [create_corner_layer_for(current_invariance_layers)]
# Keep these arrays for ease of recording and plotting
layer_collection = [S1_layers, CORNER_layers]
layer_names = ['S1', 'CORNER']

for i in range(len(layer_collection)):
    for layers in layer_collection[i].values():
        for layer in layers:
            layer.population.record('spikes')

print('========= Start simulation: {} ========='.format(sim.get_current_time()))
sim.run(2000)
print('========= Stop simulation: {} ========='.format(sim.get_current_time()))

## Start the visualization
#t_n, t_m = target_img.shape
#print('target shape: ', t_n, t_m)
#visualization_img = np.zeros( (t_n, t_m) )
#max_firing = 60
#for size, layers in S1_layers.items():
#    scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
#    for layer in layers:
#        print('layer :', layer.population.label)
#        out_data = layer.population.get_data().segments[0]
#        feature_label = layer.population.label
#        feature_img = feature_imgs_dict[feature_label]
#        print('feature img shape: ', feature_img.shape)
#        f_n, f_m = feature_img.shape
#        st_n, st_m = scaled_vis_img.shape
#        print('scaled vis shape: ', st_n, st_m)
##        n = m = int(np.sqrt(len(out_data.spiketrains)))
#        n, m = number_of_neurons_in(f_n, f_m, st_n, st_m,
#                                    args.delta_i, args.delta_j)
#        print(n, m, n * m, len(out_data.spiketrains))
#        for i in range(len(out_data.spiketrains)):
#            # each spiketrain corresponds to a layer S1 output neuron
#            copy_to_visualization(i, len(out_data.spiketrains[i]) / max_firing,
#                                  feature_img, scaled_vis_img,
#                                  f_n, f_m, st_n, st_m, n, m)
#    upscaled_vis_img = cv2.resize(src=scaled_vis_img, dsize=(t_m, t_n),
#                                  interpolation=cv2.INTER_CUBIC)
#    print('upscaled vis shape: ', upscaled_vis_img.shape)
#    visualization_img += upscaled_vis_img
#
#cv2.imwrite('{}_reconstruction.png'.format(plb.Path(args.target_name).stem),
#                                           visualization_img)

for i in range(2):
    for size, layers in layer_collection[i].items():
        spike_panels = []
        for layer in layers:
            out_data = layer.population.get_data().segments[0]
            spike_panels.append(plt.Panel(out_data.spiketrains,# xlabel='Time (ms)',
                                          xticks=True, yticks=True,
                                          xlabel='{}, {} scale layer'.format(\
                                                                layer.population.label, size)))
        plt.Figure(*spike_panels).save('plots/{}_{}_{}_scale.png'.format(\
                                                layer_names[i],
                                                plb.Path(args.target_name).stem,
                                                size))


sim.end()
