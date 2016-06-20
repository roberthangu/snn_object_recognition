#!/bin/ipython
import cv2
import numpy as np
import pyNN.utility.plotting as plt
import sys
import pathlib as plb
import argparse as ap
import pyNN.nest as sim

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
    return spike_source_layer

def recognizer_weights_from(feature_np_array):
    """
    Builds a network from the firing rates of the given feature_np_array for the
    input neurons and learns the weights to recognize the image through STDP.
    """
    in_p = create_spike_source_layer_from(feature_np_array)
    out_p = sim.Population(1, sim.IF_curr_exp(i_offset=5))
    synapse = sim.STDPMechanism(weight=-0.2,
              timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                  A_plus=0.01,
                                                  A_minus=0.005),
              weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4))
    proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
    sim.run(500)
    return proj.get('weight', 'array')

def connect_layers(input_layer, output_layer, weights, m, i_s, j_s, i_e, j_e,
                   k_out):
    """
    Connects a small recognizer output layer to a big spike source generator
    layer.
    """
    view_elements = []
    i = i_s
    while i < i_e:
        j = j_s
        while j < j_e:
            view_elements.append(m * i + j)
            j += 1
        i += 1
    sim.Projection(input_layer[view_elements],
                   output_layer[[k_out]],
                   sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=weights))

def number_of_S1_neurons(f_n, f_m, t_n, t_m, delta_i, delta_j):
    n = int((t_n - f_n) / delta_i) + ((t_n - f_n) % delta_i > 0) + 1
    m = int((t_m - f_m) / delta_j) + ((t_m - f_m) % delta_j > 0) + 1
    return (n, m)

def create_output_layer(input_layer, feature_shape, target_shape,
                        delta_i, delta_j, weights, layer_name):
    """
    Builds a layer which creates an output layer which connects to the
    input_layer according to the given parameters.
    """
    f_n, f_m = feature_shape
    t_n, t_m = target_shape
    # Determine how many output neurons can be connected to the input layer
    # according to the deltas
    overfull_n = (t_n - f_n) % delta_i > 0 # True for vertical overflow
    overfull_m = (t_m - f_m) % delta_j > 0 # True for horizontal overflow
    n, m = number_of_S1_neurons(f_n, f_m, t_n, t_m, delta_i, delta_j)
    total_output_neurons = n * m
    print('Number of output neurons {} for size {}x{}'.format(\
                                            total_output_neurons, t_n, t_m))
    output_layer = sim.Population(total_output_neurons, sim.IF_curr_exp(),
                                  label=layer_name)
    
    # Go through the lines of the image and connect input neurons to the
    # output layer according to delta_i and delta_j.
    k_out = 0
    i = 0
    while i + f_n <= t_n:
        j = 0
        while j + f_m <= t_m:
            connect_layers(input_layer, output_layer, weights,
                           t_m, i, j, i + f_n, j + f_m, k_out)
            k_out += 1
            j += delta_j
        if overfull_m:
            connect_layers(input_layer, output_layer, weights,
                           t_m, i, t_m - f_m, i + f_n, t_m, k_out)
            k_out += 1
        i += delta_i
    if overfull_n:
        j = 0
        while j + f_m <= t_m:
            connect_layers(input_layer, output_layer, weights,
                           t_m, t_n - f_n, j, t_n, j + f_m, k_out)
            k_out += 1
            j += delta_j
        if overfull_m:
            connect_layers(input_layer, output_layer, weights,
                           t_m, t_n - f_n, t_m - f_m, t_n, t_m, k_out)
            k_out += 1
    return output_layer

def create_invariance_layers_for_all_features_for(input_layer, target_shape,
                                                  weights_dict):
    """
    Takes an input spike source layer and a dict of weight arrays and creates an
    output layer for each separate feature.
    """
    invariance_layers = []
    for layer_name, (weights, feature_shape) in weights_dict.items():
        invariance_layers.append(create_output_layer(input_layer, feature_shape,
                                                     target_shape, args.delta_i,
                                                     args.delta_j, weights,
                                                     layer_name))
    return invariance_layers
 
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
target_img = cv2.imread(args.target_name, cv2.CV_8U)
feature_layers = {} # input size -> feature layers
for size in [1, 0.71, 0.5, 0.35, 0.25]:
    resized_target_np_array = cv2.resize(src=target_img, dsize=None,
                                         fx=size, fy=size,
                                         interpolation=cv2.INTER_AREA)
    print('resized target shape: ', resized_target_np_array.shape)
    input_layer = create_spike_source_layer_from(resized_target_np_array)
    print('input population size: ', input_layer.size)
    feature_layers[size] = create_invariance_layers_for_all_features_for(\
                                                  input_layer,
                                                  resized_target_np_array.shape,
                                                  weights_dict)
for layers in feature_layers.values():
    for layer in layers:
        layer.record('spikes')

print('========= Start simulation =========')
sim.run(300)
print('========= Stop  simulation =========')

# Start the visualization
t_n, t_m = target_img.shape
print('target shape: ', t_n, t_m)
visualization_img = np.zeros( (t_n, t_m) )
max_firing = 60
for size, layers in feature_layers.items():
    scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
    for layer in layers:
        print('layer :', layer.label)
        out_data = layer.get_data().segments[0]
        feature_label = layer.label
        feature_img = feature_imgs_dict[feature_label]
        print('feature img shape: ', feature_img.shape)
        f_n, f_m = feature_img.shape
        st_n, st_m = scaled_vis_img.shape
        print('scaled vis shape: ', st_n, st_m)
#        n = m = int(np.sqrt(len(out_data.spiketrains)))
        n, m = number_of_S1_neurons(f_n, f_m, st_n, st_m,
                                    args.delta_i, args.delta_j)
        print(n, m, n * m, len(out_data.spiketrains)) 
        for i in range(len(out_data.spiketrains)):
            # each spiketrain corresponds to a layer S1 output neuron
            copy_to_visualization(i, len(out_data.spiketrains[i]) / max_firing,
                                  feature_img, scaled_vis_img,
                                  f_n, f_m, st_n, st_m, n, m)
    upscaled_vis_img = cv2.resize(src=scaled_vis_img, dsize=(t_m, t_n),
                                  interpolation=cv2.INTER_CUBIC)
    print('upscaled vis shape: ', upscaled_vis_img.shape)
    visualization_img += upscaled_vis_img
        
cv2.imwrite('{}_reconstruction.png'.format(plb.Path(args.target_name).stem),
                                           visualization_img)

#for size, layers in feature_layers.items():
#    spike_panels = []
#    for layer in layers:
#        out_data = layer.get_data().segments[0]
#        spike_panels.append(plt.Panel(out_data.spiketrains,# xlabel='Time (ms)',
#                                      xticks=True, yticks=True,
#                                      xlabel='{}, {} scale layer'.format(\
#                                                            layer.label, size)))
#    plt.Figure(*spike_panels).save('plots/{}_{}_scale.png'.format(\
#                                            plb.Path(args.target_name).stem,
#                                            size))



sim.end()
