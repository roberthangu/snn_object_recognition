import numpy as np
import pyNN.nest as sim
import cv2
import pathlib as plb
import time
try:
    import stream
except ImportError:
    pass

class Layer:
    """
    Represents a layer in the network architecture.

    Attributes:

        `population`: The pyNN neuron population of the layer

        `shape`:      The shape of the layer as a tuple of (rows, cols)

        `current_spike_counts`: The spike counts which are generated from one
                                simulation run to the next
    """

    def __init__(self, population, shape):
        self.population = population
        self.shape = shape
        self.current_spike_counts = [0] * population.size

    def update_spike_counts(self):
        """
        Updates the spike counts inside self.current_spike_counts to reflect the
        latest simulation advancement. That is, to store the spike counts from
        the previous simulation to the current one.
        """
        spike_counts = self.population.get_spike_counts()
        for i in range(self.population.size):
            self.current_spike_counts[i] =\
                spike_counts[self.population[i]] - self.current_spike_counts[i]

def create_spike_source_layer_from(source_np_array):
    """
    For a given image returns a layer of spike source neurons to encode the
    image intensities in spikes. The size of the spike source layer is the
    number of pixels in the image.
    """
    reshaped_array = source_np_array.ravel()
    rates = []
    for rate in reshaped_array:
        rates.append(int(rate / 4))
    spike_source_layer = sim.Population(size=len(rates),
                                   cellclass=sim.SpikeSourcePoisson(rate=rates))
    return Layer(spike_source_layer, source_np_array.shape)

def create_spike_source_layer_from_stream(stream):
    nNeurons = stream.shape[0] * stream.shape[1]
    spike_times = []

    for neuron in range(nNeurons):
        spike_times.append([])
    for event in stream.events:
        if not event.polarity:
            # we only consider ON events
            pass
        nIdx = event.x * stream.shape[0] + event.y
        spike_times[nIdx].append(event.ts)

    spike_source_layer = sim.Population(size=len(spike_times),
                                   cellclass=sim.SpikeSourceArray(spike_times=spike_times))
    return Layer(spike_source_layer, stream.shape)

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

def create_output_layer(input_layer, weights_tuple, delta_i, delta_j,
                        layer_name, refrac):
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
    n = int((t_n - f_n) / delta_i) + ((t_n - f_n) % delta_i > 0) + 1
    m = int((t_m - f_m) / delta_j) + ((t_m - f_m) % delta_j > 0) + 1
    total_output_neurons = n * m
#    print('Number of output neurons {} for size {}x{}'.format(\
#                                            total_output_neurons, t_n, t_m))
    print('Layer:', layer_name)
    output_population = sim.Population(total_output_neurons,
                                       sim.IF_curr_exp(tau_refrac=refrac),
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

def create_all_feature_layers_for(input_layer, weights_dict, args):
    """
    Takes an input spike source layer and a dict of weight arrays and creates an
    output layer for each separate feature. Uses the commandline arguments to
    determine the horizontal and vertical deltas as well as the neuron
    refractory period.

    Arguments:

        `input_layer`: The spike source input layer

        `weights_dict`: A dictionary containing for each feature name a pair
                        of a weight list and its shape

        `args`: The commandline arguments object. It uses the deltas and the
                neuron refractory period from it

    Returns:

        A list of S1 layers.
    """
    invariance_layers = [] # list of layers
    for layer_name, weights_tuple in weights_dict.items():
        invariance_layers.append(create_output_layer(input_layer, weights_tuple,
                                                     args.delta_i, args.delta_j,
                                                     layer_name, args.refrac_s1))
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
                       sim.StaticSynapse(weight=1., delay=0.5))

    return Layer(output_population, shape)

def train_weights(feature_dir):
    """
    Trains the basic recognizer weights such that they respond to the features
    found in the directory feature_dir. This function runs a sim.start() -
    sim.end() "session".

    Arguments:

        `feature_dir`: The directory where the features are stored as images

    Returns:

        A pair of weight and feature image dictionaries of the following type:
            weights_dict        :: feature name string -> (weights, shape)
            feature_imgs_dict   :: feature name string -> feature image
    """
    sim.setup()
    weights_dict = {}       # feature name string -> (weights, shape)
    feature_imgs_dict = {}  # feature name string -> feature image
    for training_img in plb.Path(feature_dir).iterdir():
        feature_np_array = cv2.imread(training_img.as_posix(), cv2.CV_8U)
        feature_imgs_dict[training_img.stem] = feature_np_array
        weights = recognizer_weights_from(feature_np_array)
        weights_dict[training_img.stem] = (weights, feature_np_array.shape)
    sim.end()
    return (weights_dict, feature_imgs_dict)

def create_input_layers_for_scales(target, scales, is_bag=False):
    """
    Creates for a given target image and a list of scales a list of input spike
    layers one for each size.

    Parameters:

        `target`: The target image or stream from which to create the input
                  layers

        `scales`: A list of scales for which to create input layers

        `is_bag`: If set to True, the passed target will be treated as a rosbag,
                  otherwise as an image

    Returns:

        A list of input spike layers created from the given target, one for each
        scale
    """
    input_layers = {}
    neuron_count = 0
    for size in scales:
        if is_bag:
            resized_target = stream.resize_stream(target, size)
            input_layer = create_spike_source_layer_from_stream(resized_target)
        else:
            resized_target = cv2.resize(src=target, dsize=None,
                                        fx=size, fy=size,
                                        interpolation=cv2.INTER_AREA)
#            print('resized target shape: ', resized_target.shape)
            t1 = time.clock()
            input_layer = create_spike_source_layer_from(resized_target)
            print('Input layer creation for scale {} took {} s'.format(size,
                                                             time.clock() - t1))
            print('Input layer for scale {} has {} neurons'.format(size,
                                   input_layer.shape[0] * input_layer.shape[1]))
        input_layers[size] = input_layer
    return input_layers

def create_S1_layers(input_layers_dict, weights_dict, args):
    """
    Creates S1 layers for the given input layers. It creates for each input
    layer a S1 layer for each feature in the weights_dict

    Parameters:

        `input_layers_dict`: A dictionary of input layers for which to create
                             the S1 layers for all features in the weights_dict.
                             It stores for each size the corresponding input
                             layer

        `weights_dict`: A dictionary containing for each feature name a pair
                        of a weight list and its shape

        `args`: The commandline arguments

    Returns:

        A dictionary containing for each size of the target a list of S1 layers
    """
    S1_layers = {} # input size -> list of S1 feature layers
    for size, input_layer in input_layers_dict.items():
        print('Create S1 layers for size', size)
        neuron_count = 0
        t1 = time.clock()
        current_invariance_layers = create_all_feature_layers_for(input_layer,
                                                             weights_dict, args)
        print('S1 layer creation for scale {} took {} s'.format(size,
                                                            time.clock() - t1))
        S1_layers[size] = current_invariance_layers

        for layer in current_invariance_layers:
            neuron_count += layer.shape[0] * layer.shape[1]
        print('S1 layers at scale {} have {} neurons'.format(size, neuron_count))
    return S1_layers

def create_C1_layers(S1_layers_dict, refrac_c1):
    """
    Creates C1 layers for each of the given S1 layers

    Arguments:

        `S1_layers_dict`: A dictionary containing for each size of the input
                          image a list of S1 layers, for each feature one

        `refrac_c1`:

    Returns:

        A dictionary containing for each size of S1 layers a list of C1 layers
    """
    C1_layers = {} # input size -> list of C1 layers
    for size, S1_layers in S1_layers_dict.items():
        neuron_count = 0
        C1_layers[size] = []
        C1_subsampling_shape = (7, 7)
        neuron_number = C1_subsampling_shape[0] * C1_subsampling_shape[1]
        move_i, move_j = (6, 6)
        C1_weight = 5
        weights_tuple = (C1_weight * np.ones((neuron_number, 1)),
                         C1_subsampling_shape)
        t1 = time.clock()
        for S1_layer in S1_layers:
            C1_output_layer = create_output_layer(S1_layer, weights_tuple,
                                   move_i, move_j, S1_layer.population.label,
                                   refrac_c1)
            C1_layers[size].append(C1_output_layer)
            neuron_count += C1_output_layer.shape[0] * C1_output_layer.shape[1]
        print('C1 layer creation for scale {} took {} s'.format(size,
                                                            time.clock() - t1))
        print('C1 layers at scale {} have {} neurons'.format(size, neuron_count))
    return C1_layers
