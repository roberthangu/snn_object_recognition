import numpy as np
import pyNN.nest as sim
import pyNN.space as space
import cv2
import pathlib as plb
import time
import common as cm
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
        self.old_spike_counts = [0] * population.size

    def update_spike_counts(self):
        """
        Updates the spike counts inside self.current_spike_counts to reflect the
        latest simulation advancement. That is, to store the spike counts from
        the previous simulation to the current one.
        """
        spike_counts = self.population.get_spike_counts()
        for i in range(self.population.size):
            self.current_spike_counts[i] =\
                spike_counts[self.population[i]] - self.old_spike_counts[i]
            self.old_spike_counts[i] = spike_counts[self.population[i]]

def set_i_offsets(layer, source_np_array):
    """
    Sets the i_offset for the input neuons according to the pixel values of the
    input image.

    Parameters:
        `layer`: The layer for which to set the i_offsets

        `source_np_array`: The array with the pixel intensities from which to
                           set the i_offsets
    """
    layer.population.set(i_offset=list(map(lambda x: x / 255 * .6 + .75,
                                           source_np_array.ravel())))

def set_spike_source_layer_rates(layer, source_np_array):
    """
    Sets the firing rates of the already created spike source layer to the
    given source_np_array. This is also a helper function of
    create_spike_source_layer_from().
    """
    layer.population.set(rate=list(map(lambda x: x / 4,
                                       source_np_array.ravel())))

def create_empty_spike_source_layer_with_shape(shape):
    """
    Creates a spike source layer with the given shape and its firing rates set
    to zero. This is also a helper function of create_spike_source_layer_from().
    """
    spike_source_layer = sim.Population(size=shape[0] * shape[1],
                                   cellclass=sim.SpikeSourcePoisson(rate=0))
    return Layer(spike_source_layer, shape)

def create_spike_source_layer_from(source_np_array):
    """
    For a given image returns a layer of spike source neurons to encode the
    image intensities in spikes. The size of the spike source layer is the
    number of pixels in the image.
    """
    layer = create_empty_spike_source_layer_with_shape(source_np_array.shape)
    set_spike_source_layer_rates(layer, source_np_array)
    return layer

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

def create_output_layer(input_layer, weights_tuple, delta, layer_name, refrac):
    """
    Builds a layer which connects to the input_layer according to the given
    parameters.
    """
    weights = weights_tuple[0]
    f_n, f_m = weights_tuple[1]
    t_n, t_m = input_layer.shape
    # Determine how many output neurons can be connected to the input layer
    # according to the deltas
    overfull_n = (t_n - f_n) % delta > 0 # True for vertical overflow
    overfull_m = (t_m - f_m) % delta > 0 # True for horizontal overflow
    n = int((t_n - f_n) / delta) + ((t_n - f_n) % delta > 0) + 1
    m = int((t_m - f_m) / delta) + ((t_m - f_m) % delta > 0) + 1
    total_output_neurons = n * m
#    print('Number of output neurons {} for size {}x{}'.format(\
#                                            total_output_neurons, t_n, t_m))
    print('Layer:', layer_name)
    output_population = sim.Population(total_output_neurons,
                                       sim.IF_curr_exp(tau_refrac=refrac),
                                       structure=space.Grid2D(aspect_ratio=m/n),
                                       label=layer_name)

    # Go through the lines of the image and connect input neurons to the
    # output layer according to delta
    k_out = 0
    i = 0
    while i + f_n <= t_n:
        j = 0
        while j + f_m <= t_m:
            connect_layers(input_layer, output_population, weights,
                           i, j, i + f_n, j + f_m, k_out)
            k_out += 1
            j += delta
        if overfull_m:
            connect_layers(input_layer, output_population, weights,
                           i, t_m - f_m, i + f_n, t_m, k_out)
            k_out += 1
        i += delta
    if overfull_n:
        j = 0
        while j + f_m <= t_m:
            connect_layers(input_layer, output_population, weights,
                           t_n - f_n, j, t_n, j + f_m, k_out)
            k_out += 1
            j += delta
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
    feature_layers = [] # list of layers
    for layer_name, weights_tuple in weights_dict.items():
        feature_layers.append(create_output_layer(input_layer, weights_tuple,
                                                     args.delta, layer_name,
                                                     args.refrac_s1))
    return feature_layers

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

def change_rates_for_scales(input_layers, target):
    """
    Sets the rates of the given input layers according to the values of the
    target image.
    """
    for size, layer in input_layers.items():
        resized_target = cv2.resize(src=target, dsize=None,
                                    fx=size, fy=size,
                                    interpolation=cv2.INTER_AREA)
        set_spike_source_layer_rates(layer, resized_target)

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
    bag_input_layers = {}
    t_n = target.shape[0]
    t_m = target.shape[1]
    for size in scales:
        if is_bag:
            resized_target = stream.resize_stream(target, size)
            bag_input_layers[size] =\
                 create_spike_source_layer_from_stream(resized_target)
        else:
            t1 = time.clock()
            input_layer = create_empty_spike_source_layer_with_shape(\
                                       ( round(t_n * size), round(t_m * size) ))
            print('Input layer creation for scale {} took {} s'.format(size,
                                                             time.clock() - t1))
            print('Input layer for scale {} has {} neurons'.format(size,
                                   input_layer.shape[0] * input_layer.shape[1]))
            input_layers[size] = input_layer
    if is_bag:
        return bag_input_layers
    change_rates_for_scales(input_layers, target)
    return input_layers

def create_gabor_input_layers_for_scales(target, scales):
    """
    Creates input layers from the given image by using gabor filters in four
    orientations.

    Parameters:
        `target`: The target image from which to compute the gabor filters and
                  create the input layers

        `scales`: A list of the scales for which to create input layers

    Returns:
        A dictionary which contains for each scale a list of four input layers,
        one for each orientation
    """
    input_layers = {}
    for size in scales:
        print('Creating input layers for size', size)
        resized_target = cv2.resize(src=target, dsize=None, fx=size, fy=size,
                                    interpolation=cv2.INTER_AREA)
        current_feature_layers = []
        for name, edge_img in cm.get_gabor_edges(resized_target).items():
            n, m = resized_target.shape
            layer = Layer(sim.Population(n * m, sim.IF_curr_exp()), (n, m))
            layer.population.label = name
            set_i_offsets(layer, edge_img)
            current_feature_layers.append(layer)
        input_layers[size] = current_feature_layers
    return input_layers

def create_gabor_S1_layers(input_layers_dict):
    """
    Create for each input layer a S1 layer which has a one-to-one connection to
    it.

    Parameters:
        `input_layers_dict`: A dictionary containing for each size a list of
                             input layers, for each feature one

    Returns:
        A dictionary containing for each size a list with the S1 layers
    """
    S1_layers = {}
    for size, layers in input_layers_dict.items():
        current_layers = []
        for input_layer in layers:
            layer_name = input_layer.population.label
            print('Create S1 layer for size', size, 'feature', layer_name)
            new_layer = Layer(sim.Population(input_layer.population.size, 
                                             sim.IF_curr_exp(),
                                             label=layer_name),
                             input_layer.shape)
            print('Creating projection')
            sim.Projection(input_layer.population, new_layer.population,
                           sim.OneToOneConnector(),
                           sim.StaticSynapse(weight=5))
            current_layers.append(new_layer)
        S1_layers[size] = current_layers
    return S1_layers

def create_cross_layer_inhibition(layers_dict):
    """
    Creates inhibitory connections between the given feature layers for each
    size to allow only the spikes of the strongest feature to be propagated
    further

    Parameters:
        
        `layers_dict`: A dictionary of layers of the type size -> list of layers
    """
    def inhibitory_connect(layers, source, dest1, dest2, dest3, weight):
        sim.Projection(layers[source].population, layers[dest1].population,
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=weight))
        sim.Projection(layers[source].population, layers[dest2].population,
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=weight))
        sim.Projection(layers[source].population, layers[dest3].population,
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=weight))

    print('Create inhibitory connections')
    for size, layers in layers_dict.items():
        print('Create cross layer inhibiton for size', size)
        inhibitory_connect(layers, 0, 1, 2, 3, -50)
        inhibitory_connect(layers, 1, 0, 2, 3, -50)
        inhibitory_connect(layers, 2, 0, 1, 3, -50)
        inhibitory_connect(layers, 3, 0, 1, 2, -50)

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

def create_local_inhibition(layers_dict):
    """
    Creates local inhibitory connections from a neuron to its neighbors in an
    area of a fixed distance. The latency of its neighboring neurons decreases
    linearly with the distance from the spike from 15% to 5%, as described in
    Masquelier's paper. Here we assumed that a weight of -10 inhibits the
    neuron completely and took that as a starting point.
    """
    for size, layers in layers_dict.items():
        print('Create local inhibition for size', size)
        for layer in layers:
            sim.Projection(layer.population, layer.population,
                sim.DistanceDependentProbabilityConnector('d < 5',
                    allow_self_connections=False),
                sim.StaticSynapse(weight='.25 * d - 1.75'),
                space=space.Space(axes='xy')) 

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
        move = 6
        C1_weight = 5
        weights_tuple = (C1_weight * np.ones((neuron_number, 1)),
                         C1_subsampling_shape)
        t1 = time.clock()
        for S1_layer in S1_layers:
            C1_output_layer = create_output_layer(S1_layer, weights_tuple,
                                   move, S1_layer.population.label, refrac_c1)
            C1_layers[size].append(C1_output_layer)
            neuron_count += C1_output_layer.shape[0] * C1_output_layer.shape[1]
        print('C1 layer creation for scale {} took {} s'.format(size,
                                                            time.clock() - t1))
        print('C1 layers at scale {} have {} neurons'.format(size, neuron_count))
    return C1_layers
