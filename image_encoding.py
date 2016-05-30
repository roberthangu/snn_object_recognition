import pyNN.nest as sim
import cv2
import numpy as np
import pyNN.utility.plotting as plt
import sys
import pathlib as plb

training_dir = sys.argv[1]
#verif_img = sys.argv[2]

# Takes an image and computes the rates of a input neurons accorning to this
# image
def rates_from_img(source_img):
    img = np.array(cv2.imread(source_img, cv2.CV_8U)).ravel()
    rates = [] 
    # take x * 4 as the firing rate
    for i in map(lambda x: int(x / 4), img):
        rates.append(i)
    return rates

# Returns lists of rates of input neurons accorning to these images
def rate_lists_from_imgs(source_imgs):
    rate_lists = []
    for source_img in source_imgs:
        rate_lists.append(rates_from_img(source_img))
    return rate_lists

# Builds a network from a list of firing rates for the input neurons and a list
# of initial weights for the connections to the output neuron. If no weights
# list is specified, it learns the weights through STDP.
def build_network(firing_rates=None, initial_weights=None):
    in_p = None
    if firing_rates == None:
        in_p = sim.Population(size=len(initial_weights),
                              cellclass=sim.SpikeSourcePoisson(rate=0))
    else:
        in_p = sim.Population(size=len(firing_rates),
                              cellclass=sim.SpikeSourcePoisson(rate=firing_rates))
    # Determine the type of synapse depending if learning is desired or not
    synapse = None
    # Use the current offset only if training the weights with STDP
    i_offset = 0
    if initial_weights == None:
        synapse = sim.STDPMechanism(weight=-0.4,
                  timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                      A_plus=0.01,
                                                      A_minus=0.005),
                  weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4))
        i_offset = 5
    else:
        synapse = sim.StaticSynapse(weight=initial_weights)
    out_p = sim.Population(1, sim.IF_curr_exp(i_offset=i_offset))
    proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
    return {'in_p': in_p, 'out_p': out_p, 'proj': proj}

# Returns an array of networks with the given firing rate arrays and the initial
# weights.
# If initial_weights_array in not None, it should have the same size as
# firing_rates_array
def build_networks_from(firing_rates_array, initial_weights_array=None):
    networks = []
    for i in range(len(firing_rates_array)):
        initial_weights = None
        if initial_weights_array != None:
            initial_weights = initial_weights_array[i]
        network = build_network(firing_rates_array[i], initial_weights)
        networks.append(network)
    return networks

# Returns a list of network weights to recognize a list of given images through
# training
def weight_lists_from_imgs(source_imgs):
    networks = build_networks_from(rate_lists_from_imgs(source_imgs))
    sim.run(500)
    weight_arrays = []
    for network in networks:
        weight_arrays.append(network['proj'].get('weight', 'array'))
    return weight_arrays

sim.setup()
training_imgs = []
for training_img in plb.Path('img').iterdir():
    training_imgs.append(training_img.as_posix())
weight_arrays = weight_lists_from_imgs(training_imgs)
sim.end()
sim.setup()
networks = build_networks_from([None] * len(training_imgs), weight_arrays)

print('Built networks')

panels = []
for network in networks:
    network['out_p'].record('spikes')
for verif_img in training_imgs:
    # set the input spikes of every network to the current image
    input_rates = rates_from_img(verif_img)
    print(input_rates)
    for network in networks:
        network['in_p'].set(rate=input_rates)
    print('Starting simulation with image', verif_img)
    sim.run(500)

for weight_array in weight_arrays:
    panels.append(plt.Panel(weight_array.reshape(10, -1), cmap='gray',
                            xlabel='Connection weights',
                            xticks=True, yticks=True))

for network in networks:
    out_data = network['out_p'].get_data(clear=True).segments[0]
    panels.append(plt.Panel(out_data.spiketrains, xlabel='Time (ms)', xticks=True,
                            yticks=True, ylabel='out'))

plt.Figure(*panels).save('image_encoding.png')

#plt.Figure(
#    plt.Panel(weight_array.reshape(10, -1), cmap='gray',
#              xlabel='Connection weights',
#              xticks=True, yticks=True),
#    plt.Panel(out_data.spiketrains, xlabel='Time (ms)', xticks=True,
#        yticks=True, ylabel='out'),
#    plt.Panel(out_data.filter(name='v')[0], xlabel='Time (ms)', xticks=True,
#        yticks=True, ylabel='Voltage (out)')
#).save('image_encoding.png')

sim.end()
