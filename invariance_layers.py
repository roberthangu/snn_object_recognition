import pyNN.nest as sim
import cv2
import numpy as np
import pyNN.utility.plotting as plt
import sys
import pathlib as plb

def rates_from_img(source_img):
    """
    Takes an image and computes the rates of a input neurons accorning to this
    image
    """
    img = np.array(cv2.imread(source_img, cv2.CV_8U)).ravel()
    rates = [] 
    # take x * 4 as the firing rate
    for i in map(lambda x: int(x / 4), img):
        rates.append(i)
    return rates

def build_and_train_network(firing_rates):
    """
    Builds a network from the given firing_rates for the input neurons and
    learns the weights to recognize the image through STDP.
    """
    in_p = sim.Population(size=len(firing_rates),
                          cellclass=sim.SpikeSourcePoisson(rate=firing_rates))
    out_p = sim.Population(1, sim.IF_curr_exp(i_offset=5))
    synapse = sim.STDPMechanism(weight=-0.4,
              timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                  A_plus=0.01,
                                                  A_minus=0.005),
              weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4))
    proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
    return {'in_p': in_p, 'out_p': out_p, 'proj': proj}
    
def weight_list_from_img(source_img):
    """
    Returns network weights to recognize the given image through training
    """
    network = build_and_train_network(rates_from_img(source_img))
    sim.run(500)
    return network['proj'].get('weight', 'array')

def create_spike_source_layer_from(target_img):
    img = np.array(cv2.imread(target_img, cv2.CV_8U)).ravel()
    spike_source_layer = sim.Population(size=len(img),
                            cellclass=sim.SpikeSourcePoisson(rate=lambda i: int(img[i] / 4)))
    return spike_source_layer

class BasicFeatureDetector():
    """
    Represents a basic feature detector
    """

    in_p = None
    out_p = None
    proj = None

    def __init__(self, input_image, preffered_size=(10, 10),
                 initial_weights=None):
        """
        Gets an input image as a parameter from which it learns the feature.
        The final recognizer will detect features in the given size
        """
        
        # TODO: In the future do something with the preffered_size to create a
        # detector with this size
        
    def build_network(static_weights):
        """
        Builds a network from a list of static weights.
        """
        in_p = sim.Population(size=len(static_weights),
                              cellclass=sim.SpikeSourcePoisson(rate=0))
        out_p = sim.Population(1, sim.IF_curr_exp())
        synapse = sim.StaticSynapse(weight=static_weights)
        proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
        return {'in_p': in_p, 'out_p': out_p, 'proj': proj}

class PositionInvarianceLayer():
    def __init__(self, feature_img, target_img, delta_x=1, delta_y=1):
        feature_detectors = [BasicFeatureDetector(feature_img)] * 10000
        weigths = weight_list_from_img(feature_img)
    # Use a connector to provide the input images as poisson source neurons
    #def connect_to_encoded_img(img_encoder):
