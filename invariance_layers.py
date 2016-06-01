import pyNN.nest as sim
import cv2
import numpy as np
import pyNN.utility.plotting as plt
import sys
import pathlib as plb

def create_spike_source_layer_from(source_np_array):
    """
    For a given image returns a layer of spike source neurons to encode the
    image intensities in spikes. Acts as an encoder.
    """
    reshaped_array = source_np_array.ravel()
    spike_source_layer = sim.Population(size=len(reshaped_array),
                                   cellclass=sim.SpikeSourcePoisson(\
                                     rate=lambda i: int(reshaped_array[i] / 4)))
    return spike_source_layer

def recognizer_weights_from(feature_np_array):
    """
    Builds a network from the firing rates of the given feature_np_array for the input
    neurons and learns the weights to recognize the image through STDP.
    """
    sim.setup()
    in_p = create_spike_source_layer_from(feature_np_array)
    out_p = sim.Population(1, sim.IF_curr_exp(i_offset=5))
    synapse = sim.STDPMechanism(weight=-0.4,
              timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                  A_plus=0.01,
                                                  A_minus=0.005),
              weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4))
    proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
    sim.run(500)
    weights = proj.get('weight', 'array')
    sim.stop()
    return weights
    
class PositionInvarianceLayer:
    """
    Represents a position invariance layer to detect a given feature in the
    given image.
    
    Arguments:
        `feature_img`:
            The image with the feature to be recognized

        `target_img`:
            The "big" image in which to detect the feature

        `delta_n`:
            The vertical space between the connected feature detectors

        `delta_m`:
            The horizontal space between the connected feature detectors
    """

    input_layer = None
    output_layer = None
    projection = None

    def __init__(self, feature_img, target_img, delta_n=1, delta_m=1):
        feature_np_array = np.array(cv2.imread(feature_img, cv2.CV_8U))
        target_np_array = np.array(cv2.imread(target_img, cv2.CV_8U))
        # Take care of the weights of the basic feature recognizers
        weights = recognizer_weights_from(feature_np_array)
        self.input_layer = create_spike_source_layer_from(target_np_array)

        # Determine how many output neurons can be connected to the input layer
        # according to the deltas
        f_n, f_m = feature_np_array.shape
        t_n, t_m = target_np_array.shape
        m = (t_m - f_m) / delta_m + ((t_m - f_m) % delta_m > 1) + 1
        n = (t_n - f_n) / delta_n + ((t_n - f_n) % delta_n > 1) + 1
        

        # Go through the lines of the image and connect output neurons to the
        # input layer according to delta_h and delta_v.
        
        feature_detectors = [BasicFeatureDetector(feature_img)] * 10000
        weigths = recognizer_weights_from(feature_img)
