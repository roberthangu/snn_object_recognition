#!/bin/ipython
import numpy as np
import cv2
import pyNN.nest as sim
import pathlib as plb
import time

import common as cm
import network as nw
import visualization as vis

args = cm.parse_args()

# Train weights
weights_dict, feature_imgs_dict = nw.train_weights(args.feature_dir)

# Open the video capture and writer objects
cap = cv2.VideoCapture(args.target_name)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter('video_S1_reconstructions/{}.avi'.format(\
                             plb.Path(args.target_name).stem),
                         int(cap.get(cv2.CAP_PROP_FOURCC)),
                         cap.get(cv2.CAP_PROP_FPS), (cap_width, cap_height),
                         isColor=False)

sim.setup(threads=4, spike_precision='on_grid')

# Set up the network
layer_collection = {} # layer name -> dict of S1 layers of type
                      # 'scale -> layer list'

layer_collection['input'] = nw.create_input_layers_for_scales(\
                                np.zeros((cap_height, cap_width)), args.scales)
layer_collection['S1'] = nw.create_S1_layers(layer_collection['input'],
                                             weights_dict, args)
# We build only the S1 layer for the moment, to speed up the simulation time

for layers in layer_collection['S1'].values():
    for layer in layers:
        layer.population.record('spikes')

# The actual frame-by-frame simulation and input neuron updating
t1 = time.clock()
#for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
for i in range(args.frames):
    t2 = time.clock()
    img = cap.read()[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # It's assumed that the image is already filtered, so no filtering is
    # required
    nw.change_rates_for_scales(layer_collection['input'], img)
    sim.run(50)

    # Refresh the current spike counts
    for layers in layer_collection['S1'].values():
        for layer in layers:
            layer.update_spike_counts()

    reconstructed_img = vis.create_S1_feature_image(img, layer_collection,
                                                    feature_imgs_dict, args)[1]
    reconstructed_img = cv2.convertScaleAbs(reconstructed_img)
    writer.write(reconstructed_img)
    print('Frame', i, 'took', time.clock() - t2, 's to finish')
print('Processing', args.frames, 'frames took', time.clock() - t1, 's')

cap.release()
writer.release()

sim.end()
