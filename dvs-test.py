from mpl_toolkits.mplot3d import Axes3D
import pathlib as plb
import matplotlib.pyplot as plt
import pickle
import numpy as np
import stream
import pyNN.nest as sim

import common as cm
import network as nw


args = cm.parse_args()
weights_dict, feature_imgs_dict = nw.train_weights(args.feature_dir)

# The training part is done. Go on with the "actual" simulation
sim.setup(threads=4)

file_extension = plb.Path(args.target_name).suffix
filename = plb.Path(args.target_name).stem

target = stream.read_stream(args.target_name)
size = [1.]
S1_layers = nw.create_S1_layers(target, weights_dict, [1.], args,
                                is_bag=True)

# NOTE: Since in your original code you're using only size 1 in creating the
# corner layers, I don't create an extra dictionary to store only that one
# layer. Hence the `corner_layer` variable.
corner_layer = nw.create_corner_layer_for(S1_layers[1])

layer_collection = {'S1' : S1_layers,
                    'corner': corner_layer}

stimuli_duration = 0
if file_extension == '.bag':
    stimuli_duration = target.duration

print('========= Start simulation: {} ========='.format(sim.get_current_time()))
sim.run(stimuli_duration + 300)
print('========= Stop simulation: {} ========='.format(sim.get_current_time()))


# visualize spatiotemporal spiketrain
def extract_spatiotemporal_spiketrain(size, layer_name, spiketrain, shape):
    x = []
    y = []
    times = []
    for populationIdx, neuron in enumerate(spiketrain):
        imageIdx = [populationIdx / shape[0], populationIdx % shape[0]]
        for spike in neuron:
            x.append(imageIdx[0])
            y.append(imageIdx[1])
            times.append(spike)
    return [x, y, times]

allSpatioTemporal = []
for size, layers in S1_layers.items():
    for layer in layers:
        out_data = layer.population.get_data().segments[0]
        allSpatioTemporal.append(extract_spatiotemporal_spiketrain(size, layer.population.label,
                                                                   out_data.spiketrains,
                                                                   target.shape))
pickle.dump(allSpatioTemporal, open("results/spatiotemporal_{}.p".format(filename), "wb"))

max_spike_rate = 60. / 300. # mHz
max_firing = max_spike_rate * (stimuli_duration + 300.)
if args.reconstruct_s1_img:
    vis_img = reconstruct_image(max_firing, target.shape, S1_layers, feature_imgs_dict)
    cv2.imwrite('{}_reconstruction.png'.format(filename), vis_img)

# Plot the spike trains of both neuron layers
for layer_name, layer_dict in layer_collection.items():
    for size, layers in layer_dict.items():
        spike_panels = []
        for layer in layers:
            out_data = layer.population.get_data().segments[0]
            dump_filename = 'results/spiketrain_{}/{}_{}_scale.p'.format(\
                                                                         filename,
                                                                         layer.population.label,
                                                                         size)
            try:
                plb.Path(dump_filename).parent.mkdir(parents=True)
            except OSError as exc:  # Python >2.5
                pass
            pickle.dump(out_data.spiketrains,\
                        open(dump_filename, 'wb'))
            spike_panels.append(pynnplt.Panel(out_data.spiketrains,# xlabel='Time (ms)',
                                          xticks=True, yticks=True,
                                          xlabel='{}, {} scale layer'.format(\
                                                    layer.population.label, size)))
        pynnplt.Figure(*spike_panels).save('plots/{}_{}_{}_scale.png'.format(\
                                                layer_name,
                                                filename,
                                                size))
sim.end()
