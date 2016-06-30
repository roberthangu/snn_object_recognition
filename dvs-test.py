from mpl_toolkits.mplot3d import Axes3D
import pathlib as plb
import matplotlib.pyplot as plt
import rosbag
import pickle

import common as cm
import network as nw

class Stream:
    def __init__(self, events, shape, duration):
        self.events = events
        self.shape = shape
        self.duration = duration

def resize_stream(stream, size):
    # no interpolation so far
    resized_shape = np.ceil(np.multiply(stream.shape, size)).astype(int)
    resized_events = np.copy(stream.events)
    for event in resized_events:
        event.x = int(np.floor(event.x * size))
        event.y = int(np.floor(event.y * size))
    return Stream(resized_events, resized_shape, stream.duration)

def read_stream(filename):
    bag = rosbag.Bag(filename)
    allEvents = []
    initial_time = None
    last_time = 0
    for topic, msg, t in bag.read_messages(topics=['/dvs/events']):
        if not initial_time and msg.events:
            # we want the first event to happen at 1ms
            initial_time = int(msg.events[0].ts.to_sec() * 1000) - 1
        for event in msg.events:
            event.ts = int(event.ts.to_sec() * 1000) - initial_time
        allEvents = np.append(allEvents, msg.events)
        last_time = t.to_sec() * 1000
        # NOTE: I forgot to specify in the Layer class that the shape is
        # specified in matrix notation like (rows, cols). So here maybe
        # (msg.height, msg.width) could be more appropriate?
        shape = [msg.width, msg.height]
    bag.close()

    return Stream(allEvents, shape, last_time - initial_time)

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

args = cm.parse_args()
weights_dict, feature_imgs_dict = nw.train_weights(args.feature_dir)

file_extension = plb.Path(args.target_name).suffix
filename = plb.Path(args.target_name).stem

target = read_stream(args.target_name)
S1_layers = nw.create_S1_layers(target, weights_dict, args.refrac_s1, [1],
                                is_bag=True)

# NOTE: Since in your original code you're using only size 1 in creating the
# corner layers, I don't create an extra dictionary to store only that one
# layer. Hence the `corner_layer` variable.
corner_layer = nw.create_corner_layer_for(S1_layers[1])

layer_collection = [S1_layers, corner_layer]
layer_names = ['S1', 'corner']

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
if args.reconstruct_img:
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
