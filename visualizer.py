from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle
from cycler import cycler
from os import listdir
from os.path import isfile, join

pylab.ion()
color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']
marker_list = [ '+', 'x', 'o', 'v', '^', '<', '>' ]

def plot_3d_spatiotemporal():
    allSpikesPerLayer = pickle.load(open('results/spatiotemporal_dvs-page2-30s_2016-06-24-18-15-21.p', 'r'))

    fig = plt.figure(figsize=(40,5))
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('times')

    for i, layer in enumerate(allSpikesPerLayer):
        ax.scatter(layer[0], layer[2], layer[1], c=color_list[i])

    fig.show()

def plot_2d_spiketrains(pathToPickles):
    fig = plt.figure(figsize=(40,5))
    ax = fig.add_subplot(1, 1, 1)

    pickleFilenames = listdir(pathToPickles)
    allPickles = [join(pathToPickles, f) for f in pickleFilenames if isfile(join(pathToPickles, f))]
    allSpiketrains = [ pickle.load(open(pickleFile, 'r')) for pickleFile in allPickles ]
    for featureMapIdx, spiketrain in enumerate(allSpiketrains):
        populationName = pickleFilenames[featureMapIdx]
        x = []
        y = []
        for i, neuron in enumerate(spiketrain):
            for spike in neuron:
                x.append(spike)
                y.append(i)
        if populationName.startswith("corner"):
            markersize = 10
            alpha = 1
        else:
            markersize = 1
            alpha = 0.1

        ax.scatter(x, y, marker=marker_list[featureMapIdx], c=color_list[featureMapIdx], label=populationName, linewidths=markersize, alpha = alpha)
    ax.legend()

# plot_2d_spiketrains('results/spiketrain_dvs-page4_2016-06-23-14-58-02')
plot_2d_spiketrains('results/spiketrain_dvs-page2-30s_2016-06-24-18-15-21')

raw_input("Press Enter to continue...")
