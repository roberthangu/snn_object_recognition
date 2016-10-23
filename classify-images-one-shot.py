import pathlib as plb
import numpy as np
import pickle
import argparse as ap
import re
import matplotlib.pyplot as mplt
import pyNN.nest as sim
from sklearn import metrics

parser = ap.ArgumentParser()
parser.add_argument('--training-c2-dumpfile', type=str, required=True,
                    help='The output file to contain the C2 spiketrains for\
                         training')
parser.add_argument('--validation-c2-dumpfile', type=str, required=True,
                    help='The output file to contain the C2 spiketrains for\
                         validation')
parser.add_argument('--training-labels', type=str, required=True,
                    help='Text file which contains the labels of the training\
                          dataset')
parser.add_argument('--validation-labels', type=str, required=True,
                    help='Text file which contains the labels of the validation\
                          dataset')
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

# Get metadata from filenames
training_dumpfile_name = plb.Path(args.training_c2_dumpfile).stem
validation_dumpfile_name = plb.Path(args.validation_c2_dumpfile).stem
training_image_count = int(re.search('\d*imgs',
                                     training_dumpfile_name).group()[:-4])
validation_image_count = int(re.search('\d*imgs',
                                       validation_dumpfile_name).group()[:-4])
training_sim_time = float(re.search('\d+\.\d+ms',
                                        training_dumpfile_name).group()[:-2])
validation_sim_time = float(re.search('\d+\.\d+ms',
                                        validation_dumpfile_name).group()[:-2])
imgs_per_category = int(re.search('\d+learn',
                                        training_dumpfile_name).group()[:-5])
categories = training_image_count // imgs_per_category

# Read the training and validation labels from file
training_labels = open(args.training_labels, 'r').read().splitlines()
validation_labels = open(args.validation_labels, 'r').read().splitlines()

# Read the spike train structures from the pickled dumpfiles
c2_training_spikes = pickle.load(open(args.training_c2_dumpfile, 'rb'))
c2_validation_spikes = pickle.load(open(args.validation_c2_dumpfile, 'rb'))
s2_prototype_cells = len(c2_training_spikes[0][1])

def create_C2_populations(spiketrains):
    C2_populations = [sim.Population(1,
                            sim.SpikeSourceArray(spike_times=[spiketrains[prot]]),
                            label=str(prot))\
                        for prot in range(len(spiketrains))]
    compound_C2_population = C2_populations[0]
    for pop in C2_populations[1:]:
        compound_C2_population += pop
    return (C2_populations, compound_C2_population)

results_label = '{}_{}valimgs_{}valsimtime'\
                    .format(training_dumpfile_name, validation_image_count,
                            validation_sim_time)
logfile = open('log/{}.log'.format(results_label), 'w')

def plot_spikes(C2_populations, classifier_neurons, t_sim_time, appendix):
    fig_settings = {
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.labelsize': 'small',
        'legend.fontsize': 'small',
    }
    mplt.rcParams.update(fig_settings)
    mplt.figure(figsize=(10, 8))
    mplt.subplot(311)
    mplt.axis([0, t_sim_time, -.2, len(C2_populations) - .8])
    mplt.xlabel('Time (ms)')
    mplt.ylabel('Neuron index')
    mplt.grid(True)
    for i in range(len(C2_populations)):
        st = C2_populations[i].get_data().segments[0].spiketrains[0]
        mplt.plot(st, np.ones_like(st) * i, '.')
    mplt.subplot(312)
#    mplt.axis([0, t_sim_time, -.2, len(classifier_neurons) - .8])
    for i in range(len(classifier_neurons)):
        st = classifier_neurons[i].get_data().segments[0].spiketrains[0]
        mplt.plot(st, np.ones_like(st) * i, '.')
    mplt.subplot(313)
    for i in range(len(classifier_neurons)):
        segm = classifier_neurons[i].get_data().segments[0]
        voltages = segm.filter(name='v')[0]
        mplt.plot(voltages.times, voltages, label=str(i))
    mplt.savefig('plots/CLF/{}_{}.png'.format(results_label, appendix))

# Datastructure to store the learned weights from all epochs
all_epochs_weights = []
for training_pair, validation_pair in\
        zip(c2_training_spikes[31:32], c2_validation_spikes[31:32]):

    # ============= Training ============= #
    print('Construct training network')
    sim.setup(threads=args.threads, min_delay=.1)
    # Create the C2 layer and connect it to the single output neuron
    training_spiketrains = [[s for s in st] for st in training_pair[1]]
    C2_populations, compound_C2_population =\
            create_C2_populations(training_spiketrains)
    out_p = sim.Population(1, sim.IF_curr_exp())
    stdp = sim.STDPMechanism(weight=.4,
           timing_dependence=sim.SpikePairRule(tau_plus=5.0, tau_minus=5.0,
                                               A_plus=0.05, A_minus=0.03),
           weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0))
    learn_proj = sim.Projection(compound_C2_population, out_p,
                                sim.AllToAllConnector(), stdp)

    epoch = training_pair[0]

    # Record the spikes for visualization purposes
    compound_C2_population.record('spikes')
    out_p.record(['spikes', 'v'])
    #for pop in C2_populations:
    #    pop.record('spikes')

    # Datastructure for storing the computed STDP weights for this epoch
    classifier_weights = [] # type: List[List[List[float]]]
    
    # Training STDP weights for every category and save them to classifier_weights
    for category in range(categories):
        print('Train weights for', training_labels[category])
        sim.run(training_sim_time * imgs_per_category)
        classifier_weights.append(learn_proj.get('weight', 'array'))
        print('weights')
        print(classifier_weights[-1])
        learn_proj.set(weight=.2)

    plot_spikes(C2_populations, [out_p], training_sim_time * training_image_count,
                'training')

    sim.end()

    # ============= Validation ============= #
    print('Constructing new network with the learned weights')
    sim.setup(threads=args.threads, min_delay=.1)
    
    # Create the validation network and connect the C2 neurons to it
    validation_spiketrains = [[s for s in st] for st in validation_pair[1]]
    C2_populations, compound_C2_population =\
                                create_C2_populations(validation_spiketrains)
    classifier_neurons = [sim.Population(1, sim.IF_curr_exp())\
                                for cat in range(categories)]
    for category in range(categories):
        sim.Projection(compound_C2_population, classifier_neurons[category],
                       sim.AllToAllConnector(),
                       sim.StaticSynapse(weight=classifier_weights[category]))

    # Record the spikes for visualization purposes
    compound_C2_population.record('spikes')
    for pop in classifier_neurons:
        pop.record(['spikes', 'v'])

    predicted_labels = []
    # Simulate and classify the images
#    for i in range(validation_image_count):
    for i in range(4):
        print('Simulating for image', i)
        sim.run(validation_sim_time)
        # Find the neuron which fired most
        # TODO: after taking the max, clear the recorded spikes with
        #       get_data(clear=True)
        label, count = max(zip(training_labels,
                       map(lambda pop: list(pop.get_spike_counts().values())[0],
                           classifier_neurons)), key=lambda pair: pair[1])
        print('label', label, 'count', count)
#        predicted_labels.append(label)
#        for clf_n in classifier_neurons:
#            clf_n.get_data(clear=True)

    plot_spikes(C2_populations, classifier_neurons, 4 * validation_sim_time,
                'validation')

#    print('============================================================',
#          file=logfile)
#    print('Epoch', epoch, file=logfile)
#    clf_report = metrics.classification_report(validation_labels, predicted_labels)
#    conf_matrix = metrics.confusion_matrix(validation_labels, predicted_labels)
#    print(clf_report, file=logfile)
#    print(clf_report)
#    print(conf_matrix, file=logfile)
#    print(conf_matrix)
#
#    all_epochs_weights.append((epoch, classifier_weights))
    sim.end()

#print('Wrote log to file', logfile.name())
#clf_dumpname = 'CLF_weights/{}.bin'.format(results_label)
#clf_dumpfile = open(clf_dumpname, 'wb', protocol=4)
#print('Dumping classificator weights to file', clf_dumpname)
#pickle.dump(all_epochs_weights, clf_dumpfile)
