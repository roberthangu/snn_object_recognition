# One-Shot Object Appearance Learning using Spiking Neural Networks #

_Note: This project is no longer actively maintained and supported._

This is a Spiking Neural Network used for testing one-shot object appearance 
learning. That is the learning of new object features based on one or very few 
training instances.

It is written in Python and runs on the NEST neurosimulator, giving the
framework a better biological plausibility over other networks of this kind,
which use own implementations for the neural mechanics.

## Features ##

The network consists of 5 layers of spiking neurons. The first 4 are 
alternating simple and complex cell layers (S1, C1, S2, C2), and the 5th is a 
classifier (e.g. a SVM). The learning of the object features happens between 
the C1 - S2 layers using Spike-Timing-Dependent Plasticity (STDP). This 
architecture is inspired by the work of [Masquelier et al.][masq] Some sample 
features learned by the network can be seen below.

![](samples/mo_1.png) ![](samples/mo_2.png) ![](samples/mo_3.png) ![](samples/mo_4.png)

![](samples/fa_1.png) ![](samples/fa_2.png) ![](samples/fa_3.png) ![](samples/fa_4.png)

Table: Features extracted from motorbikes (top) and faces (bottom)

These features were extracted from the **Motorbikes** and **Faces** datasets of the
[Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) image 
training set. They were learned by presenting only pictures of the same
dataset to the network. In contrast to that, the image below shows a set of smaller
features extracted by showing images of three classes combined to the network,
namely of **Airplanes**, **Motorbikes** and **Faces**. 

![](samples/combined_airplanes_mo_fa.png)

These _combined features_ are used for the One-Shot appearance learning, as the
network tries to "find" these features inside of new, unseen object classes.

There are also videos showing the convergence of the weights during the
training with [motorbikes and faces](video/motorbikes-faces.avi) and
[airplanes, motorbikes and pianos](video/airplanes-motorbikes-pianos.avi).

## Usage ## 

Since running all the layers of the network at once is computationally very 
slow, the process is divided into several steps, which are run separately, one 
after another. Specifically, the basic data on which the computation relies are
the **spiketrains**. Spikes are propagated from the first layer to the last.
Thus, to speed up the computation, the simulation can be "stopped" after a certain
layer, dump the spiketrains to a file and use them as an input for the next
layer in a later simulation, thus avoiding the need to recompute the same
information again when tuning or testing a certain layer.

For this purpose there are three scripts:

* `dump-c1-spikes.py` or `dump-blanked-c1-spikes.py`:
  Runs from input image to the C1 layer. The output is the C1 spiketrains. The
  second script adds a blanktime between each consecutive images. This is
  beneficial for the recognition later.
* `learn-features.py`: Simulate the C1 - S2 layers. This is the place where the
  S2 weights are learned, i.e. the "features", and are dumped to file,
  from which they can be later used for classification. The filename is
  automatically generated from the given command line parameters and the name of
  the C1 spike dumpfile.
* `dump-c2-spikes.py`: C1 to the C2 layer
* `classify-images.py` or `classify-images-one-shot.py`: These scripts use the
  weights learned previously to learn new object classes images in a one-shot manner. The 
  first script uses a SVM for the classification of the images and does not rely
  on the dumped C2 spikes. The second script does "real" one-shot classification
  by training an extra fully connected neural layer with STDP instead of just 
  using an SVM.  Thus it uses the dumped C2 spikes to speed up the training of 
  the last layer. **Both scripts use S2 weights pre-learned from a set of classes
  and apply them to learn the characteristics of new classes**.

The usage of each file can be seen by running it with the `--help` command line
argument. Below is also a minimal example for each script with some sane
defaults.

1. To dump the C1 spiketrains with a blanktime between consecutive images:

    ```
    ./dump-blanked-c1-spikes.py --
        --dataset-label <your label>
        --training-dir <training images>
    ```

2. Train the C1 - S2 weights (i.e. extract the features). The filename of the
   weights dumpfile is automatically generated:

    ```
    ./learn-features.py --
        --c1-dumpfile <c1 spiketrain dumpfile> 
    ```

3. [Optional. Used for accelerating the STDP learning in the one-shot classifier]
   Dump the C2 spiketrains:

    ```
    ./dump-c2-spikes.py --
        --training-c1-dumpfile <c1 spiketrain dumpfile>
        --weights-from <S2 weigths dumpfile from step 2>
    ```

4. Learn and classify new classes by using the weights of step 2 either with an
   SVM (first script) or with a fully connected end-layer using STDP:

    ```
    ./classify-images.py --
       --training-c2-dumpfile <c1 dumpfile of the training dataset>  
       --validation-c1-dumpfile <c1 dumpfile of the validation dataset>
       --training-labels <text file containing the labels of the training images>
       --validation-labels <text file containing the labels of the validation images>
       --weights-from <S2 weigths dumpfile from step 2>

    ./classify-images-one-shot.py --
       <same parameters as above>
    ```

## Installation ##

**NOTE:** At the moment the network relies on a NEST extension which adds a 
shared-weights synapse type. This mechanism greatly speeds up the computation, 
since weight changes to one synapse no longer need to be copied to all the 
others, but is read from a single shared table.

In order to run the code, the user needs to install 
[NEST](http://nest-simulator.org/) 2.10.0 with 
[PyNN](http://neuralensemble.org/PyNN/) 0.8.1. Please
consult their corresponding web pages for installation instructions.

The code is written in **Python 3**, thus a working installation of it is also 
required.

[masq]: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030031
