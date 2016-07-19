import numpy as np
import pathlib as plb
import pyNN.utility.plotting as plt
import cv2

def copy_to_visualization(pos, ratio, feature_img, visualization_img,
                          layer_shape, delta_i, delta_j):
    """
    Copies a feature image onto the visualization canvas at the given position
    of the neuron layer with the given intensity.

    Arguments:
        `pos`: The position where the feature should be painted

        `ratio`: The intensity ratio with which the feature should be painted

        `feature_img`: The feature which should be drawn

        `visualization_img`: The image onto which to paint the feature

        `layer_shape`: The shape of the neuron layer which detects the features

        `delta_i`: The horizontal offset between the feature layers

        `delta_j`: The vertical offset between the feature layers
    """
    n, m = layer_shape
    f_n = feature_img.shape[0]
    f_m = feature_img.shape[1]
    t_n = visualization_img.shape[0]
    t_m = visualization_img.shape[1]
    p_i = int(pos / m)
    p_j = pos % m
    start_i = delta_i * p_i
    start_j = delta_j * p_j
    if p_i == n - 1:
        start_i = t_n - f_n
    if p_j == m - 1:
        start_j = t_m - f_m
    rationated_feature_img = ratio * feature_img
    for i in range(f_n):
        for j in range(f_m):
            visualization_img[start_i + i][start_j + j] +=\
                rationated_feature_img[i][j]

def plot_weights(weights_dict):
    weight_panels = []
    for name, (weights, shape) in weights_dict.items():
        weight_panels.append(pynnplt.Panel(weights.reshape(10, -1), cmap='gray',
                                       xlabel='{} Connection weights'.format(name),
                                       xticks=True, yticks=True))

    pynnplt.Figure(*weight_panels).save('plots/weights_plot_blurred.png')

def visualization_parts(target_img_shape, layers_dict, feature_imgs_dict,
                        delta_i, delta_j, canvas=None):
    """
    Reconstructs the initial image by drawing the features on the recognized
    positions with an intensity proportional to the respective neuron firing rate.

    Parameters:
        `target_image_shape`: The shape of the original image

        `layers_dict`: A dictionary containing for each image scale all feature
                       layers after recording, one layer for each feature.

        `feature_imgs_dict`: A dictionary containing for each name the
                             corresponding feature image

        `delta_i`: The horizontal offset between the feature layers

        `delta_j`: The vertical offset between the feature layers

    Returns:
        A dictionary which contains for each size a list of pairs of the
        reconstructed images and their feature name, one pair for each feature
    """
    t_n, t_m = target_img_shape
    three_channels = True
    if canvas == None:
        three_channels = False
        visualization_img = np.zeros( (t_n, t_m) )
    else:
        visualization_img = canvas
    max_firing = 1
    for size, layers in layers_dict.items():
        for layer in layers:
            spiketrains = layer.population.get_data().segments[0].spiketrains
            for spiketrain in spiketrains:
                if len(spiketrain) > max_firing:
                    max_firing = len(spiketrain)
    partial_reconstructions_dict = {}   # size -> list of pairs of reconstructed
                                        # images and their feature name
    for size, layers in layers_dict.items():
        partial_reconstructions_dict[size] = []
        if three_channels:
            scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size), 3) )
        else:
            scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
        for layer in layers:
            print('scale: {}, layer: {}'.format(size, layer.population.label))
            out_data = layer.population.get_data().segments[0]
            feature_label = layer.population.label
            feature_img = feature_imgs_dict[feature_label]
            st_n = scaled_vis_img.shape[0]
            st_m = scaled_vis_img.shape[1]
            for i in range(len(out_data.spiketrains)):
                # each spiketrain corresponds to a layer S1 output neuron
                copy_to_visualization(i, len(out_data.spiketrains[i]) / max_firing,
                                      feature_img, scaled_vis_img, layer.shape,
                                      delta_i, delta_j)
            upscaled_vis_img = cv2.resize(src=scaled_vis_img, dsize=(t_m, t_n),
                                          interpolation=cv2.INTER_CUBIC)
            partial_reconstructions_dict[size].append(\
                (visualization_img + upscaled_vis_img.astype(np.int32),
                 feature_label))
            if three_channels:
                scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size), 3) )
            else:
                scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
    return partial_reconstructions_dict

def reconstruct_S1_features(target_img, layer_collection, feature_imgs_dict,
                            args):
    """
    Reconstructs the recognized features of the S1 layer by drawing
    the features onto a black canvas with their intensity proportional to
    the recognition strength.

    Parameters:
        
        `target_img`: The target image

        `layer_collection`: A dictionary containing for each layer name a
                            dictionary containing for each size a list of layers
                            for all features

        `feature_imgs_dict`: A dictionary containing for each name the
                             corresponding feature image

        `args`: The commandline arguments object. Uses the deltas and the target
                image name from it
        
    """
    print('Reconstructing S1 features')
    vis_img = np.zeros(target_img.shape)
    vis_parts = visualization_parts(target_img.shape,
                                    layer_collection['S1'],
                                    feature_imgs_dict,
                                    args.delta_i, args.delta_j)
    for size, img_pairs in vis_parts.items():
        for img, feature_label in img_pairs:
            vis_img += img
    cv2.imwrite('S1_reconstructions/{}_S1_reconstruction.png'.format(\
                                    plb.Path(args.target_name).stem), vis_img)
    
def reconstruct_C1_features(target_img, layer_collection, feature_imgs_dict,
                            args):
    """
    Reconstructs the recognized features of the C1 layer by drawing
    the features onto a copy of the target_img, with their intensity
    proportional to the recognition strength.

    Arguments:

        `target_img`: The target image

        `layer_collection`: A dictionary containing for each layer name a list
                            of those layers

        `feature_imgs_dict`: A dictionary containing for each name the
                             corresponding feature image

        `args`: The commandline arguments object. Uses the deltas and the target
                image name from it

    """
    print('Reconstructing C1 features')
    # Create the RGB canvas to draw colored rectangles for the features
    canvas = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
    # Create the colored squares for the features in a map
    colored_squares_dict = {} # feature name -> colored square
    # A set of predefined colors
    colors = {'red': (255, 0, 0),
              'green': (0, 255, 0),
              'blue': (0, 0, 255),
              'yellow': (255, 255, 0),
              'purple': (255, 0, 255)}
    color_iterator = colors.__iter__()
    for feature_name, feature_img in feature_imgs_dict.items():
        color_name = color_iterator.__next__()
        f_n, f_m = feature_img.shape
        bf_n = 6 * args.delta_i + f_m   # "big" f_n
        bf_m = 6 * args.delta_j + f_m   # "big" f_m
        # Create a square to cover all pixels of a C1 neuron
        square = np.zeros((bf_n, bf_m, 3))
        cv2.rectangle(square, (0, 0), (bf_n - 1, bf_m - 1), colors[color_name])
        print('feature name and color: ', feature_name, color_name)
        colored_squares_dict[feature_name] = square
    vis_parts = visualization_parts(target_img.shape, layer_collection['C1'],
                                    colored_squares_dict,
                                    6 * args.delta_i,
                                    6 * args.delta_j, canvas)
    img_name_stem = plb.Path(args.target_name).stem
    output_dir = plb.Path('C1_reconstructions/' + img_name_stem)
    if not output_dir.exists():
        output_dir.mkdir()
    for size, img_pairs in vis_parts.items():
        for img, feature_label in img_pairs:
            cv2.imwrite(output_dir.as_posix() + '/{}_{}_C1_{}_reconstruction.png'.\
                        format(img_name_stem, size, feature_label), img)

def plot_spikes(layer_collection, args):
    """
    Plots the spikes of the layers in the given dictionary


    Arguments:

        `layer_collection`: A dictionary containing for each layer name a
                            dictionary containing for each size of the target a
                            list of S1 layers

        `args`: The commandline arguments object. Uses the target image name
                from it
    """
    for layer_name in ['S1', 'C1']:
        for size, layers in layer_collection[layer_name].items():
            spike_panels = []
            for layer in layers:
                out_data = layer.population.get_data().segments[0]
                spike_panels.append(plt.Panel(out_data.spiketrains,# xlabel='Time (ms)',
                                              xticks=True, yticks=True,
                                              xlabel='{}, {} scale layer'.format(\
                                                        layer.population.label, size)))
            plt.Figure(*spike_panels).save('plots/{}_{}_{}_scale.png'.format(\
                                                    layer_name,
                                                    plb.Path(args.target_name).stem,
                                                    size))
