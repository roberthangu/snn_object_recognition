import numpy as np
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
                       layers after recording.

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
            print('layer :', layer.population.label)
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
                (visualization_img + upscaled_vis_img.astype(np.int64),
                 feature_label))
            if three_channels:
                scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size), 3) )
            else:
                scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
    return partial_reconstructions_dict

