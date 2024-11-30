"""
Run this file to analyse the neural responses to static gratings of the different RNN models (Panels C, D, E from Fig 6.).

Before running this script, make sure to run the script model_response_to_static_gratings.py to generate the neural responses
to the static gratings of the different RNN models.
"""


import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from model_response_to_static_gratings import generate_paths
from RF_static_gratings.generate_static_gratings import static_gratings_params


plt.style.use("default")
plt.rcParams['figure.dpi'] = 300
font = {'family': 'sans-serif', 'sans-serif': ['Myriad Pro'], 'size': 14}
matplotlib.rc('font', **font)


def model_color(model_name):
    if model_name == "RW":
        return '#0065bd'
    elif model_name == "NI":
        return '#000000'
    elif model_name == "RD":
        return '#4cc997'

def model_edge_color(model_name):
    if model_name == "RW":
        return '#0343DF'
    elif model_name == "NI":
        return '#F97306'
    elif model_name == "RD":
        return '#008000'

def model_label(model_name):
    if model_name == "RW":
        return 'DirectedProp'
    elif model_name == "NI":
        return 'Fully trained'
    elif model_name == "RD":
        return 'Randomly Initialized'

def model_label_2_lines(model_name):
    if model_name == "RW":
        return 'Directed\n Prop'
    elif model_name == "NI":
        return 'Fully\n trained'
    elif model_name == "RD":
        return 'Randomly\n Initialized'

def layer_label(layer_name):
    if 'convrecurrent' in layer_name:
        return f'ConvRec{layer_name[-1]}'
    elif 'conv_output' in layer_name:
        return 'ConvOutput'




#--------------------------------------------------------------------------------------------------------------------
# Plotting functions (Histograms)

def plot_histograms_std_changes(dict_std_changes, layer_name):
    # Determine the global range across all models for the specific layer
    all_std = np.concatenate([dict_std_changes[model_name][layer_name] for model_name in dict_std_changes])
    global_min = np.min(all_std)
    global_max = np.max(all_std)

    # Define bin edges based on the overall range
    number_of_bins = 50
    bin_edges = np.linspace(global_min, global_max, number_of_bins + 1)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    for model_name in dict_std_changes.keys():
        std = dict_std_changes[model_name][layer_name]
        c = model_color(model_name)
        l = model_label(model_name)
        ax.hist(std, density=True, bins=bin_edges, alpha=0.5, color=c, label=l)

    if "convrecurrent" in layer_name:
        ax.set_xlim(0, 0.5)
    else:
        ax.set_xlim(0, 1.2)
    ax.set_xlabel('Standard deviation of the change in activity')
    ax.set_ylabel('Density')
    layer_l = layer_label(layer_name)
    plt.title(f'{layer_l}')
    plt.legend()
    path_fig = os.path.join(f'std_changes_{layer_name}.png')
    plt.savefig(path_fig)
    plt.close()
    return None

def violin_plots_changes_in_activity(dict_changes, layer_name):
    assert layer_name == "convrecurrent3"

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    nrn_idx = 15
    positions = []
    labels = []
    for i, model_name in enumerate(dict_changes.keys()):
        changes = dict_changes[model_name][layer_name] # Shape: (n_exps, n_neurons)
        c = model_color(model_name)
        e = model_edge_color(model_name)
        l = model_label_2_lines(model_name)
        changes_distribution_nrn = changes[:, nrn_idx]
        pos = i*0.75 + 1  # Position for the current model's violin plot
        positions.append(pos)
        labels.append(l)
        parts = ax.violinplot(changes_distribution_nrn, showmeans=True, showmedians=False, showextrema=True,
                              vert=True, widths=0.5, positions=[pos], points=60, bw_method=0.5)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_edgecolor(e)
            pc.set_alpha(0.5)

        if 'cbars' in parts:
            parts['cbars'].set_color(e)
        if 'cmeans' in parts: # Customize the mean line color
            parts['cmeans'].set_color(e)  # Set the color of the mean line
        if 'cmins' in parts: # Customize the extrema lines color
            parts['cmins'].set_color(e)  # Set the color of the minimum line
        if 'cmaxes' in parts:
            parts['cmaxes'].set_color(e)  # Set the color of the maximum line

        # Calculate the mean and standard deviation
        mean_val = np.mean(changes_distribution_nrn)
        std_dev = np.std(changes_distribution_nrn)

        # Annotate the standard deviation with an arrow around the mean
        ax.annotate('', xy=(pos, mean_val + std_dev), xytext=(pos, mean_val),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        ax.annotate('', xy=(pos, mean_val - std_dev), xytext=(pos, mean_val),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Change in activity across gratings')
    ax.set_title(f'Change in activity distributions for one randomly sampled nrn in {layer_label(layer_name)}')
    path_fig = os.path.join(f'violin_plot_{layer_name}.png')
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(path_fig)
    plt.close()
    return None


#--------------------------------------------------------------------------------------------------------------------
# Plotting functions (Comparisons of distributions across training regimes)

def plot_means(m):
    # Input: dictinary with model names as keys and dictionaries of layer names and empirical means as values
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    # Keep track of labels to prevent duplicates in the legend
    handled_labels = set()

    for model_name, layers in m.items():
        # Get color and label for each model
        color = model_color(model_name)
        label = model_label(model_name)

        # Prepare data for plotting
        layer_data_m = [layers.get(layer, 0) for layer in layers.keys()]  # Use 0 for missing layers
        layer_names = [layer_label(layer) for layer in layers.keys()]

        # Plotting each model's line with dots
        if label not in handled_labels:
            ax.plot(layer_names, layer_data_m, marker='o', linestyle='-', color=color, label=label)
            handled_labels.add(label)
        else:
            ax.plot(layer_names, layer_data_m, marker='o', linestyle='-', color=color)

    ax.set_ylabel('Empirical mean')
    ax.set_ylim(0, None)
    plt.xticks()
    plt.legend()
    plt.grid(True)

    # Save the figure
    path_fig = os.path.join('means_std_changes_distribution.png')
    plt.savefig(path_fig)
    plt.close()
    return None


#--------------------------------------------------------------------------------------------------------------------


def std_change_in_activity_distributions_all_models():
    """
    Generate panels C, D, E from Figure 6.
    """
    model_names = ["NI", "RW", "RD"]
    np.random.seed(37)
    nrn_idx = np.random.randint(0, 1000, 32)
    changes_to_plot = {'RD': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'RW': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'NI': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None}}
    std_changes = {'RD': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'RW': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'NI': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None}}
    mean_std_distribution = {'RW': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'NI': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'RD': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None}}
    max_dist_to_mean = {'RW': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'NI': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'RD': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None}}

    for model_name in model_names:
        _, _, path_responses = generate_paths(model_name)
        with open(f'{path_responses}/changes_in_activity.pkl', 'rb') as file:
            changes = pickle.load(file)

            for layer_name in ["convrecurrent1", "convrecurrent2", "convrecurrent3", "conv_output"]:

                changes_layer = [changes[f'{layer_name}_o_{o}_sf_{sf}_p_{p}'] for o in static_gratings_params['orientation'] for sf in static_gratings_params['spatial_frequency'] for p in static_gratings_params['phase']]
                changes_layer = np.stack(changes_layer, axis=0)

                if 'convrecurrent' in layer_name:
                    n_exps, n_nrn = changes_layer.shape[0], changes_layer.shape[1]*changes_layer.shape[2]*changes_layer.shape[3]
                elif 'conv_output' in layer_name:
                    n_exps, n_nrn = changes_layer.shape[0], changes_layer.shape[1]*changes_layer.shape[2]

                changes_layer = changes_layer.reshape(n_exps, -1)
                changes_to_plot[model_name][layer_name] = changes_layer[:, nrn_idx]
                std_change_per_nrn = changes_layer.std(axis=0)

                std_changes[model_name][layer_name] = std_change_per_nrn
                mean_std_distribution[model_name][layer_name] = std_change_per_nrn.mean()

                avg_change_per_nrn = changes_layer.mean(axis=0)
                distance_to_mean = np.abs(changes_layer - avg_change_per_nrn)
                m = distance_to_mean.max(axis=0)
                max_dist_to_mean[model_name][layer_name] = m.max()

    violin_plots_changes_in_activity(changes_to_plot, "convrecurrent3")
    plot_means(mean_std_distribution)
    for layer_name in ["convrecurrent1", "convrecurrent2", "convrecurrent3", "conv_output"]:
        plot_histograms_std_changes(std_changes, layer_name)

    return None






if __name__ == '__main__':

    std_change_in_activity_distributions_all_models()