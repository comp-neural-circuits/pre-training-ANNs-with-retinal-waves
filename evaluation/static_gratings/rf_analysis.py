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
import seaborn as sns

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

def plot_hist_all_changes_in_activity(dict_changes, layer_name):
    """
    Plot the distribution of changes in activity for all neurons and all gratings in a given layer,
    with each model's distribution on a separate subplot arranged vertically.
    """
    std_dict = {'RD': None, 'RW': None, 'NI': None}
    
    plt.figure(figsize=(6, 4))
    for model_name in dict_changes.keys():
        changes = dict_changes[model_name][layer_name] # Shape: (n_exps, n_neurons)
        changes = changes.flatten()
        c = model_color(model_name)  # You should define this function to map model names to colors
        l = model_label_2_lines(model_name)  # Define this to get labels possibly breaking into two lines

        sns.kdeplot(changes, color=c, label=l, fill=True)
        std = np.std(changes)
        std_dict[model_name] = std
    
    plt.legend(loc='upper right')
    plt.ylabel('Kernel Density Estimate')
    plt.xlabel('Changes in Activity')  # Only set x-label on the last subplot
    plt.title(f'{layer_label(layer_name)}')

    path_fig = os.path.join(f'all_changes_in_activity_{layer_name}.pdf')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.close()
    
    return std_dict


#--------------------------------------------------------------------------------------------------------------------
# Plotting functions (Comparisons of distributions across training regimes)

def plot_std(std_dict):
    # Input: dictinary with model names as keys and dictionaries of layer names and empirical means as values
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    # Keep track of labels to prevent duplicates in the legend
    handled_labels = set()

    reorganised_dict = {'RD': [], 'RW': [], 'NI': []}
    layer_names = ['convrecurrent1', 'convrecurrent2', 'convrecurrent3', 'conv_output']
    for i, (layer_name, layer_stds) in enumerate(std_dict.items()):
        layer_names[i] = layer_label(layer_name)
        for model_name, std in layer_stds.items():
            reorganised_dict[model_name].append(std)
            
    for model_name, std_list in reorganised_dict.items():        
        # Get color and label for each model
        color = model_color(model_name)
        label = model_label(model_name)
            
        # Plotting each model's line with dots
        if label not in handled_labels:
            ax.plot(layer_names, std_list, marker='o', linestyle='-', color=color, label=label)
            handled_labels.add(label)
        else:
            ax.plot(layer_name, std, marker='o', linestyle='-', color=color)

    ax.set_ylabel('Standard deviation')
    ax.set_ylim(0, None)
    plt.xticks()
    plt.legend()
    plt.grid(True)

    # Save the figure
    path_fig = os.path.join('std_changes_distribution.pdf')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.close()
    return None


#--------------------------------------------------------------------------------------------------------------------

def plot():
    model_names = ["NI", "RW", "RD"]
    changes_to_plot = {'RD': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'RW': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None},
                    'NI': {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None}}

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
                changes_to_plot[model_name][layer_name] = changes_layer

    
    std_dict = {'convrecurrent1': None, 'convrecurrent2': None, 'convrecurrent3': None, 'conv_output': None}
    for layer_name in ["convrecurrent1", "convrecurrent2", "convrecurrent3", "conv_output"]:
        std_dict[layer_name] = plot_hist_all_changes_in_activity(changes_to_plot, layer_name)
    plot_std(std_dict)
    return None






if __name__ == '__main__':
    plot()