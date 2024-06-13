import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.stats as scistats
import os
import seaborn as sns
import pandas as pd
from scipy import stats
import itertools


def read_measurements(current_model_dir,
                      parent_dir,
                      measurement,
                      show_epoch_zero_vali,
                      dataset,
                      print_epoch_zero_vali=False,
                      summarize=True
                      ):
    mse_train, mse_vali = {}, {}

    # Iterate through all 5 directories (replicates) in the model directory and for each read the loss.txt file
    for sub_dir in os.listdir(f'{parent_dir}/{current_model_dir}'):
        if 'LM' not in sub_dir:
            continue
        fo = open(f'{parent_dir}/{current_model_dir}/{sub_dir}/{measurement}.txt')
        lines = fo.readlines()
        epoch = int(lines[0].split(',')[0].split(': ')[1])
        if epoch not in mse_vali.keys():
            mse_vali[epoch] = []
        mse_vali[epoch].append(float(lines[0].split(',')[1].split(': ')[1]))
        for line in lines[1:]:
            split_line = line.split(',')
            epoch = int(split_line[0].split(': ')[1])
            if epoch not in mse_train.keys():
                mse_train[epoch] = []
            if epoch not in mse_vali.keys():
                mse_vali[epoch] = []

            mse_train[epoch].append(float(split_line[1].split(': ')[1]))
            mse_vali[epoch].append(float(split_line[2].split(': ')[1]))
        fo.close()

    if not summarize:
        return mse_train, mse_vali

    mean_mse_train = [statistics.mean(mse_train[epoch]) for epoch in mse_train.keys()]
    mean_mse_vali = [statistics.mean(mse_vali[epoch]) for epoch in mse_vali.keys()]
    sd_mse_train = [scistats.tstd(mse_train[epoch]) for epoch in mse_train.keys()]
    sd_mse_vali = [scistats.tstd(mse_vali[epoch]) for epoch in mse_vali.keys()]

    if print_epoch_zero_vali:
        print(f'{current_model_dir} - Epoch 0: Mean MSE: {mean_mse_vali[0]}, SD: {sd_mse_vali[0]}')

    if not show_epoch_zero_vali:
        mean_mse_vali = mean_mse_vali[1:]
        sd_mse_vali = sd_mse_vali[1:]
        if len(mean_mse_vali) != len(mean_mse_train):
            raise ValueError('Validation and training data have different lengths')

    if dataset == 'train':
        mean_mse = mean_mse_train
        sd_mse = sd_mse_train
    elif dataset == 'validation':
        mean_mse = mean_mse_vali
        sd_mse = sd_mse_vali

    return mean_mse, sd_mse


# parent_directory is the directory where all Output directories are located
# Each model_dir is expected to contain 5 folders, each with a loss.txt file
# skip_epoch expects a tupe such as ("Pretrain_ONE_EPOCH", 2) -> in this case the first 2 epochs are skipped for the specified model
def plot_mse_with_break(parent_dir,
                        model_dirs,
                        model_names,
                        model_designs,
                        save_path,
                        cell_type,
                        measurement='loss',
                        dataset='train',
                        ylim='auto',
                        show_epoch_zero_vali=False,
                        figsize=(7, 4),
                        start_break=20,
                        end_break=35,
                        width_ratios=[1.4, 1],
                        skip_epoch=False,
                        print_epoch_zero=False,
                        print_epoch_one=False,
                        xlim_max=45
                        ):
    plt.rcParams['figure.figsize'] = figsize
    f, (ax, ax2) = plt.subplots(1, 2, sharey='all', facecolor='w', gridspec_kw={'width_ratios': width_ratios})

    for current_model_dir, label, design in zip(model_dirs, model_names, model_designs):
        mean_mse, sd_mse = read_measurements(current_model_dir, parent_dir, measurement, show_epoch_zero_vali, dataset, print_epoch_zero)

        cropped_mean_mse = mean_mse[:start_break]
        cropped_mean_mse.extend(mean_mse[end_break:])
        cropped_sd_mse = sd_mse[:start_break]
        cropped_sd_mse.extend(sd_mse[end_break:])

        if show_epoch_zero_vali:
            start, end = 0, len(cropped_mean_mse)
        else:
            start, end = 1, (len(cropped_mean_mse) + 1)

        if skip_epoch is not False:
            for skip_label, skip_epochs in skip_epoch:
                if skip_label == label:
                    start += skip_epochs
                    end += skip_epochs

        if print_epoch_one:
            print(f'{current_model_dir} - Epoch 1: Mean MSE: {cropped_mean_mse[0]}, SD: {cropped_sd_mse[0]}')

        ax.plot(range(start, end), cropped_mean_mse, label=label, **design)
        ax2.plot(range(start, end), cropped_mean_mse, label=label, **design)

        ax.fill_between(range(start, end),
                        np.asarray(cropped_mean_mse) - np.asarray(cropped_sd_mse),
                        np.asarray(cropped_mean_mse) + np.asarray(cropped_sd_mse), alpha=0.25,
                        facecolor=design['color'])
        ax2.fill_between(range(start, end),
                         np.asarray(cropped_mean_mse) - np.asarray(cropped_sd_mse),
                         np.asarray(cropped_mean_mse) + np.asarray(cropped_sd_mse), alpha=0.25,
                         facecolor=design['color'])

    ax.set_xlim(0, start_break)
    ax2.set_xlim(end_break, xlim_max)

    ax.set_xticks(range(0, start_break, 2))
    ax2.set_xticks(range(end_break, xlim_max, 2))

    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.tick_params(labelleft=True, labelright=False)

    ax2.tick_params(labelleft=False, left=False, right=False)

    ax.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
    ax.plot((1 - (d * 0.6), 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - (d * 0.6), 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d * 2, +d * 6), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d * 6), (-d, +d), **kwargs)

    f.text(0.5, 0.001, 'Training Epoch', ha='center')
    f.text(-0.001, 0.5, 'Mean Squared Error', va='center', rotation='vertical')

    if ylim != 'auto':
        plt.ylim(ylim)

    plt.legend(handletextpad=1, labelspacing=1)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(f'{save_path}/MSE_{measurement}_{cell_type}_{dataset}_model_names.pdf', bbox_inches='tight')
    plt.show()


def plot_mse_without_break(parent_dir,
                           model_dirs,
                           model_names,
                           model_designs,
                           nr_epochs,
                           save_path,
                           cell_type,
                           measurement='loss',
                           dataset='train',
                           ylim='auto',
                           show_epoch_zero_vali=False,
                           figsize=(5, 4),
                           skip_epoch=False,
                           show_legend=False
                           ):
    plt.rcParams['figure.figsize'] = figsize
    fig, ax = plt.subplots()

    for current_model_dir, label, design in zip(model_dirs, model_names, model_designs):
        mean_mse, sd_mse = read_measurements(current_model_dir, parent_dir, measurement, show_epoch_zero_vali, dataset)

        cropped_mean_mse, cropped_sd_mse = mean_mse[:nr_epochs], sd_mse[:nr_epochs]

        if show_epoch_zero_vali:
            start, end = 0, len(cropped_mean_mse)
        else:
            start, end = 1, (len(cropped_mean_mse) + 1)

        if skip_epoch is not False:
            for skip_label, skip_epochs in skip_epoch:
                if skip_label == label:
                    start += skip_epochs
                    #end += skip_epochs
        #if skip_epoch is not False and skip_epoch[0] == label:
         #   start += skip_epoch[1]
            # remove last element from cropped_mean_mse
                    cropped_mean_mse = cropped_mean_mse[:(-1 * skip_epochs)]
                    cropped_sd_mse = cropped_sd_mse[:(-1 * skip_epochs)]

        ax.plot(range(start, end), cropped_mean_mse, label=label, **design)

        ax.fill_between(range(start, end),
                        np.asarray(cropped_mean_mse) - np.asarray(cropped_sd_mse),
                        np.asarray(cropped_mean_mse) + np.asarray(cropped_sd_mse), alpha=0.25,
                        facecolor=design['color'])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    if ylim != 'auto':
        plt.ylim(ylim)

    if show_legend:
        plt.legend(handletextpad=1, labelspacing=1)

    plt.ylabel('Mean Squared Error')
    plt.xlabel('Training Epoch')
    plt.tight_layout()
    plt.savefig(f'{save_path}/MSE_{measurement}_{cell_type}_{dataset}_model_names.pdf', bbox_inches='tight')
    plt.show()

def boxplots_one_epoch(epoch, parent_dir, model_dirs, model_names, model_designs, save_path, cell_type,
                           measurement='loss', ylim='auto', show_legend=False):
    plt.rcParams['figure.figsize'] = (3, 3)
    fig, ax = plt.subplots()

    distributed_x = [-0.2, -0.1, 0, 0.1, 0.2]
    for i, (current_model_dir, label, design) in enumerate(zip(model_dirs, model_names, model_designs)):
        mse_train, mse_vali = read_measurements(current_model_dir, parent_dir, measurement, show_epoch_zero_vali=True, dataset="vali", summarize=False)
        if epoch == 'max':
            # Find highest epoc with 5 values
            sorted_epochs = sorted(mse_vali.keys())
            for current_epoch in sorted_epochs[::-1]:
                if len(mse_vali[current_epoch]) == 5:
                    break
            print(f'{label}: Epoch {current_epoch}')
            epoch_vali = mse_vali[current_epoch]
        else:
            epoch_vali = mse_vali[epoch]
        ax.scatter([x+i for x in distributed_x], epoch_vali, label=label, **design, alpha=0.8)
        ax.boxplot(epoch_vali, positions=[i], widths=0.6, showfliers=False, boxprops=dict(color=design['color']), whiskerprops=dict(color=design['color']), capprops=dict(color=design['color']))

    # Set model names as x-ticks and rotate them
    ax.set_xticks(range(len(model_names)), [x.replace(' ', '\n') for x in model_names], rotation=45, ha='right', rotation_mode='anchor')
    ax.grid(True, alpha=0.3)

    if ylim != 'auto':
        plt.ylim(ylim)

    if show_legend:
        # Place legend right next to plot
        plt.legend(handletextpad=1, labelspacing=1, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylabel('Mean Squared Error')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(f'{save_path}/MSE_{measurement}_{cell_type}_epoch_{epoch}.pdf', bbox_inches='tight')
    plt.show()


def performance_boxplots(validation_loss_dict, model_dirs, model_names, palette, save_path, epoch, set, ylim='auto',
                         y_offset_fixed=0, y_offset_scale=0.1, **kwargs):
    plt.rcParams['figure.figsize'] = (3, 3.5)
    fig, ax = plt.subplots()

    plot_df = pd.DataFrame({"Model": [], f"{set} MSE": []})
    for model, mse in validation_loss_dict.items():
        plot_df = pd.concat([plot_df, pd.DataFrame({"Model": [model] * len(mse), f"{set} MSE": mse})])

    sns.boxplot(data=plot_df, x="Model", y=f"{set} MSE", palette=palette, showfliers=False, ax=ax, **kwargs)

    # Set model names as x-ticks and rotate them
    ax.set_xticks(range(len(model_names)), [x.replace(' ', '\n') for x in model_names], rotation=45, ha='right',
                  rotation_mode='anchor')
    ax.grid(True, alpha=0.3)

    nr_combinations = len(list(itertools.combinations(model_dirs, 2)))
    y_max_global = max([max(validation_loss_dict[model]) for model in model_dirs])
    seen = []
    # Add statistical significance
    for x1, model1 in enumerate(model_dirs):
        for x2, model2 in enumerate(model_dirs):
            if x1 == x2 or (model1, model2) in seen or (model2, model1) in seen:
                continue
            seen.append((model1, model2))

            layer = abs(x2 - x1)
            y_offset = layer * y_offset_scale * y_max_global - y_offset_fixed

            y_max = max([max(validation_loss_dict[model1]), max(validation_loss_dict[model2])])
            ax.plot([min(x1, x2) + 0.1, max(x1, x2) - 0.1], [y_max + y_offset, y_max + y_offset], color='black', linewidth=0.6)

            p_value = stats.ttest_rel(validation_loss_dict[model1], validation_loss_dict[model2])[1]
            p_val_adj = p_value * nr_combinations
            print(f"{model1} vs {model2}: {p_val_adj}")
            if p_val_adj < 0.001:
                ax.text((x1 + x2) / 2, y_max + y_offset + (0.01 * y_max_global), "***", ha='center', va='center',
                        color='black', fontsize=6)
            elif p_val_adj < 0.01:
                ax.text((x1 + x2) / 2, y_max + y_offset + (0.01 * y_max_global), "**", ha='center', va='center', color='black', fontsize=6)
            elif p_val_adj < 0.05:
                ax.text((x1 + x2) / 2, y_max + y_offset + (0.01 * y_max_global), "*", ha='center', va='center', color='black', fontsize=6)
            else:
                ax.text((x1 + x2) / 2, y_max + y_offset + (0.03 * y_max_global), "n.s.", ha='center', va='center',
                        color='black', fontsize=8)

    if ylim != 'auto':
        plt.ylim(top=ylim)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{set}_boxplot_epoch_{epoch}.pdf', bbox_inches='tight')
    plt.show()