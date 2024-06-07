import act_max_util_V1 as amu
import torch
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import os
from Compound_models_V3 import NextFramePredictor
import argparse

# add command line argument model path
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--model_name', help='Name of model')
parser.add_argument('-p', '--model_path', help='Path to model')

args = parser.parse_args()

model_name = args.model_name
model_dir = args.model_path
if model_dir == 'None':
    model_dir = None

NR_TRIALS_PER_MODEL = 10
NR_STEPS_PER_TRIAL = 100
unit = (32, 48)  # Target unit in output layer
alpha = torch.tensor(20)  # learning rate (step size)
verbose = False  # print activation every step
L2_Decay = True  # enable L2 decay regularizer
Gaussian_Blur = True  # enable Gaussian regularizer
Norm_Crop = True  # True # enable norm regularizer
Contrib_Crop = True  # enable contribution regularizer
scale = False
clip_image = True

parent_dir = "/home/ge28jib/RetinalWaves/Output"
OUTPUT_DIR = "/home/ge28jib/RetinalWaves/Output/Receptive_Field_Investigation_Per_Model"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
#plt.rcParams['figure.dpi'] = 200


# get path of file that has "best_model_epoch" and "seq2latent" in its name
def get_model_files(model_name, dir_path):
    if model_name == 'Pretrained_Directional_One_Epoch':
        for file in os.listdir(f'{dir_path}/saved_models_per_epoch'):
            if 'model_epoch_1' in file and 'seq2latent' in file:
                model_seq2latent_path = f'{dir_path}/saved_models_per_epoch/{file}'
            elif 'model_epoch_1' in file and 'latent2nextframe' in file:
                model_latent2nextframe_path = f'{dir_path}/saved_models_per_epoch/{file}'
    else:
        for file in os.listdir(dir_path):
            if 'best_model_epoch' in file and 'seq2latent' in file:
                model_seq2latent_path = f'{dir_path}/{file}'
            elif 'best_model_epoch' in file and 'latent2nextframe' in file:
                model_latent2nextframe_path = f'{dir_path}/{file}'

    return model_seq2latent_path, model_latent2nextframe_path


def avrg_receptive_field_for_one_model(model_name, current_model_dir):
    print(f"Investigating receptive field of model: {model_name}...")

    dir_list = [None, None, None, None, None]
    if current_model_dir != None:
        dir_list = os.listdir(f'{parent_dir}/{current_model_dir}')

    five_model_average = np.zeros((10, 64, 64))
    # Iterate through all 5 directories in the model directory
    for model_nr, sub_dir in enumerate(dir_list):
        if sub_dir != None:
            if 'LM' not in sub_dir:
                continue
            model_seq2latent_path, model_latent2nextframe_path = get_model_files(model_name,
                                                                                 f'{parent_dir}/{current_model_dir}/{sub_dir}')

        model = NextFramePredictor(num_channels=1, num_kernels=64,
                                   kernel_size=(3, 3), padding=(1, 1), activation="relu",
                                   frame_size=(64, 64), cell_type='RNN').to(device)

        if sub_dir != None:
            state_dict = torch.load(model_seq2latent_path, map_location=device)
            model.seq2latent.load_state_dict(state_dict)
            state_dict = torch.load(model_latent2nextframe_path, map_location=device)
            model.latent2nextframe.load_state_dict(state_dict)

        sum_output = np.zeros((10, 64, 64))
        for trial in range(NR_TRIALS_PER_MODEL):
            input = torch.rand(1, 10, 64, 64).unsqueeze(0).to(device)
            input.requires_grad_(True)

            activation_dictionary = {}
            layer_name = 'conv_output'
            model.latent2nextframe.conv_output.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

            output = amu.act_max(network=model, input=input, layer_activation=activation_dictionary,
                                 layer_name=layer_name, unit=unit, device=device, steps=NR_STEPS_PER_TRIAL, alpha=alpha,
                                 verbose=verbose, L2_Decay=L2_Decay, Gaussian_Blur=Gaussian_Blur, Norm_Crop=Norm_Crop,
                                 Contrib_Crop=Contrib_Crop, scale=scale, clip_image=clip_image)

            output = output[:10, :, :]

            sum_output += output.squeeze().detach().numpy()

            if trial % 2 == 0:
                print(f'{trial}/{NR_TRIALS_PER_MODEL} trials done for model {model_name}, sub-model {model_nr}/5')

            averaged_output = sum_output / NR_TRIALS_PER_MODEL

        os.makedirs(f'{OUTPUT_DIR}/{model_name}', exist_ok=True)
        np.save(f'{OUTPUT_DIR}/{model_name}/sum_output_model{model_nr}.npy', sum_output)
        np.save(f'{OUTPUT_DIR}/{model_name}/averaged_output_model{model_nr}.npy', averaged_output)
        five_model_average += averaged_output

    five_model_average = five_model_average / 5
    np.save(f'{OUTPUT_DIR}/{model_name}/five_model_average.npy', five_model_average)
    return five_model_average


# Here the actual calculation is happening
averaged_output = avrg_receptive_field_for_one_model(model_name, model_dir)
