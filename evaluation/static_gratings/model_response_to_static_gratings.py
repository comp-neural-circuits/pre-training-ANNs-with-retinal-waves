"""
Run this script to generate the response to static gratings of the different RNN models
(trained only with Natural Images (NI) or only pretrained with Retinal Waves datasets (RW) 
or Randomly Initialised (RD)).

For each grating, the corresponding blank sweep is presented and the mean change in activity is computed
with the baseline activity of the blank sweep.

For each neuron, for each grating, the changes in activity are stored in a dictionary and saved in a .npz file, 
as well as the mean response to all stimuli.

Before running this script, make sure to run the script generate_static_gratings.py to generate
the input stimuli (static gratings and blank sweeps) for the different orientations, spatial frequencies and phases.
"""




    
import os
import torch
import copy
import numpy as np
import pickle


from model._next_frame_predictor import NextFramePredictor
from RF_static_gratings.generate_static_gratings import static_gratings_params







def generate_paths(model_name):
    this_path = os.path.dirname(os.path.abspath(__file__))
    path_main = os.path.join(this_path, "RF_static_gratings")
    path_models = os.path.join(this_path, "Pretrained_Models")
    path_stimuli = os.path.join(path_main, "stimuli")
    path_nrn_responses = os.path.join(path_main, "neuron_responses")
    
    if model_name == "RW":
        model_path = "Pretrained_Output/RW_Directional_Dataset/RNN/LM20230706_Pretrain_RNN_Directional_1"
        path_model = os.path.join(path_models, model_path)
        path_responses = os.path.join(path_nrn_responses, model_path)
    elif model_name == "NI":
        model_path = "Training_Output/NI_Maze_Dataset/RNN/LM20230707_Train_RNN_NIMaze_None_1"
        path_model = os.path.join(path_models, model_path)
        path_responses = os.path.join(path_nrn_responses, model_path)
    elif model_name == "RD":
        path_model = None
        path_responses = os.path.join(path_nrn_responses, 'random_init')
    else:
        raise ValueError("Invalid model name. Please choose 'NI', 'RW', or 'RD'.")
    return path_stimuli, path_model, path_responses


def get_model_files(dir_path):
    """
    Get path of file that has "best_model_epoch" and "seq2latent" in its name.
    """
    
    for file in os.listdir(dir_path):
        if 'best_model_epoch' in file and 'seq2latent' in file:
            model_seq2latent_path = f'{dir_path}/{file}'
        elif 'best_model_epoch' in file and 'latent2nextframe' in file:
            model_latent2nextframe_path = f'{dir_path}/{file}'

    return model_seq2latent_path, model_latent2nextframe_path





@torch.no_grad()
def compute_response_to_static_gratings(path_model, path_stimuli, path_responses, device):
    
    # Create and load model
    model = NextFramePredictor(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), cell_type='RNN').to(device)
    
    if path_model is not None:
        model_seq2latent_path, model_latent2nextframe_path = get_model_files(path_model)
        state_dict = torch.load(model_seq2latent_path, map_location=device)
        model.seq2latent.load_state_dict(state_dict)
        state_dict = torch.load(model_latent2nextframe_path, map_location=device)
        model.latent2nextframe.load_state_dict(state_dict)
    
    gratings_dict = np.load(f'{path_stimuli}/static_gratings.npz')
    activities_grating = {}
    activities_blank = {}
    mean_activity_grating = {}
    mean_activity_blank = {}
    changes_in_activity = {}
    
    for o in static_gratings_params['orientation']:
        for sf in static_gratings_params['spatial_frequency']:
            for p in static_gratings_params['phase']:
                
                grating = gratings_dict[f'grating_o_{o}_sf_{sf}_p_{p}']
                blank_sweep = gratings_dict[f'blank_sweep_o_{o}_sf_{sf}_p_{p}']
                
                grating = torch.tensor(grating, dtype=torch.float32).to(device)
                blank_sweep = torch.tensor(blank_sweep, dtype=torch.float32).to(device)
                
                grating = grating.repeat(10, 1, 1).unsqueeze(0).unsqueeze(0)
                blank_sweep = blank_sweep.repeat(10, 1, 1).unsqueeze(0).unsqueeze(0)
                
                # Compute the response to the grating
                model.register_hooks()
                _, activations = model.forward(grating)
                activities_grating[f'o_{o}_sf_{sf}_p_{p}'] = copy.deepcopy(activations)
                model.clear_hooks()
               
                # Compute the response to the blank sweep
                model.register_hooks()
                _, activations = model.forward(blank_sweep)
                activities_blank[f'o_{o}_sf_{sf}_p_{p}'] = copy.deepcopy(activations)
                model.clear_hooks()
                
                layer_names = ["convrecurrent1", "convrecurrent2", "convrecurrent3", "conv_output"]
                for layer in layer_names:
                    activities_grating[f'o_{o}_sf_{sf}_p_{p}'][layer] = np.stack(activities_grating[f'o_{o}_sf_{sf}_p_{p}'][layer], axis=0).squeeze()
                    mean_grating = activities_grating[f'o_{o}_sf_{sf}_p_{p}'][layer].mean(axis=0)
                    mean_activity_grating[f'{layer}_o_{o}_sf_{sf}_p_{p}'] = mean_grating
                    activities_blank[f'o_{o}_sf_{sf}_p_{p}'][layer] = np.stack(activities_blank[f'o_{o}_sf_{sf}_p_{p}'][layer], axis=0).squeeze()
                    mean_blank = activities_blank[f'o_{o}_sf_{sf}_p_{p}'][layer].mean(axis=0)
                    mean_activity_blank[f'{layer}_o_{o}_sf_{sf}_p_{p}'] = mean_blank
                    changes_in_activity[f'{layer}_o_{o}_sf_{sf}_p_{p}'] = mean_grating - mean_blank
    

    with open(f'{path_responses}/nrn_responses_gratings.pkl', 'wb') as f:
        pickle.dump(activities_grating, f)
        
    with open(f'{path_responses}/nrn_responses_blank.pkl', 'wb') as f:
        pickle.dump(activities_blank, f)
        
    with open(f'{path_responses}/mean_activity_grating.pkl', 'wb') as f:
        pickle.dump(mean_activity_grating, f)
        
    with open(f'{path_responses}/mean_activity_blank.pkl', 'wb') as f:
        pickle.dump(mean_activity_blank, f)
        
    with open(f'{path_responses}/changes_in_activity.pkl', 'wb') as f:
        pickle.dump(changes_in_activity, f)
    return None





if __name__ == '__main__':

    model_name = "RW" # "NI" "RW" "RD"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")


    path_stimuli, path_model, path_responses = generate_paths(model_name)
    compute_response_to_static_gratings(path_model=path_model, path_stimuli=path_stimuli, path_responses=path_responses, device=device)