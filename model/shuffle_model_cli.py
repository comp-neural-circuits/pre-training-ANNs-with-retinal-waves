import torch
import os
import argparse
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-i1", "--Input1", help="Path to the Seq2Latent model .pth object that you want to shuffle")
parser.add_argument("-i2", "--Input2", help="Path to the Latent2NextFrame model .pth object that you want to shuffle")
parser.add_argument("-o", "--Output", help="Path to the directory where shuffled files should be saved; must not exist")

args = parser.parse_args()

# check if the results directory exists, if yes throw an error; this is to avoid overwriting results
if os.path.exists(args.Output):
    raise Exception('Output directory already exists, please delete it or specify a different directory')
else:  # create the results directory
    os.mkdir(args.Output)

SEQ2LATENT_MODEL_PATH = args.Input1
LATENT2NEXTFRAME_MODEL_PATH = args.Input2
SHUFFLED_DIR = args.Output

# Load the models
seq2latent = torch.load(SEQ2LATENT_MODEL_PATH)
latent2nextframe = torch.load(LATENT2NEXTFRAME_MODEL_PATH)

# get file names
seq2latent_name = os.path.basename(SEQ2LATENT_MODEL_PATH)
latent2nextframe_name = os.path.basename(LATENT2NEXTFRAME_MODEL_PATH)

# Shuffle the weights of the models within each layer
for model, filename, input_path in zip([seq2latent, latent2nextframe], [seq2latent_name, latent2nextframe_name],
                                       [SEQ2LATENT_MODEL_PATH, LATENT2NEXTFRAME_MODEL_PATH]):
    for param_name in model.keys():
        if "num_batches_tracked" in param_name:
            print(f'Could not shuffle {param_name}')
            continue
        new_param = torch.Tensor(np.random.permutation(model[param_name].cpu().numpy()))
        model[param_name] = new_param

    # Save the model with shuffled weights
    torch.save(model, f'{SHUFFLED_DIR}/SHUFFLED_{filename}')
