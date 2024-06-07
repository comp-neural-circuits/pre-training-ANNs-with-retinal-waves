import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from _next_frame_predictor import NextFramePredictor
import seaborn as sns
from _data_loading import *
import os
import copy
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--Dataset",
                    help="Path to the directory where training and validation set can be found; "
                         "it is expected that the directory contains two subdirectories: training and validation")
parser.add_argument("-r", "--Results", help="Path to the directory where output files should be saved")
parser.add_argument("-c", "--Cell", help="Cell type of RNN to use: LSTM, GRU, RNN")
parser.add_argument("-m1", "--Model1", help="Path to the Seq2Latent model .pth object")
parser.add_argument("-m2", "--Model2", help="Path to the Latent2NextFrame model .pth object")

args = parser.parse_args()

# check if the results directory exists, if yes throw an error; this is to avoid overwriting results
if os.path.exists(args.Results):
    raise Exception('Results directory already exists, please delete it or specify a different directory')
else:  # create the results directory
    os.mkdir(args.Results)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize parser arguments
DATASET_PATH = args.Dataset
RESULTS_PATH = args.Results
CELL_TYPE = args.Cell
SEQ2LATENT_MODEL_PATH = args.Model1
LATENT2NEXTFRAME_MODEL_PATH = args.Model2

# Initialize hyperparameters
BATCH_SIZE = 16

train_loader = get_next_frame_dataloader(f'{DATASET_PATH}/training', batch_size=BATCH_SIZE)
val_loader = get_next_frame_dataloader(f'{DATASET_PATH}/validation', batch_size=BATCH_SIZE)

print('Dataloader created!')

# The input video frames are grayscale, thus single channel
model = NextFramePredictor(num_channels=1,
                           num_kernels=64,
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           cell_type=CELL_TYPE
                           ).to(device)

if SEQ2LATENT_MODEL_PATH != "None":
    model.load_models(SEQ2LATENT_MODEL_PATH, LATENT2NEXTFRAME_MODEL_PATH)
    print('Models loaded!')

optim = Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss(reduction='sum')  # Use sum as reduction, because the values are between 0 and 1

print(f'{CELL_TYPE} model initialized!')

with open(f"{RESULTS_PATH}/loss.txt", "a") as file_object:
    file_object.write(CELL_TYPE)

# Create file to save model for each epoch
if not os.path.exists(f'{RESULTS_PATH}/saved_models_per_epoch'):
    os.makedirs(f'{RESULTS_PATH}/saved_models_per_epoch')


def visualize_predictions(output, target, epoch):
    if not os.path.exists(f'{RESULTS_PATH}/pred_vis_epoch_{epoch}'):
        os.makedirs(f'{RESULTS_PATH}/pred_vis_epoch_{epoch}')
    if f'{device}' == 'cuda':
        output_batch = output.cpu().detach().numpy()
        target_batch = target.cpu().detach().numpy()
    else:
        output_batch = output.detach().numpy()
        target_batch = target.detach().numpy()
    out_nr = 0
    for out, target in zip(output_batch, target_batch):
        plt.rcParams["figure.figsize"] = (11, 4)

        plt.subplot(1, 2, 1)
        o = out.squeeze()
        sns.heatmap(o, vmin=0.0, vmax=1.0, cmap='binary', square=True)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.title('Output')

        plt.subplot(1, 2, 2)
        tar = target.squeeze()
        sns.heatmap(tar, vmin=0.0, vmax=1.0, cmap='binary', square=True)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.title('Ground Truth')

        plt.savefig(f'{RESULTS_PATH}/pred_vis_epoch_{epoch}/prediction_{out_nr}.png')
        plt.close()
        out_nr += 1


train_loss_list, val_loss_list = [], []
min_val_loss = np.Inf
num_epochs = 200
epochs_no_improvement = 0
n_epochs_stop = 5
best_state_dict = None
best_epoch = -1

print('Starting to train...')
for epoch in range(0, num_epochs + 1):
    if epoch >= 1:
        print(f'\n=====Training Epoch {epoch}=====')
        train_loss = 0
        total_samples_trained = 0
        correct_samples_trained = 0
        model.train()
        # Iterate over all batches in one epoch
        for batch_num, (input, target) in enumerate(train_loader, 1):
            output = model(input)
            loss = criterion(output.flatten(), target.flatten())
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss += loss.item()

            if batch_num % 20 == 0:
                print(f'Train Loss: {loss.item() / target.size(0)} \t '
                      f'batch:{batch_num}/{int(len(train_loader.dataset) / BATCH_SIZE)}')
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)

    # Validation after each epoch
    val_loss = 0
    #total_samples_validated, correct_samples_validated = 0, 0
    model.eval()
    with torch.no_grad():
        for input, target in val_loader:
            output = model(input)
            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)

    if epoch == 0:
        print("Epoch:{} Validation Loss:{:.2f}\n".format(epoch, val_loss))
        with open(f"{RESULTS_PATH}/loss.txt", "a") as file_object:
            file_object.write(f"Epoch: {epoch}, Val loss: {val_loss:>8f} \n")

    else:
        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}".format(epoch, train_loss, val_loss))
        with open(f"{RESULTS_PATH}/loss.txt", "a") as file_object:
            file_object.write(f"Epoch: {epoch}, Train loss: {train_loss:>8f}, Val loss: {val_loss:>8f} \n")

    visualize_predictions(output, target, epoch)

    torch.save(copy.deepcopy(model.state_dict()), f'{RESULTS_PATH}/saved_models_per_epoch/model_epoch_{epoch}_full.pth')
    model.save_models(f'{RESULTS_PATH}/saved_models_per_epoch/model_epoch_{epoch}_')

    # early stopping
    if val_loss < min_val_loss:  # epoch improved loss -> new best epoch
        min_val_loss = val_loss
        epochs_no_improvement = 0
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = epoch
    else:  # no improvement
        epochs_no_improvement += 1
        if epochs_no_improvement == n_epochs_stop:
            break  # -> early stopping

# save best model
torch.save(best_state_dict, f'{RESULTS_PATH}/best_model_epoch_{best_epoch}_full.pth')
model.save_models(f'{RESULTS_PATH}/best_model_epoch_{best_epoch}_')

try:
    plt.rcParams['figure.dpi'] = 200
    plt.figure(figsize=(4, 4))
    plt.plot(range(0, (epoch + 1)), train_loss_list, label='train MSE loss')
    plt.plot(range(0, (epoch + 1)), val_loss_list, label='validation MSE loss')
    plt.legend()
    plt.title(f'Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(f'{RESULTS_PATH}/MSE_loss.png')
except:
    pass
