import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import random


def collate_next_frame(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.FloatTensor(np.array(batch)).unsqueeze(1)
    batch = batch.to(device)
    return batch[:, :, :10], batch[:, :, 10]


def get_next_frame_dataloader(image_dir, batch_size):
    data_list = []
    # Iterate through all the files in the image directory, print every 100th file
    nr_files, file_counter = len(os.listdir(image_dir)), 1
    for filename in os.listdir(image_dir):
        if (file_counter - 1) % 100 == 0:
            print(f"Loading file {file_counter} of {nr_files}...")
        f = os.path.join(image_dir, filename)
        data_list.append(np.load(f))
        file_counter += 1

    random.seed(42)
    random.shuffle(data_list)

    print(f"Dataset with {len(data_list)} samples loaded.")

    dataloader = DataLoader(data_list, shuffle=True, batch_size=batch_size, collate_fn=collate_next_frame)
    return dataloader
