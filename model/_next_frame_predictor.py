import torch
import torch.nn as nn
from _model_parts import Seq2Latent, Latent2NextFrame
import copy


class NextFramePredictor(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, padding, cell_type):
        super(NextFramePredictor, self).__init__()

        self.seq2latent = Seq2Latent(num_channels=num_channels,
                                     num_kernels=num_kernels,
                                     kernel_size=kernel_size,
                                     cell_type=cell_type
                                     )
        self.latent2nextframe = Latent2NextFrame(num_channels=num_channels,
                                                 num_kernels=num_kernels,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 cell_type=cell_type
                                                 )

    def load_models(self, seq2latent_path, latent2nextframe_path, device=None):
        if device is not None:
            state_dict = torch.load(seq2latent_path, map_location=torch.device(device))
        else:
            state_dict = torch.load(seq2latent_path)
        self.seq2latent.load_state_dict(state_dict)

        if device is not None:
            state_dict = torch.load(latent2nextframe_path, map_location=torch.device(device))
        else:
            state_dict = torch.load(latent2nextframe_path)
        self.latent2nextframe.load_state_dict(state_dict)

    def save_models(self, path):
        torch.save(copy.deepcopy(self.seq2latent.state_dict()), f'{path}seq2latent.pth')
        torch.save(copy.deepcopy(self.latent2nextframe.state_dict()), f'{path}latent2nextframe.pth')

    def forward(self, X):
        output = self.seq2latent(X)
        output = self.latent2nextframe(output)
        return output
