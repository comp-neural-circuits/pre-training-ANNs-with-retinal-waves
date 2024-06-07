import torch
import torch.nn as nn
from model._conv_recurrent_cells import Conv2dRNNCell, Conv2dLSTMCell, Conv2dGRUCell


class Seq2Latent(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, cell_type):
        super(Seq2Latent, self).__init__()
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module("convrecurrent1",
                                   ConvRecurrentLayer(in_channels=num_channels, out_channels=num_kernels,
                                                      kernel_size=kernel_size, cell_type=cell_type))

        self.sequential.add_module("batchnorm1", nn.BatchNorm3d(num_features=num_kernels))

        self.sequential.add_module(f"convrecurrent2",
                                   ConvRecurrentLayer(in_channels=num_kernels, out_channels=num_kernels,
                                                      kernel_size=kernel_size, cell_type=cell_type))

        self.sequential.add_module(f"batchnorm2", nn.BatchNorm3d(num_features=num_kernels))

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)
        return output


class Latent2NextFrame(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, padding, cell_type):
        super(Latent2NextFrame, self).__init__()
        self.sequential = nn.Sequential()

        self.sequential.add_module(f"convrecurrent3",
                                   ConvRecurrentLayer(in_channels=num_kernels, out_channels=num_kernels,
                                                      kernel_size=kernel_size, cell_type=cell_type))

        self.sequential.add_module(f"batchnorm3", nn.BatchNorm3d(num_features=num_kernels))

        # Add Convolutional Layer to predict output frame
        self.conv_output = nn.Conv2d(in_channels=num_kernels, out_channels=num_channels,
                                     kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv_output(output[:, :, -1])
        return nn.Sigmoid()(output)


class ConvRecurrentLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, cell_type):
        super(ConvRecurrentLayer, self).__init__()

        self.out_channels = out_channels
        self.cell_type = cell_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # We will unroll this over time steps
        if cell_type == 'LSTM':
            self.convCell = Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        elif cell_type == 'GRU':
            self.convCell = Conv2dGRUCell(in_channels, out_channels, kernel_size)
        elif cell_type == 'RNN':
            self.convCell = Conv2dRNNCell(in_channels, out_channels, kernel_size)
        else:
            raise ValueError('Invalid Cell Type')

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device=self.device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device=self.device)

        if self.cell_type == 'LSTM':
            # Initialize Cell Input
            C = torch.zeros(batch_size, self.out_channels,
                            height, width, device=self.device)

        # Unroll over time steps
        for time_step in range(seq_len):
            if self.cell_type == 'LSTM':
                H, C = self.convCell(X[:, :, time_step], H, C)
            else:
                H = self.convCell(X[:, :, time_step], H)

            output[:, :, time_step] = H

        return output
