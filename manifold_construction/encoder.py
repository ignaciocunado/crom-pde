import random
import math
from utils import *
import lightning as L
import torch
from torch import nn



"""
For the encoder network eθe, the input is a 1D vector of length P with d channels. The output is
a vector of dimension r. Using MLPs would lead to enormous network size when P is large. We
therefore design an encoder network first to apply multiple 1D convolution layers of kernel size 6,
stride size 4, and output channel size d until the output 1D vector’s length is the closest to 32/d but
no smaller. Afterward, the encoder network reshapes the vector into 1 channel and applies an MLP
layer to reduce the dimension of the vector to 32. The last MLP layer then transforms the previous
32-dimensional vector into dimension r. Discretization-specific encoder networks can also be used,
e.g., PointNet for point clouds and convolutional neural network for grid data.
"""


class Encoder(L.LightningModule):
    def __init__(self, data_format, strides, kernel_size, label_length):
        super(Encoder, self).__init__()

        self.strides = strides
        self.kernel_size = kernel_size
        self.label_length = label_length

        self.convolution_layers = nn.ModuleList()
        l_in = data_format['npoints']

        while True:
            l_out = math.floor(float(l_in - (self.kernel_size - 1) - 1) / self.strides + 1)
            if data_format['o_dim'] * l_out >= 32:
                l_in = l_out
                self.convolution_layers.append(
                    nn.Conv1d(data_format['o_dim'], data_format['o_dim'], self.kernel_size, self.strides))
            else:
                break

        self.enc10 = nn.Linear(data_format['o_dim'] * l_in, 32)
        self.enc11 = nn.Linear(32, self.label_length)

        self.standardizeQ = standardizeQ(data_format)
        self.act = Activation()

        self.init_weights()

    def forward(self, state):
        state = self.standardizeQ(state)

        state = torch.transpose(state, 1, 2)

        for layer in self.convolution_layers:
            state = self.act(layer(state))

        state = torch.transpose(state, 1, 2)

        state = state.reshape(-1, state.size(1) * state.size(2))
        state = self.act(self.enc10(state))
        xhat = self.act(self.enc11(state))
        xhat = xhat.view(xhat.size(0), 1, xhat.size(1))

        return xhat

    def init_weights(self):
        with torch.no_grad():
            for m in self.children():
                random.seed(0)
                torch.manual_seed(0)

                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)

                elif type(m) == nn.ModuleList:
                    for c in m:
                        nn.init.xavier_uniform_(c.weight)

                torch.manual_seed(torch.initial_seed())

class standardizeQ(nn.Module):
    def __init__(self, data_format):
        super(standardizeQ, self).__init__()
        self.register_buffer('mean_q_torch', torch.zeros(data_format['o_dim']))
        self.register_buffer('std_q_torch', torch.zeros(data_format['o_dim']))

    def set_params(self, preprop_params):
        self.mean_q_torch = torch.from_numpy(preprop_params['mean_q']).float()
        self.std_q_torch = torch.from_numpy(preprop_params['std_q']).float()

    def forward(self, q):
        return (q - self.mean_q_torch) / self.std_q_torch