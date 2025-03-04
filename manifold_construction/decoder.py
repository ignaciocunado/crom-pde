import random
from utils import *

import lightning as L
from torch import nn


"""
Consistent with the implicit neural representation literature, we parameterize the manifold-
parameterization function manifold with an MLP network gθg. The input dimension of the network is
m+ r while the output dimension of the network is d, where mis the dimension of the input spatial
vector while dis the output dimension of the vector field of interest. Our MLP network contains 5
hidden layers, each of which has a width of (β·d), where β is the hyperparameter that defines the
learning capacity of the network. Essentially, our network has two tunable hyperparameters, rand β.
A detailed study on these hyperparameters will be discussed in Appendix F.
Since we require the network to be continuously differentiable with respect to both the spatial
coordinates x and the latent space vector q, we adopt continuously differentiable activation functions.
In practice, either ELU (Clevert et al., 2015) or SIREN (Sitzmann et al., 2020b) serves this purpose.
"""


class Decoder(L.LightningModule):

    def __init__(self, data_format, strides, kernel_size, scale_mlp, label_length):
        super(Decoder, self).__init__()

        self.strides = strides
        self.kernel_size = kernel_size
        self.scale_mlp = scale_mlp
        self.label_length = label_length

        self.dec0 = nn.Linear(self.label_length + data_format['i_dim'], data_format['i_dim'] * scale_mlp)
        self.dec1 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec2 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec3 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec4 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.out = nn.Linear(data_format['i_dim'] * scale_mlp,
                             data_format['o_dim'])

        self.act = Activation()
        self.invStandardizeQ = invStandardizeQ(data_format)
        self.prepare = Prepare(self.label_length, data_format)

        self.layers = []
        self.layers.append(self.prepare)
        self.layers.append(self.dec0)
        self.layers.append(self.act)
        self.layers.append(self.dec1)
        self.layers.append(self.act)
        self.layers.append(self.dec2)
        self.layers.append(self.act)
        self.layers.append(self.dec3)
        self.layers.append(self.act)
        self.layers.append(self.dec4)
        self.layers.append(self.act)
        self.layers.append(self.out)
        self.layers.append(self.invStandardizeQ)

        self.init_weights()
        self.init_grads()

    def init_weights(self):
        with torch.no_grad():
            for layer in self.layers:
                if type(layer) == nn.Linear:
                    random.seed(0)
                    torch.manual_seed(0)

                    nn.init.xavier_uniform_(layer.weight)

    def init_grads(self):
        for layer in self.layers:
            if type(layer) == nn.Linear:
                layer.grad_func = make_linear_grad_of(layer.weight)
            elif type(layer) == Activation:
                layer.grad_func = make_elu_grad_of(1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def computeJacobianFullAnalytical(self, x):
        y = x.detach().clone()
        grad = None
        for layer in self.layers:
            if grad is None:
                grad = layer.grad_func(y)
            else:
                grad = torch.matmul(layer.grad_func(y), grad)
            y = layer(y)
            x = layer(x)
        return grad, x

class invStandardizeQ(nn.Module):
    def __init__(self, data_format):
        super(invStandardizeQ, self).__init__()

        self.register_buffer('mean_q_torch', torch.zeros(data_format['o_dim']))
        self.register_buffer('std_q_torch', torch.zeros(data_format['o_dim']))

    def set_params(self, preprop_params):
        self.mean_q_torch = torch.from_numpy(preprop_params['mean_q']).float()
        self.std_q_torch = torch.from_numpy(preprop_params['std_q']).float()

    def forward(self, q_standardized):
        return self.mean_q_torch + q_standardized * self.std_q_torch

    def grad_func(self, x):
        # `(N, in\_features)`
        assert (len(x.shape) == 2)
        grad_batch = self.std_q_torch.expand_as(x)
        grad_batch = torch.diag_embed(grad_batch)
        return grad_batch


class Prepare(nn.Module):
    def __init__(self, lbllength, data_format):
        super(Prepare, self).__init__()
        self.lbllength = lbllength

        self.register_buffer('min_x_torch', torch.zeros(data_format['i_dim']))
        self.register_buffer('max_x_torch', torch.zeros(data_format['i_dim']))
        self.register_buffer('mean_x_torch', torch.zeros(data_format['i_dim']))
        self.register_buffer('std_x_torch', torch.zeros(data_format['i_dim']))

    def set_params(self, preprop_params):
        self.min_x_torch = torch.from_numpy(preprop_params['min_x']).float()
        self.max_x_torch = torch.from_numpy(preprop_params['max_x']).float()
        self.mean_x_torch = torch.from_numpy(preprop_params['mean_x']).float()
        self.std_x_torch = torch.from_numpy(preprop_params['std_x']).float()

    def forward(self, x):
        xhat = x[:, :self.lbllength]

        x0 = x[:, self.lbllength:]
        x0 = self.prep(x0)

        x = torch.cat((xhat, x0), 1)

        return x

    def clipX(self, x):
        return 2 * (x - self.min_x_torch) / (self.max_x_torch - self.min_x_torch) - 1

    def standardizeX(self, x):
        return (x - self.mean_x_torch) / self.std_x_torch

    def prep(self, x):
        return self.standardizeX(x)

    def grad_func(self, x):
        # `(N, in\_features)`
        assert (len(x.shape) == 2)

        xhat = x[:, :self.lbllength]
        x0 = x[:, self.lbllength:]

        with torch.no_grad():
            grad_xhat = torch.Tensor([1]).type_as(x)
        grad_xhat = grad_xhat.expand_as(xhat)

        multi = torch.reciprocal(self.std_x_torch)
        grad_x0 = multi.expand_as(x0)

        grad_batch = torch.cat((grad_xhat, grad_x0), 1)
        grad_batch = torch.diag_embed(grad_batch)

        return grad_batch