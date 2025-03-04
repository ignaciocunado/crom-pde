import os

import torch
from torch import nn

class Activation(nn.Module):
    def __init__(self, ):
        super(Activation, self).__init__()

    def forward(self, input):
        return nn.functional.elu(input)


def make_linear_grad_of(weight):
    def grad(x):
        # `(N, in\_features)`
        assert (len(x.shape) == 2)
        weight_batch = weight.view(1, weight.size(0), weight.size(1))
        weight_batch = weight_batch.expand(x.size(0), weight.size(0), weight.size(1))
        return weight_batch

    return grad


def make_elu_grad_of(alpha):
    def grad(x):
        # `(N, in\_features)`
        assert (len(x.shape) == 2)
        grad_batch = torch.where(x > 0.0, torch.ones_like(x), alpha * torch.exp(x))
        grad_batch = torch.diag_embed(grad_batch)
        return grad_batch

    return grad


def convertInputFilenameIntoOutputFilename(filename_in, path_basename):
    basename = os.path.basename(filename_in)

    dirname = os.path.dirname(filename_in)
    pardirname = os.path.dirname(dirname)
    pardirname += '_pred'
    pardirname += '_' + path_basename
    dirname = os.path.basename(dirname)

    return os.path.join(pardirname, dirname, basename)


def get_weightPath(trainer):
    return os.getcwd() + "/outputs/weights/{}/epoch={}-step={}.ckpt".format(trainer.logger.version,
                                                                            trainer.current_epoch - 1,
                                                                            trainer.global_step)
