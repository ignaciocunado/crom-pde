import argparse
import os

import lightning as L
import numpy as np
import torch
import torch.optim as optim
from lightning import Trainer
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from decoder import Decoder
from encoder import Encoder
from data_modules import DataState, ManifoldConstructionDataModule
from utils import convertInputFilenameIntoOutputFilename


class FullNetwork(L.LightningModule):
    def __init__(self, data_format, preprop_params, example_input_array, initial_lr, batch_size, label_length, scale, kernel_size,
                 strides, learning_rate_scaling, epo, loaded_from=None):
        super(FullNetwork, self).__init__()

        self.data_format = data_format
        self.example_input_array = example_input_array
        self.preprop_params = preprop_params

        self.lr = initial_lr
        self.batch_size = batch_size

        self.learning_rates, self.accumulated_epochs = self.generateEPOCHS(learning_rate_scaling, epo)

        self.loaded_from = loaded_from

        self.label_length = label_length
        self.scale = scale
        self.kernel_size = kernel_size
        self.strides = strides

        self.criterion = nn.MSELoss()

        self.encoder = Encoder(data_format, self.strides, self.kernel_size, self.label_length)
        self.decoder = Decoder(data_format, self.strides, self.kernel_size, self.scale, self.label_length)

        self.sim_state_list = []

        self.save_hyperparameters()

    def setup(self, stage):
        if stage == "fit":
            self.decoder.invStandardizeQ.set_params(self.preprop_params)
            self.decoder.prepare.set_params(self.preprop_params)
            self.encoder.standardizeQ.set_params(self.preprop_params)

        if stage == "test":
            self.path_basename = os.path.split(os.path.dirname(self.loaded_from))[-1]

    def training_step(self, train_batch, batch_idx):
        encoder_input = train_batch['encoder_input']
        q = train_batch['q']
        outputs_local, _, _ = self.forward(encoder_input)
        loss = 1000 * self.criterion(outputs_local, q)

        tensorboard_logs = {'train_loss_step': loss}

        self.log_dict(tensorboard_logs, prog_bar=True)

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, test_batch, batch_idx):
        encoder_input = test_batch['encoder_input']
        q = test_batch['q']
        outputs_local, labels, _ = self.forward(encoder_input)
        loss = 1000 * self.criterion(outputs_local, q)
        self.log('test_loss', loss)

        labels = labels.detach().cpu().numpy()

        batch_size_local = encoder_input.size(0)

        output_regular = outputs_local.detach().cpu().numpy()
        x_regular = test_batch['x'].detach().cpu().numpy().astype(float)
        filenames = test_batch['filename']
        times = test_batch['time'].detach().cpu().numpy()


        for i in range(batch_size_local):
            label = labels[i, :]
            input_q = output_regular[i, :, :]
            input_x = x_regular[i, :, :]
            input_t = np.array([[times[i]]])
            filename = filenames[i]
            filename_out = convertInputFilenameIntoOutputFilename(filename, self.path_basename)
            sim_state = DataState(filename_out, False, input_x, input_q, input_t, label=label)
            self.sim_state_list.append(sim_state)

        return loss

    def forward(self, x):
        x = x

        # encoder input
        state = x[:, :, :self.data_format['o_dim']]

        # x0 for decoder
        x0 = x[:, :, self.data_format['o_dim']:]

        # encoder -> xhat
        xhat = self.encoder.forward(state)

        # Store label
        label = xhat.view(xhat.size(0), xhat.size(2))

        # decoder input
        xhat = xhat.expand(xhat.size(0), self.data_format['npoints'], xhat.size(2))
        x = torch.cat((xhat, x0), 2)

        # store original shape for later & reshape for decoder
        batch_size_local = x.size(0)
        x = x.view(x.size(0) * x.size(1), x.size(2))

        # Store decoder for computing Jacobian
        decoder_input = x

        # decoder -> x
        x = self.decoder.forward(decoder_input)

        # return to original shape
        x = x.view(batch_size_local, -1, x.size(1))

        return x, label, decoder_input

    def adaptiveLRfromRange(self, epoch):
        for idx in range(len(self.accumulated_epochs) - 1):
            do = self.accumulated_epochs[idx]
            up = self.accumulated_epochs[idx + 1]
            if do <= epoch < up:
                return self.learning_rates[idx]
        if epoch == self.accumulated_epochs[-1]:  # last epoch
            return self.learning_rates[-1]
        else:
            exit('invalid epoch for adaptiveLRfromRange')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=self.adaptiveLRfromRange)

        return [optimizer], [scheduler]

    def test_epoch_end(self, outputs):
        for sim_state in self.sim_state_list:
            sim_state.write_to_file()

    def generateEPOCHS(self, learning_rates, epochs):
        assert (len(learning_rates) == len(epochs))
        accumulated_epochs = [0]
        accumulated_epochs.extend(np.cumsum(epochs))
        EPOCH_SIZE = accumulated_epochs[-1]

        return learning_rates, accumulated_epochs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', help='train or test',
                        type=str, required=True)

    args = parser.parse_args()

    batch_size = 16
    strides = 4
    kernel_size = 6
    scale = 20
    label_length = 3
    epochs = [3000, 3000, 3000, 3000, 3000, 1000]
    initial_learning_rate = 1e-4
    learning_rate_scaling = [10, 5, 2, 1, 0.5, 0.2]

    output_path = os.getcwd() + '/manifold_construction/outputs'

    trainer = Trainer(default_root_dir=output_path, max_epochs=int(np.sum(epochs)))

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if args.mode == "train":
        dirs = [str(i) for i in range(0, 8)]
        dm = ManifoldConstructionDataModule()
        data_format, example_input_array = dm.get_dataFormat()
        preprop_params = dm.get_dataParams()

        net = FullNetwork(data_format, preprop_params, example_input_array, initial_learning_rate, batch_size, label_length, scale, kernel_size,
                          strides, learning_rate_scaling, epochs, loaded_from=None)

        net.to(device)

        trainer.fit(net, dm)

    elif args.mode == "reconstruct":
        dirs = [str(i) for i in range(8, 12)]
        dm = ManifoldConstructionDataModule()

        net = FullNetwork.load_from_checkpoint('/output/')

        net.to(device)

        trainer.test(net, dm)

    else:
        exit(1)


if __name__ == "__main__":
    main()
