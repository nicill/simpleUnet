import time
import numpy as np
from functools import partial
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from layers import AttentionGate2D, DownsampledMultiheadAttention2D
from utils import time_to_string


def compute_filters(n_inputs, conv_filters):
    conv_in = [n_inputs] + conv_filters[:-2]
    conv_out = conv_filters[:-1]
    down_out = conv_filters[-2::-1]
    up_out = conv_filters[:0:-1]
    deconv_in = list(map(sum, zip(down_out, up_out)))
    deconv_out = down_out
    return conv_in, conv_out, deconv_in, deconv_out


class BaseModel(nn.Module):
    """"
    This is the baseline model to be used for any of my networks. The idea
    of this model is to create a basic framework that works similarly to
    keras, but flexible enough.
    For that reason, I have "embedded" the typical pytorch main loop into a
    fit function and I have defined some intermediate functions and callbacks
    to alter the main loop. By itself, this model can train any "normal"
    network with different losses and scores for training and validation.
    It can be easily extended to create adversarial networks (which I have done
    in other repositories) and probably to other more complex problems.
    The network also includes some print functions to check the current status.
    """
    def __init__(self):
        """
        Main init. By default some parameters are set, but they should be
        redefined on networks inheriting that model.
        """
        super().__init__()
        # Init values
        self.init = True
        self.optimizer_alg = None
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.dropout = 0
        self.final_dropout = 0
        self.ann_rate = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.best_state = None
        self.best_opt = None
        self.train_functions = [
            {'name': 'train', 'weight': 1, 'f': None},
        ]
        self.val_functions = [
            {'name': 'val', 'weight': 1, 'f': None},
        ]
        self.acc_functions = {}
        self.acc = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.log_file = None
        self.batch_file = None

    def forward(self, *inputs):
        """

        :param inputs: Inputs to the forward function. We are passing the
         contents by reference, so if there are more than one input, they
         will be separated.
        :return: Nothing. This has to be reimplemented for any class.
        """
        return None

    def mini_batch_loop(
            self, data, train=True
    ):
        """
        This is the main loop. It's "generic" enough to account for multiple
        types of data (target and input) and it differentiates between
        training and testing. While inherently all networks have a training
        state to check, here the difference is applied to the kind of data
        being used (is it the validation data or the training data?). Why am
        I doing this? Because there might be different metrics for each type
        of data. There is also the fact that for training, I really don't care
        about the values of the losses, since I only want to see how the global
        value updates, while I want both (the losses and the global one) for
        validation.
        :param data: Dataloader for the network.
        :param train: Whether to use the training dataloader or the validation
         one.
        :return:
        """
        losses = list()
        mid_losses = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            # In case we are training the the gradient to zero.
            if self.training:
                self.optimizer_alg.zero_grad()

            # First, we do a forward pass through the network.
            if isinstance(x, list) or isinstance(x, tuple):
                x_cuda = tuple(x_i.to(self.device) for x_i in x)
                pred_labels = self(*x_cuda)
            else:
                pred_labels = self(x.to(self.device))
            if isinstance(y, list) or isinstance(y, tuple):
                y_cuda = tuple(y_i.to(self.device) for y_i in y)
            else:
                y_cuda = y.to(self.device)

            # After that, we can compute the relevant losses.
            if train:
                # Training losses (applied to the training data)
                batch_losses = [
                    l_f['weight'] * l_f['f'](pred_labels, y_cuda)
                    for l_f in self.train_functions
                ]
                batch_loss = sum(batch_losses)
                if self.training:
                    batch_loss.backward()
                    self.optimizer_alg.step()
                    self.batch_update(batch_i, len(data))

            else:
                # Validation losses (applied to the validation data)
                batch_losses = [
                    l_f['f'](pred_labels, y_cuda)
                    for l_f in self.val_functions
                ]
                batch_loss = sum([
                    l_f['weight'] * l
                    for l_f, l in zip(self.val_functions, batch_losses)
                ])
                mid_losses.append([loss.tolist() for loss in batch_losses])

            # It's important to compute the global loss in both cases.
            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout / Adaptive dropout
            # Here we could modify dropout to be updated for each batch.
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )

        # Mean loss of the global loss (we don't need the loss for each batch).
        mean_loss = np.mean(losses)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            return mean_loss, mean_losses

    def fit(
            self,
            train_loader,
            val_loader,
            test_loader=None,
            epochs=100,
            patience=20,
            log_file=None,
            verbose=True
    ):
        # Init
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_file = log_file
        best_e = 0
        l_names = ['train', ' val '] + [
            '{:^6s}'.format(l_f['name'][:6])
            for l_f in self.val_functions
        ]
        l_bars = '--|--'.join(
            ['-' * 5] * 2 +
            ['-' * 6] * len(l_names[2:])
        )
        l_hdr = '  |  '.join(l_names)
        # Since we haven't trained the network yet, we'll assume that the
        # initial values are the best ones.
        self.best_state = deepcopy(self.state_dict())
        t_start = time.time()

        # We'll just take the maximum losses and accuracies (inf, -inf)
        # and print the headers.
        print('\033[K', end='')
        print('Epoch num |  {:}  |'.format(l_hdr))
        print('----------|--{:}--|'.format(l_bars))
        best_loss_tr = [np.inf] * len(self.val_functions)
        best_loss_val = [np.inf] * len(self.val_functions)

        if log_file is not None:
            log_file.writerow(
                ['Epoch', 'train', 'val'] + [
                    'train_' + l_f['name']
                    for l_f in self.val_functions
                ] + [
                    'val_' + l_f['name']
                    for l_f in self.val_functions
                ] + ['time']
            )

        # We are looking for the output, without training, so no need to
        # use grad.
        with torch.no_grad():
            loss_tr, best_loss_tr, _, mid_tr = self.validate(
                self.train_loader, best_loss_tr
            )

            loss_val, best_loss_val, losses_val_s, mid_val = self.validate(
                self.val_loader, best_loss_val
            )

            # Doing this also helps setting an initial best loss for all
            # the necessary losses.
            if verbose:
                # This is just the print for each epoch, but including the
                # header.
                # Mid losses check
                t_out = time.time() - self.t_val
                t_s = time_to_string(t_out)

                epoch_s = '\033[K\033[32mInit     \033[0m'
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
                loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_val)
                final_s = ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] + losses_val_s + [t_s]
                )
                print(final_s)
            if log_file is not None:
                log_file.writerow(
                    [
                        'Init',
                        '{:7.4f}'.format(loss_tr),
                        '{:7.4f}'.format(loss_val)
                    ] + mid_tr.tolist() + mid_val.tolist() + [t_s]
                )

        no_improvement = 0
        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            # First we train and check if there has been an improvement.
            # with torch.autograd.detect_anomaly():
            #     loss_tr = self.mini_batch_loop(self.train_loader)
            loss_tr = self.mini_batch_loop(self.train_loader)
            improvement_tr = self.best_loss_tr > loss_tr
            if improvement_tr:
                self.best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.4f}'.format(loss_tr)

            # Then we validate and check all the losses
            _, best_loss_tr, _, mid_tr = self.validate(
                self.train_loader, best_loss_tr
            )

            loss_val, best_loss_val, losses_val_s, mid_val = self.validate(
                self.val_loader, best_loss_val
            )

            # Patience check
            # We check the patience to stop early if the network is not
            # improving. Otherwise we are wasting resources and time.
            improvement_val = self.best_loss_val > loss_val
            loss_s = '{:7.4f}'.format(loss_val)
            if improvement_val:
                self.best_loss_val = loss_val
                epoch_s = '\033[32mEpoch {:03d}\033[0m'.format(self.epoch)
                loss_s = '\033[32m{:}\033[0m'.format(loss_s)
                best_e = self.epoch
                self.best_state = deepcopy(self.state_dict())
                no_improvement = 0
            else:
                epoch_s = 'Epoch {:03d}'.format(self.epoch)
                no_improvement += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            if verbose:
                print('\033[K', end='', flush=True)
                final_s = ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] + losses_val_s + [t_s]
                )
                print(final_s)
            if self.log_file is not None:
                self.log_file.writerow(
                    [
                        'Epoch {:03d}'.format(self.epoch),
                        '{:7.4f}'.format(loss_tr),
                        '{:7.4f}'.format(loss_val)
                    ] + mid_tr.tolist() + mid_val.tolist() + [t_s]
                )

            self.epoch_update(epochs)

            if no_improvement == patience:
                break

        self.epoch = best_e
        self.load_state_dict(self.best_state)
        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                    'Training finished in {:} epochs ({:}) '
                    'with minimum validation loss = {:f} '
                    '(epoch {:03d})'.format(
                        self.epoch + 1, t_end_s, self.best_loss_val, best_e
                    )
            )

    def validate(self, data, best_loss=None):
        with torch.no_grad():
            self.t_val = time.time()
            self.eval()
            loss, mid_losses = self.mini_batch_loop(
                data, False
            )

        # Mid losses check
        if best_loss is not None:
            losses_s = [
                '\033[36m{:8.5f}\033[0m'.format(l) if bl > l
                else '{:8.5f}'.format(l) for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            best_loss = [
                l if bl > l else bl for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            return loss, best_loss, losses_s, mid_losses
        else:
            return loss, mid_losses

    def epoch_update(self, epochs):
        """
        Callback function to update something on the model after the epoch
        is finished. To be reimplemented if necessary.
        :param epochs: Maximum number of epochs
        :return: Nothing.
        """
        return None

    def batch_update(self, batch, batches):
        """
        Callback function to update something on the model after the batch
        is finished. To be reimplemented if necessary.
        :param batch: Current batch
        :param batches: Maximum number of epochs
        :return: Nothing.
        """
        return None

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss):
        """
        Function to print the progress of a batch. It takes into account
        whether we are training or validating and uses different colors to
        show that. It's based on Keras arrow progress bar, but it only shows
        the current (and current mean) training loss, elapsed time and ETA.
        :param batch_i: Current batch number.
        :param n_batches: Total number of batches.
        :param b_loss: Current loss.
        :param mean_loss: Current mean loss.
        :return: None.
        """
        init_c = '\033[0m' if self.training else '\033[38;5;238m'
        percent = 25 * (batch_i + 1) // n_batches
        progress_s = ''.join(['█'] * percent)
        remainder_s = ''.join([' '] * (25 - percent))
        loss_name = 'train_loss' if self.training else 'val_loss'

        if self.training:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)
        epoch_hdr = '{:}Epoch {:03} ({:03d}/{:03d} - {:05.2f}%) [{:}] '
        loss_s = '{:} {:f} ({:f}) {:} / ETA {:}'
        batch_s = (epoch_hdr + loss_s).format(
            init_c, self.epoch, batch_i + 1, n_batches,
            100 * (batch_i + 1) / n_batches, progress_s + remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

    def freeze(self):
        """
        Method to freeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Method to unfreeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = True

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class BaseConv2dBlock(BaseModel):
    def __init__(self, filters_in, filters_out, kernel):
        super().__init__()
        self.conv = partial(
            nn.Conv2d, kernel_size=kernel, padding=kernel // 2
        )

    def forward(self, inputs):
        return self.conv(inputs)

    @staticmethod
    def default_activation(n_filters):
        return nn.ReLU()


class Conv2dBlock(BaseConv2dBlock):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None
    ):
        super().__init__(filters_in, filters_out, kernel)
        if activation is None:
            activation = self.default_activation
        self.block = nn.Sequential(
            self.conv(filters_in, filters_out),
            activation(filters_out),
            norm(filters_out)
        )

    def forward(self, inputs):
        return self.block(inputs)


class DoubleConv2dBlock(BaseConv2dBlock):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None,
    ):
        super().__init__(filters_in, filters_out, kernel)
        if activation is None:
            activation = self.default_activation
        self.block = nn.Sequential(
            self.conv(filters_in, filters_out),
            activation(filters_out),
            norm(filters_out),
            self.conv(filters_out, filters_out),
            activation(filters_out),
            norm(filters_out)
        )

    def forward(self, inputs):
        return self.block(inputs)


class ResConv2dBlock(BaseConv2dBlock):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None
    ):
        super().__init__(filters_in, filters_out, kernel)
        if activation is None:
            activation = self.default_activation
        conv = nn.Conv2d

        self.first = nn.Sequential(
            self.conv(filters_in, filters_out),
            activation(filters_out),
            norm(filters_out)
        )

        if filters_in != filters_out:
            self.res = nn.Sequential(
                conv(filters_in, filters_out, 1),
                activation(filters_out),
                norm(filters_out)
            )
        else:
            self.res = None

    def forward(self, inputs):
        res = inputs if self.res is None else self.res(inputs)
        res = self.first(inputs) + res
        return res


class Autoencoder(BaseModel):
    """
    Main autoencoder class. This class can actually be parameterised on init
    to have different "main blocks", normalisation layers and activation
    functions.
    """
    def __init__(
            self,
            conv_filters,
            n_inputs=1,
            kernel=3,
            norm=None,
            activation=None,
            block=None,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        """
        Constructor of the class. It's heavily parameterisable to allow for
        different autoencoder setups (residual blocks, double convolutions,
        different normalisation and activations).
        :param conv_filters: Filters for both the encoder and decoder. The
         decoder mirrors the filters of the encoder.
        :param n_inputs: Number of input channels.
        :param kernel: Kernel width for the main block.
        :param norm: Normalisation block (it has to be a pointer to a valid
         normalisation Module).
        :param activation: Activation block (it has to be a pointer to a valid
         activation Module).
        :param block: Main block. It has to be a pointer to a valid block from
         this python file (otherwise it will fail when trying to create a
         partial of it).
        :param device: Device where the model is stored (default is the first
         cuda device).
        """
        super().__init__()
        # Init
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if block is None:
            block = Conv2dBlock
        block_partial = partial(
            block, kernel=kernel, norm=norm, activation=activation
        )
        self.device = device
        self.filters = conv_filters

        conv_in, conv_out, deconv_in, deconv_out = compute_filters(
            n_inputs, conv_filters
        )

        # Down path
        # We'll use the partial and fill it with the channels for input and
        # output for each level.
        self.down = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                conv_in, conv_out
            )
        ])

        # Bottleneck
        self.u = block_partial(conv_filters[-2], conv_filters[-1])

        # Up path
        # Now we'll do the same we did on the down path, but mirrored. We also
        # need to account for the skip connections, that's why we sum the
        # channels for both outputs. That basically means that we are
        # concatenating with the skip connection, and not suming.
        self.up = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                deconv_in, deconv_out
            )
        ])

    def encode(self, input_s):
        # We need to keep track of the convolutional outputs, for the skip
        # connections.
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = c(input_s)
            down_inputs.append(input_s)
            input_s = F.max_pool2d(input_s, 2)

        self.u.to(self.device)
        bottleneck = self.u(input_s)

        return down_inputs, bottleneck

    def decode(self, input_s, skip_inputs):
        for d, i in zip(self.up, skip_inputs[::-1]):
            d.to(self.device)
            input_s = F.interpolate(input_s, size=i.size()[2:])
            input_s = d(torch.cat((input_s, i), dim=1))

        return input_s

    def forward(self, input_s, keepfeat=False):
        down_inputs, input_s = self.encode(input_s)
        input_s = self.decode(input_s, down_inputs)

        output = (input_s, down_inputs) if keepfeat else input_s

        return output


class AttentionAutoencoder(Autoencoder):
    """
    Main autoencoder class. This class can actually be parameterised on init
    to have different "main blocks", normalisation layers and activation
    functions. This autoencoder expands on the regular AutoEncoder class
    by introducing attention gates.
    """
    def __init__(
            self,
            conv_filters,
            n_inputs=1,
            kernel=3,
            norm=None,
            activation=None,
            block=None,
            attention=32,
            att_regions=4,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        """
        Constructor of the class. It's heavily parameterisable to allow for
        different autoencoder setups (residual blocks, double convolutions,
        different normalisation and activations).
        :param conv_filters: Filters for both the encoder and decoder. The
         decoder mirrors the filters of the encoder.
        :param n_inputs: Number of input channels.
        :param kernel: Kernel width for the main block.
        :param norm: Normalisation block (it has to be a pointer to a valid
         normalisation Module).
        :param activation: Activation block (it has to be a pointer to a valid
         activation Module).
        :param block: Main block. It has to be a pointer to a valid block from
         this python file (otherwise it will fail when trying to create a
         partial of it).
        :param attention: Number of features channels for the attention gate
         blocks.
        :param att_regions: Number of regions used on the attention gate block.
         While the original paper used only one region, they propose an
         extension to include multiple attention maps/gates.
        :param device: Device where the model is stored (default is the first
         cuda device).
        """
        super().__init__(
            conv_filters=conv_filters, n_inputs=n_inputs, kernel=kernel,
            norm=norm, activation=activation, block=block, device=device
        )
        # Init
        conv_in, conv_out, deconv_in, deconv_out = compute_filters(
            n_inputs, conv_filters
        )

        # Attention gates. This is the only part that should differ from the
        # regular autoencoder. The super will take care of initialising
        # everything else.
        self.ag = nn.ModuleList([
            AttentionGate2D(f_in, f_g, attention, regions=att_regions)
            for f_in, f_g in zip(conv_out[::-1], conv_filters[::-1])
        ])

    def decode(self, input_s, skip_inputs):
        # This is the only other difference. The encoding process is exactly
        # the same.
        attention_gates = []
        for d, ag, i in zip(self.up, self.ag, skip_inputs[::-1]):
            d.to(self.device)
            output_ag, attention = ag(i, input_s, True)
            attention_gates.append(attention)
            input_s = F.interpolate(input_s, size=i.size()[2:])

            input_s = d(torch.cat((input_s, output_ag), dim=1))

        return input_s, attention_gates

    def forward(self, input_s, keepfeat=False):
        # Since we change the encoding process, we also need to change the
        # forward function.
        down_inputs, input_s = self.encode(input_s)

        input_s, gates = self.decode(input_s, down_inputs)

        output = (input_s, gates) if keepfeat else input_s

        return output


class TransAutoencoder(BaseModel):
    """
    Main autoencoder class. This class can actually be parameterised on init
    to have different "main blocks", normalisation layers and activation
    functions. In contrast with the AttentionAutoencoder, this new class
    is a redesign of the whole AutoEncoder concept. The encoding path combines
    both convolutional blocks and multi-headed self-attention, while the
    decoding path is entirely comprised of attention gates. For that reason,
    we redesigned the whole class instead of extending the main AutoEncoder
    model.
    """
    def __init__(
            self,
            conv_filters,
            n_inputs=8,
            kernel=3,
            norm=None,
            activation=None,
            block=None,
            conv_depth=3,
            heads=8,
            downsampling=2,
            att_regions=4,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
    ):
        """
        Constructor of the class. It's heavily parameterisable to allow for
        different autoencoder setups (residual blocks, double convolutions,
        different normalisation and activations).
        :param conv_filters: Filters for both the encoder and decoder. The
         decoder mirrors the filters of the encoder.
        :param n_inputs: Number of input channels.
        :param kernel: Kernel width for the main block.
        :param norm: Normalisation block (it has to be a pointer to a valid
         normalisation Module).
        :param activation: Activation block (it has to be a pointer to a valid
         activation Module).
        :param block: Main block. It has to be a pointer to a valid block from
         this python file (otherwise it will fail when trying to create a
         partial of it).
        :param heads: Number of self-attention blocks used on the multi-headed
         attention block.
        :param conv_depth: Number of convolutional blocks followed by pooling.
         Using 3D volumes creates a challenge in terms of GPU memory. To both
         work around that limitation and to also include local information
         (self-attention is technically a non-local operation) we start the
         encoding path with convolutional blocks and then apply self-attention
         on a lower resolution. That, combined with the downsampling rate, allows
         us to take the best of both worlds within the memory restrictions.
        :param downsampling: Dowsampling rate for the self-attention blocks.
         Instead of using the whole feature map, we resample it to save memory.
        :param att_regions: Number of regions used on the attention gate block.
         While the original paper used only one region, they propose an
         extension to include multiple attention maps/gates.
        :param device: Device where the model is stored (default is the first
         cuda device).
        """
        super().__init__()
        # Init
        self.downsampling = downsampling
        if norm is None:
            norm = nn.InstanceNorm3d
        if activation is None:
            activation = nn.ReLU
        if block is None:
            block = Conv2dBlock
        block_partial = partial(
            block, kernel=kernel, norm=norm, activation=activation
        )
        self.conv_depth = conv_depth
        self.device = device
        self.filters = conv_filters
        # The self-attention depth is dependent on the convolutional depth.
        # First we use a normal convolutional encoder, and after a few
        # downsampling steps we start using self-attention. It reduces memory
        # and separates the local part (convolutional) from the "semantic"
        # part defined by long-term spatial connections.
        # This will define the inputs and outputs of each block as follows:
        sa_depth = len(self.filters) - conv_depth - 1
        sa_filters = self.filters[conv_depth:]
        sa_out = self.filters[conv_depth - 1]

        conv_in = [n_inputs] + self.filters[:conv_depth - 1]
        conv_out = self.filters[:conv_depth]

        ag_in = [sa_out] * sa_depth + conv_out[::-1]
        ag_out = [fi * att_regions for fi in ag_in]
        g_in = [sa_out] + ag_out[:-1]

        # Down path
        # We'll use the partial and fill it with the channels for input and
        # output for each level.
        self.sa_seq = self.ag_seq = nn.ModuleList([
            nn.Sequential(
                norm(sa_out)
            )
            for _ in sa_filters
        ])
        self.down = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                conv_in, conv_out
            )
        ])
        self.sa = nn.ModuleList([
            DownsampledMultiheadAttention2D(
                sa_out, conv_att, heads, downsampling,
            )
            for conv_att in sa_filters
        ])
        self.ag = nn.ModuleList([
            AttentionGate2D(x_feat, g_feat, conv_att, regions=att_regions)
            for x_feat, g_feat, conv_att in zip(
                ag_in, g_in, conv_filters[::-1]
            )
        ])
        self.ag_seq = nn.ModuleList([
            nn.Sequential(
                activation(),
                norm(att_in)
            )
            for att_in in ag_out
        ])

    def encode(self, x):
        # We need to keep track of the convolutional outputs, for the skip
        # connections.

        # This is the initial convolutional path. Since the self-attention
        # gates have "learned pooling" operations performed by Conv3D blocks
        # with the same stride and kernel size, we don't need to apply pooling
        # to the output of the last convolutional.
        down_inputs = []
        for i, c_i in enumerate(self.down):
            c_i.to(self.device)
            x = c_i(x)
            if i < (self.conv_depth - 1):
                down_inputs.append(x)
                x = F.max_pool2d(x, self.downsampling)
            # if self.training:
            #     print(
            #         'Conv3D {:}: {:6.3f} ± {:6.3f} [{:6.3f}, {:6.3f}]'.format(
            #             i,
            #             torch.mean(x.detach().cpu()).numpy().tolist(),
            #             torch.std(x.detach().cpu()).numpy().tolist(),
            #             torch.min(x.detach().cpu()).numpy().tolist(),
            #             torch.max(x.detach().cpu()).numpy().tolist(),
            #         )
            #     )

        # Self-attention path. Pooling is embedded on the self-attention
        # block.
        for i, (sa_i, end_i) in enumerate(zip(self.sa, self.sa_seq)):
            down_inputs.append(x)
            sa_i.to(self.device)
            res = F.max_pool2d(x, self.downsampling)
            x = sa_i(x)
            # if self.training:
            #     print(
            #         'SA {:}: {:6.3f} ± {:6.3f} [{:6.3f}, {:6.3f}]'.format(
            #             i,
            #             torch.mean(x.detach().cpu()).numpy().tolist(),
            #             torch.std(x.detach().cpu()).numpy().tolist(),
            #             torch.min(x.detach().cpu()).numpy().tolist(),
            #             torch.max(x.detach().cpu()).numpy().tolist(),
            #         )
            #     )
            x = end_i(x + res)

        return down_inputs, x

    def decode(self, x, skip_inputs):
        attention_gates = []
        for i, (ag_i, skip_i, end_i) in enumerate(
                zip(self.ag, skip_inputs[::-1], self.ag_seq)
        ):
            ag_i.to(self.device)
            x, attention = ag_i(skip_i, x, True)
            # if self.training:
            #     print(
            #         'AG {:}: {:6.3f} ± {:6.3f} [{:6.3f}, {:6.3f}]'.format(
            #             i,
            #             torch.mean(x.detach().cpu()).numpy().tolist(),
            #             torch.std(x.detach().cpu()).numpy().tolist(),
            #             torch.min(x.detach().cpu()).numpy().tolist(),
            #             torch.max(x.detach().cpu()).numpy().tolist(),
            #         )
            #     )
            x = end_i(x)
            attention_gates.append(attention)

        return x, attention_gates

    def forward(self, x, keepfeat=False):
        down_inputs, x = self.encode(x)
        x,  gates = self.decode(x, down_inputs)

        output = (x, gates) if keepfeat else x

        return output
