import itertools
import time
import numpy as np
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F
from base import BaseModel
from utils import to_torch_var, time_to_string
from criteria import cross_entropy


class Autoencoder2D(BaseModel):
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            kernel_size=3
    ):
        super().__init__()
        # Init
        self.device = device
        # Down path
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    f_in, f_out, kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(f_out)
            ) for f_in, f_out in zip(
                [n_inputs] + conv_filters[:-2], conv_filters[:-1]
            )
        ])

        self.u = nn.Sequential(
            nn.Conv2d(
                conv_filters[-2], conv_filters[-1], kernel_size,
                padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[-1])
        )

        # Up path
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    f_in, f_out, kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.BatchNorm2d(f_out)
            ) for f_in, f_out in zip(
                deconv_in, down_out
            )
        ])

    def forward(self, input_s):
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = c(input_s)
            down_inputs.append(input_s)
            input_s = F.max_pool2d(input_s, 2)

        self.u.to(self.device)
        input_s = self.u(input_s)

        for d, i in zip(self.up, down_inputs[::-1]):
            d.to(self.device)
            input_s = d(
                torch.cat(
                    (F.interpolate(input_s, size=i.size()[2:]), i), dim=1
                )
            )

        return input_s


class Unet2D(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_inputs=3, n_outputs=1
    ):
        super(Unet2D, self).__init__()
        # Init values
        if conv_filters is None:
            conv_filters = [32, 64, 128, 256,512]
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        # I plan to change that to use the AutoEncoder framework in
        # data_manipulation.
        self.autoencoder = Autoencoder2D(
            conv_filters, device, n_inputs
        )

        # Final segmentation block.
        if n_outputs == 1:
            n_outputs = 2
        self.n_outputs = n_outputs
        self.seg = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], n_outputs, 1)
        )
        self.seg.to(device)

        # <Loss function setup>
        self.train_functions = [
            # Focal loss for the main branch.
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: cross_entropy(p[0], t, p[1])
            }
        ]
        self.val_functions = [
            # Focal loss for validation.
            # The weight is 0 because I feel like DSC is more important.
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: cross_entropy(p[0], t, p[1])
            }
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        # self.optimizer_alg = torch.optim.Adadelta(model_params)
        self.optimizer_alg = torch.optim.Adam(model_params)
        # self.optimizer_alg = torch.optim.SGD(model_params, lr=1e-1)
        # self.autoencoder.dropout = 0.99
        # self.dropout = 0.99
        # self.ann_rate = 1e-2

    def forward(self, input_ae, mask=None):
        input_s = self.autoencoder(input_ae)

        # Since we are dealing with a binary problem, there is no need to use
        # softmax.
        seg = torch.softmax(self.seg(input_s), dim=1)

        if mask is None:
            return seg
        else:
            return seg, mask

    def test(
            self, data, patch_size=256, verbose=True
    ):
        # Init
        self.eval()
        seg = list()

        # Init
        t_in = time.time()

        for i, im in enumerate(data):

            # Case init
            t_case_in = time.time()

            # This branch is only used when images are too big. In this case
            # they are split in patches and each patch is trained separately.
            # Currently, the image is partitioned in blocks with no overlap,
            # however, it might be a good idea to sample all possible patches,
            # test them, and average the results. I know both approaches
            # produce unwanted artifacts, so I don't know.
            if patch_size is not None:

                # Initial results. Filled to 0.
                seg_i = np.zeros((self.n_outputs,) + im.shape[1:])
                counts_i = np.zeros((self.n_outputs,) + im.shape[1:])

                limits = tuple(
                    list(
                        range(0, lim - patch_size, patch_size // 4)
                    ) + [lim - patch_size]
                    for lim in im.shape[1:]
                )
                limits_product = list(itertools.product(*limits))

                n_patches = len(limits_product)

                # The following code is just a normal test loop with all the
                # previously computed patches.
                for pi, (xi, xj) in enumerate(limits_product):
                    # Here we just take the current patch defined by its slice
                    # in the x and y axes. Then we convert it into a torch
                    # tensor for testing.
                    xslice = slice(xi, xi + patch_size)
                    yslice = slice(xj, xj + patch_size)
                    data_tensor = to_torch_var(
                        np.expand_dims(im[slice(None), xslice, yslice], axis=0)
                    )

                    # Testing itself.
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        seg_pi = self(data_tensor)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Then we just fill the results image.
                    seg_i[slice(None), xslice, yslice] += np.squeeze(seg_pi.cpu().numpy())
                    counts_i[slice(None), xslice, yslice] += 1

                    # Printing
                    init_c = '\033[0m' if self.training else '\033[38;5;238m'
                    whites = ' '.join([''] * 12)
                    percent = 20 * (pi + 1) // n_patches
                    progress_s = ''.join(['-'] * percent)
                    remainder_s = ''.join([' '] * (20 - percent))

                    t_out = time.time() - t_in
                    t_case_out = time.time() - t_case_in
                    time_s = time_to_string(t_out)

                    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
                    eta_s = time_to_string(t_eta)
                    batch_s = '{:}Case {:03}/{:03} ({:03d}/{:03d})' \
                              ' [{:}>{:}] {:} ETA: {:}'.format(
                        init_c + whites, i + 1, len(data), pi + 1, n_patches,
                        progress_s, remainder_s, time_s, eta_s + '\033[0m'
                    )
                    print('\033[K', end='', flush=True)
                    print(batch_s, end='\r', flush=True)

                seg_i[counts_i > 0] = seg_i[counts_i > 0] / counts_i[counts_i > 0]

            else:
                # If we use the whole image the process is way simpler.
                # We only need to convert the data into a torch tensor,
                # test it and return the results.
                data_tensor = to_torch_var(np.expand_dims(im, axis=0))

                # Testing
                with torch.no_grad():
                    torch.cuda.synchronize(self.device)
                    seg_pi = self(data_tensor)
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()

                # Image squeezing.
                # The images have a batch number at the beginning. Since each
                # batch is just an image, that batch number is useless.
                seg_i = np.squeeze(seg_pi.cpu().numpy())

                # Printing
                init_c = '\033[0m' if self.training else '\033[38;5;238m'
                whites = ' '.join([''] * 12)
                percent = 20 * (i + 1)
                progress_s = ''.join(['-'] * percent)
                remainder_s = ''.join([' '] * (20 - percent))

                t_out = time.time() - t_in
                t_case_out = time.time() - t_case_in
                time_s = time_to_string(t_out)

                t_eta = (t_case_out / (i + 1)) * (len(data) - i + 1)
                eta_s = time_to_string(t_eta)
                batch_s = '{:}Case {:03}/{:03} [{:}>{:}] {:} ETA: {:}'.format(
                    init_c + whites, i + 1, len(data),
                    progress_s, remainder_s, time_s, eta_s + '\033[0m'
                )
                print('\033[K', end='', flush=True)
                print(batch_s, end='\r', flush=True)

            #if verbose:
            #    print(
            #        '\033[K%sSegmentation finished' % ' '.join([''] * 12)
            #    )

            seg.append(seg_i)

        return seg
