# "`-''-/").___..--''"`-._
#  (`6_ 6  )   `-.  (     ).`-.__.`)   WE ARE ...
#  (_Y_.)'  ._   )  `._ `. ``-..-'    PENN STATE!
#    _ ..`--'_..-_/  /--'_.' ,'
#  (il),-''  (li),'  ((!.-'
#
# Author: Weiming Hu <weiming@psu.edu>
#         Geoinformatics and Earth Observation Laboratory (http://geolab.psu.edu)
#         Department of Geography and Institute for CyberScience
#         The Pennsylvania State University
#
# This file includes definitions of ConvLSTM.
#
# This code is heavily referenced from
#
# 1. https://github.com/czifan/ConvLSTM.pytorch/blob/master/networks/ConvLSTM.py
# 2. https://github.com/Jimexist/conv_lstm_pytorch/blob/master/conv_lstm/conv_lstm.py
#

import torch

import torch.nn as nn

from collections import OrderedDict


class MaxPool2dSequence(nn.MaxPool2d):
    def forward(self, x):
        # The input x is of shapre [B, S, C, H, W]
        b, s, c, h, w = x.shape

        # Flatten to [B, C * S, H, W]
        x = x.flatten(1, 2)

        x = nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                                     self.dilation, self.ceil_mode, self.return_indices)

        # Reconstruct [B, S, C, H, W]
        x = x.reshape(b, s, c, x.shape[2], x.shape[3])

        return x


class Dropout2dSequence(nn.Dropout2d):
    def forward(self, x):
        # The input x is of shape [B, S, C, H, W]
        b, s, c, h, w = x.shape

        # Change to [B, C, S * H, W]
        x = x.transpose(1, 2).flatten(2, 3)

        x = nn.functional.dropout2d(x, self.p, self.training, self.inplace)

        # Change the shape back to [B, S, C, H, W]
        x = x.reshape(b, c, s, h, w).transpose(1, 2)

        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, input_features, hidden_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = hidden_features
        self.conv = self._make_layer(input_features + hidden_features, hidden_features * 4,
                                     kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :return: (B, S, C, H, W)
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)


class ConvLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers,
                 layer_types='conv_lstm', conv_kernel_size=3, pool_kernel_size=2, dropout=0.0, batch_first=True):
        super().__init__()

        conv_kernel_size = self._extend_for_multilayer(conv_kernel_size, num_layers)
        pool_kernel_size = self._extend_for_multilayer(pool_kernel_size, num_layers)
        hidden_features = self._extend_for_multilayer(hidden_features, num_layers)
        layer_types = self._extend_for_multilayer(layer_types, num_layers)
        dropout = self._extend_for_multilayer(dropout, num_layers)

        assert len(conv_kernel_size) == len(hidden_features) == len(layer_types) == \
               len(pool_kernel_size) == len(dropout) == num_layers, \
            'Length of (conv/pool kernel, dropout, layer type, hidden features) do not match the number of layers'

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.layer_types = layer_types

        self.layers = OrderedDict()
        for i in range(self.num_layers):
            sub_layers = OrderedDict()

            cur_input_dim = self.input_features if i == 0 else self.hidden_features[i-1]

            if self.layer_types[i] == 'conv_lstm':
                sub_layers['ConvLSTMCell'] = ConvLSTMCell(input_features=cur_input_dim,
                                                          hidden_features=self.hidden_features[i],
                                                          kernel_size=self.conv_kernel_size[i])

                sub_layers['Dropout2d'] = Dropout2dSequence(p=self.dropout[i], inplace=True)
                sub_layers['MaxPool2d'] = MaxPool2dSequence(kernel_size=self.pool_kernel_size[i])

            elif self.layer_types[i] == 'conv':
                sub_layers['Conv2d'] = nn.Conv2d(in_channels=cur_input_dim,
                                                 out_channels=self.hidden_features[i],
                                                 kernel_size=self.conv_kernel_size[i],
                                                 padding=1, stride=1)

                sub_layers['BatchNorm2d'] = nn.BatchNorm2d(self.hidden_features[i])
                sub_layers['LeakyReLU'] = nn.LeakyReLU(inplace=True)

                sub_layers['Dropout2d'] = nn.Dropout2d(p=self.dropout[i], inplace=True)
                sub_layers['MaxPool2d'] = nn.MaxPool2d(kernel_size=self.pool_kernel_size[i])

            else:
                raise Exception('Unknown layer type: {}'.format(self.layer_types[i]))

            self.layers['{}{}'.format(self.layer_types[i], i)] = nn.Sequential(sub_layers)

        self.layers = nn.Sequential(self.layers)

    def forward(self, x):
        """
        :param inputs: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            inputs = x.permute(1, 0, 2, 3, 4)

        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == 'conv':
                B, T, C, H, W = x.shape
                x = x.reshape(B * T, C, H, W)

            x = layer(x)

            if layer_type == 'conv':
                x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])

        return x

    @staticmethod
    def _extend_for_multilayer(param, num_layers: int):
        if isinstance(param, int) or isinstance(param, float) or isinstance(param, str):
            return (param,) * num_layers
        elif isinstance(param, list):
            return tuple(param)
        else:
            return param
