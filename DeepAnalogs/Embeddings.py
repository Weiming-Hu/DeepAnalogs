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
# This file includes definitions of various embedding networks.
#

import torch
from torch import nn
from DeepAnalogs.ConvLSTM import ConvLSTM


class EmbeddingLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, hidden_layers, output_features,
                 scaler=None, dropout=0.0, subset_variables_index=None, fc_last=False):
        
        super().__init__()
        
        self.scaler = scaler
        self.embedding_type = 1
        self.fc_last = fc_last

        if subset_variables_index is None:
            self.subset_variables_index = None
        else:
            self.subset_variables_index = torch.tensor(subset_variables_index, dtype=torch.long)

        # LSTM layer in stateless mode because I'm not saving the hidden states across batches.
        self.lstm = nn.LSTM(input_size=input_features,
                            hidden_size=hidden_features,
                            num_layers=hidden_layers,
                            batch_first=True,
                            dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_features,
                            out_features=output_features)

        if not self.fc_last:
            assert hidden_features == output_features, \
                'Hidden and output features should be the same when no fully connected layers at the end'

    def forward(self, x, add_cpp_routines=torch.full((1,), False, dtype=torch.bool)):
        # Input x dimensions are [samples, features, 1 station, lead times]
        # Input lead_time_coding dimensions are [samples, 1]
        #
        
        # Sanity check
        assert len(x.shape) == 4, '4-dimensional input is expected [samples, features, 1 station, lead times]'
        assert x.shape[2] == 1, 'This network works for only one station!'

        if add_cpp_routines.item():

            # Select variables
            if not self.subset_variables_index is None:
                x = torch.index_select(x, 1, self.subset_variables_index)

            # The input x is of shape [N samples, features (parameters), 1 station, lead_times],
            # The scaler takes shapes of [features, *, *, *].
            # Therefore, I need to fix the dimensions.
            #
            x = torch.transpose(x, 0, 1)
            x = self.scaler.transform(x)
            x = torch.transpose(x, 0, 1)

        # Fix dimensions
        x = x.squeeze(2)
        
        # Transpose the dimensions to be [batch (samples), sequences (lead times), features (parameters)]
        x = torch.transpose(x, 1, 2)

        # LSTM forward pass
        output, lstm_state = self.lstm(x)
        output = output.select(1, -1)

        if self.fc_last:
            output = self.fc(output)

        return output


class EmbeddingConvLSTM(nn.Module):
    def __init__(self, input_width, input_height, input_features, hidden_features,
                 hidden_layers, hidden_layer_types, conv_kernel_size, pool_kernel_size, output_features,
                 dropout=0.0, scaler=None, subset_variables_index=None, fc_last=False):
        super().__init__()

        self.scaler = scaler
        self.embedding_type = 2
        self.input_width = input_width
        self.input_height = input_height
        self.fc_last = fc_last

        if subset_variables_index is None:
            self.subset_variables_index = None
        else:
            self.subset_variables_index = torch.tensor(subset_variables_index, dtype=torch.long)

        self.conv_lstm = ConvLSTM(input_features=input_features,
                                  hidden_features=hidden_features,
                                  num_layers=hidden_layers,
                                  layer_types=hidden_layer_types,
                                  conv_kernel_size=conv_kernel_size,
                                  pool_kernel_size=pool_kernel_size,
                                  dropout=dropout)

        # Estimate the number of grids after convolution
        rand_input = torch.rand(1, 2, input_features, self.input_height, self.input_width)
        shape_after_conv = self.conv_lstm(rand_input).shape[-2:]

        n_grids = shape_after_conv[0] * shape_after_conv[1]
        assert n_grids > 0, 'ConvLSTM produces 0 length output! Check your network hyperparameters!'

        # Use all grids left after convolution as input variables
        hidden_features = hidden_features[-1] if isinstance(hidden_features, list) else hidden_features
        self.fc = nn.Linear(in_features=hidden_features * n_grids, out_features=output_features)

        if not self.fc_last:
            assert hidden_features == output_features, \
                'Hidden and output features should be the same when no fully connected layers at the end'

    def forward(self, x, add_cpp_routines=torch.full((1,), False, dtype=torch.bool)):
        # Input x dimensions are [samples, features, height, width, lead times]
        assert x.shape[2] == self.input_height, "Expect height of {}, got {}".format(self.input_height)
        assert x.shape[3] == self.input_width, "Expect width of {}, got {}".format(self.input_width)

        # Sanity check
        assert len(x.shape) == 5, '5-dimensional input is expected [samples, features, height, width, lead times]'

        if add_cpp_routines.item():

            if self.subset_variables_index is not None:
                x = torch.index_select(x, 1, self.subset_variables_index)

            # The scaler takes shapes of [features, *, *, *].
            # Therefore, I need to fix the dimensions.
            # 
            B, C, H, W, S = x.shape
            x = x.transpose(0, 1).flatten(2, 3)
            x = self.scaler.transform(x)
            x = x.reshape(C, B, H, W, S).transpose(0, 1)

        # Fix dimensions to be [B, T, C, H, W] for [samples, lead times, features, height, width]
        x = x.permute(0, 4, 1, 2, 3)

        # Forward pass
        output = self.conv_lstm(x)

        # Select the last timestamp and flatten all grids left in the spatial domain to be 1-dimensional
        output = output.select(1, -1).flatten(1, 3)

        if self.fc_last:
            output = self.fc(output)

        # Output dimensions [samples, latent features]
        return output


class EmbeddingNaiveSpatialMask(nn.Module):
    def __init__(self, input_width, input_height, scaler=None, subset_variables_index=None):
        super().__init__()

        self.scaler = scaler
        self.embedding_type = 2
        self.input_width = input_width
        self.input_height = input_height

        if subset_variables_index is None:
            self.subset_variables_index = None
        else:
            self.subset_variables_index = torch.tensor(subset_variables_index, dtype=torch.long)

    def forward(self, x, add_cpp_routines=torch.full((1,), False, dtype=torch.bool)):
        # Input x dimensions are [samples, features, height, width, lead times]
        assert x.shape[2] == self.input_height, "Expect height of {}, got {}".format(self.input_height)
        assert x.shape[3] == self.input_width, "Expect width of {}, got {}".format(self.input_width)

        # Sanity check
        assert len(x.shape) == 5, '5-dimensional input is expected [samples, features, height, width, lead times]'

        if add_cpp_routines.item():

            if self.subset_variables_index is not None:
                x = torch.index_select(x, 1, self.subset_variables_index)

            # The scaler takes shapes of [features, *, *, *].
            # Therefore, I need to fix the dimensions.
            #
            B, C, H, W, S = x.shape
            x = x.transpose(0, 1).flatten(2, 3)
            x = self.scaler.transform(x)
            x = x.reshape(C, B, H, W, S).transpose(0, 1)

        # Calculate average across lead times
        x = x.mean(axis=4)

        # Shovel everything else as features
        x = x.reshape(x.shape[0], -1)

        return x
