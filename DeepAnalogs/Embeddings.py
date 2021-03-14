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


class Dropout1d(nn.Dropout2d):
    """
    To use the spatial dropout algorithm for 1-D convolution, I need to
    manually add a fourth dimension and later on remove this dimension.
    """
    def forward(self, x):
        # The input x is of shape [samples, channels, layers]

        # Add a dimensions so that x is [samples, channels, layers, 1]
        x = x.unsqueeze(3)

        # Spatial dropout
        x = nn.functional.dropout2d(x, self.p, self.training, self.inplace)

        # Change the shape back to [samples, channels, layers]
        x = x.squeeze(3)

        return x


class EmbeddingLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, hidden_layers, output_features,
                 scaler=None, dropout=0.0, subset_variables_index=None):
        
        super().__init__()
        
        self.scaler = scaler
        self.embedding_type = 1
        self.subset_variables_index = None if subset_variables_index is None else torch.tensor(subset_variables_index, dtype=torch.long)

        # LSTM layer in stateless mode because I'm not saving the hidden states across batches.
        self.lstm = nn.LSTM(input_size=input_features,
                            hidden_size=hidden_features,
                            num_layers=hidden_layers,
                            batch_first=True,
                            dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_features,
                            out_features=output_features)

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
        output = self.fc(output)
        return output


class EmbeddingConvLSTM(nn.Module):
    def __init__(self, additional_lstm_channels, conv_in_channels, conv_channels, conv_kernels, pool_kernels,
                 lstm_features, lstm_layers, output_channels, preprocessor,
                 conv_dropout=0.0, lstm_dropout=0.0, scaler=None, subset_variables_index=None):
        
        super().__init__()
        
        self.scaler = scaler
        self.embedding_type = 2
        self.preprocessor = preprocessor
        self.subset_variables_index = None if subset_variables_index is None else torch.tensor(subset_variables_index, dtype=torch.long)

        ######################
        # Convolution layers #
        ######################
        
        assert len(conv_channels) == len(conv_kernels), "Assert incorrect channels and kernels!"
        assert len(pool_kernels) == len(conv_kernels), "Assert incorrect conv and pool kernels!"
        conv_modules = []
        
        for index in range(len(conv_channels)):
            
            # Add a convolution layer
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
            #
            conv_modules.append(nn.Conv1d(
                    in_channels=conv_in_channels if index == 0 else conv_channels[index-1],
                    out_channels=conv_channels[index],
                    kernel_size=conv_kernels[index]))

            # Batch norm (Removed because the network overfits too fast with this latyer)
            # conv_modules.append(nn.BatchNorm1d(conv_channels[index]))
            
            # Non linear activation
            conv_modules.append(nn.PReLU())
            
            # Spatial dropout
            conv_modules.append(Dropout1d(p=conv_dropout))
        
            # Add a pooling layer
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d
            #
            conv_modules.append(torch.nn.MaxPool1d(kernel_size=pool_kernels[index]))

        self.conv = nn.Sequential(*conv_modules)
        
        ###############
        # LSTM layers #
        ###############
        # 
        # LSTM layer in stateless mode because I'm not saving the hidden states across batches.
        #
        self.lstm = nn.LSTM(input_size=conv_channels[-1] + additional_lstm_channels,
                            hidden_size=lstm_features,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout,
                            batch_first=True)

        #################
        # Linear layers #
        #################
        
        self.fc = nn.Linear(in_features=lstm_features,
                            out_features=output_channels)

    def forward(self, x, add_cpp_routines=torch.full((1,), False, dtype=torch.bool)):
        # Input dimensions are [samples, features, 1 station, lead times]
        
        # Sanity check
        assert len(x.shape) == 4, '4-dimensional input is expected [samples, features, 1 station, lead times]'
        assert x.shape[2] == 1, 'This network works for only one station!'

        if add_cpp_routines.item():

            # Select variables
            if self.subset_variables_index is not None:
                x = torch.index_select(x, 1, self.subset_variables_index)

            # The input x is of shape [N samples, features (parameters), 1 station, lead_times],
            # The scaler takes shapes of [features, *, *, *].
            # Therefore, I need to fix the dimensions.
            #
            x = torch.transpose(x, 0, 1)
            x = self.scaler.transform(x)
            x = torch.transpose(x, 0, 1)
        
        # Get dimensions
        num_samples = x.size(0)
        num_features = x.size(1)
        num_lead_times = x.size(3)
        
        # Remove single length dimension
        x = x.squeeze(2)
        
        # Split variables into vertical and general
        x = torch.transpose(x, 1, 2)
        x = x.reshape(-1, num_features)
        x_split = self.preprocessor.do(x)
        
        # Fix the dimensions of general variables
        # At this point, the shape is [samples, sequences, features]
        #
        x_general = x_split['general'].reshape(num_samples, num_lead_times, -1)
        
        x_vertical = self.conv(x_split['vertical'])
        x_vertical = x_vertical.squeeze(2)
        
        # At this point, the shape is [samples, sequences, features]
        x_vertical = x_vertical.reshape(num_samples, num_lead_times, -1)
        
        # Combine both variables to be a spatial-temporal tensor
        x_st = torch.cat([x_general, x_vertical], 2)
        
        # LSTM
        output, lstm_state = self.lstm(x_st)
        
        # Linear
        output = output.select(1, -1)
        output = self.fc(output)
        return output
