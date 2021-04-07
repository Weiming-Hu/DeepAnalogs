#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# This script trains a embedding model.
#

import os
import time
import json
import torch
import pickle
import random
import configargparse

import numpy as np

from tqdm import tqdm
from pprint import pprint
from datetime import datetime, timezone

from DeepAnalogs import __version__
from DeepAnalogs.AnEnDict import AnEnDict
from DeepAnalogs.utils import sort_distance_mc, summary_pytorch
from DeepAnalogs.Embeddings import EmbeddingLSTM, EmbeddingConvLSTM
from DeepAnalogs.AnEnDataset import AnEnDatasetWithTimeWindow, AnEnDatasetOneToMany, AnEnDatasetSpatial

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Global functions
def backup(filename, d):
    assert isinstance(d, dict), 'Only supports backing up a dictionary!'
    
    print('\n**************************************************************')
    print('Saving an intermediate file {}'.format(filename))

    with open(filename, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    print('The following objects have been saved: {}'.format(', '.join(d.keys())))
    print('If you specify this file next time, the program will start from here.')
    print('**************************************************************')


def restore(filename):
    print('\n**************************************************************')
    print('Reading from an intermediate file {}'.format(filename))

    with open(filename, "rb") as f:
        d = pickle.load(f)
        
    assert isinstance(d, dict), 'Only supports restoring a dictionary!'

    print('The following objects have been updated: {}'.format(', '.join(d.keys())))
    print('**************************************************************')
    
    return d


def main():
    start_time = datetime.now()

    ###################
    # Parse arguments #
    ###################
    parser = configargparse.ArgParser(description='Train an embedding network v {}'.format(__version__))

    required_general = parser.add_argument_group('Required arguments for all trainings')
    required_general.add_argument('--out', help='Output folder', required=True)
    required_general.add_argument('--forecast', help='An NetCDF file for forecasts', required=True)
    required_general.add_argument('--observation', help='An NetCDF file for observations', required=True)
    required_general.add_argument('--anchor-start', help='Start date for anchors', required=True, dest='anchor_start')
    required_general.add_argument('--anchor-end', help='End date for anchors', required=True, dest='anchor_end')
    required_general.add_argument('--search-start', help='Start date for search', required=True, dest='search_start')
    required_general.add_argument('--search-end', help='End date for search', required=True, dest='search_end')
    required_general.add_argument('--split', required=True, dest='split',
                                  help='Date to split train/test. This date will be included in testing.')
    required_general.add_argument('--analogs', help='Number of analogs to train', required=True, type=int)
    required_general.add_argument('--embeddings', help='Number of embedding features', required=True, type=int)
    required_general.add_argument('--lr', help='Learning rate', required=True, type=float)
    required_general.add_argument('--batch', help='Training Batch size', required=True, type=int)
    required_general.add_argument('--epochs', help='Number of training epochs', required=True, type=int)

    required_lstm = parser.add_argument_group(
        'Required arguments for training an LSTM model.\n' +
        'LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm')
    required_lstm.add_argument('--lstm-radius', help='The radius of lead time window',
                               required=True, type=int, dest='lstm_radius')
    required_lstm.add_argument('--lstm-hidden', help='The number of hidden features',
                               required=True, type=int, dest='lstm_hidden')
    required_lstm.add_argument('--lstm-layers', help='The number of layers',
                               required=True, type=int, dest='lstm_layers')

    optional_conv = parser.add_argument_group('Optional arguments for using a Convolutional LSTM model.')
    optional_conv.add_argument('--use-conv-lstm', help='Use a ConvLSTM embedding network', required=False,
                               action='store_true', dest='use_conv_lstm')
    optional_conv.add_argument('--conv-kernel-size', help='Kernel size(s) for Convolution operation', required=False,
                               default=3, type=int, dest='conv_kernel_size', nargs='*')
    optional_conv.add_argument('--maxpool-kernel-size', help='Kernel size(s) for MaxPool operation', required=False,
                               default=2, type=int, dest='pool_kernel_size', nargs='*')
    optional_conv.add_argument('--spatial-mask-width', help='Width of the spatial mask', required=False,
                               default=3, type=int, dest='spatial_mask_width')
    optional_conv.add_argument('--spatial-mask-height', help='Height of the spatial mask', required=False,
                               default=3, type=int, dest='spatial_mask_height')

    optional = parser.add_argument_group('More optional arguments')
    optional.add_argument('config', help='Config file', is_config_file=True)
    optional.add_argument('--dropout', help='Dropout probability during embedding training',
                          required=False, default=0.0, type=float, nargs='*', dest='dropout')
    optional.add_argument('--scaler-type', help='The scaling method to use while training the model',
                          required=False, default='MinMaxScaler', dest='scaler_type')
    optional.add_argument('--fcst-variables', help='Names or indices of forecast variables to use',
                          required=False, dest='fcst_variables', default=None, nargs='*')
    optional.add_argument('--obs-weights', help='Observation variable weights for reverse analogs',
                          required=False, dest='obs_weights', default=None, nargs='*', type=float)
    optional.add_argument('--positive-predictand-index', required=False, default=None, type=int, dest='positive_index',
                          help='The observation variable index that should always be positive')
    optional.add_argument('--triplet-sample-prob', required=False, default=1.0, type=float, dest='triplet_sample_prob',
                          help='If the entire dataset is too large, set this probability carry out random sampling.')
    optional.add_argument('--triplet-sample-method', required=False, default='fitness', dest='triplet_sample_method',
                          help='The sample method for selecting triplets')
    optional.add_argument('--datetime-format', required=False, help='Date time format',
                          dest='datetime_format', default='%Y/%m/%d %H:%M:%S')
    optional.add_argument('--fitness-num-negative', required=False, default=1, type=int, dest='fitness_num_negative',
                          help='The number of negative cases for each positive case if fitness sample method is used')
    optional.add_argument('--load-workers', required=False, dest='load_workers', default=4, type=int,
                          help='The number of data loader workers for training')
    optional.add_argument('--test-load-workers', required=False, dest='test_load_workers', default=8, type=int,
                          help='The number of data loader workers for testing')
    optional.add_argument('--test-batch', required=False, dest='test_batch', default=50000, type=int,
                          help='The batch size for testing')
    optional.add_argument('--use-cpu', required=False, action='store_true', dest='use_cpu',
                          help='Use CPU for model training. By default, GPU is used.')
    optional.add_argument('--train-margin', required=False, dest='train_margin', default=0.9, type=float,
                          help='Triplet loss margin for training')
    optional.add_argument('--dataset-margin', required=False, dest='dataset_margin', default=np.nan, type=float,
                          help='The margin used while creating the triplet dataset')
    optional.add_argument('--obs-stations-index', required=False, dest='obs_stations_index', default=None, nargs='*',
                          type=int, help='The station indices to subset after reading observations')
    optional.add_argument('--fcst-stations-index', required=False, dest='fcst_stations_index', default=None, nargs='*',
                          type=int, help='The station indices to subset after reading forecasts')
    optional.add_argument('--cpu-cores', required=False, dest='cpu_cores', default=1, type=int,
                          help='The number of CPU to use during data preprocessig')
    optional.add_argument('--intermediate-file', required=False, dest='intermediate_file', default='',
                          help='A file saved or to be saved before the model training stage with necessary variables')
    optional.add_argument('--julian-weight', required=False, dest='julian_weight', default=0.0, type=float,
                          help='The weight for Julian days when selecting reverse analogs')
    optional.add_argument('--optimizer', required=False, default='Adam', help='The optimizer to user from PyTorch')
    optional.add_argument('--use-amsgrad', required=False, action='store_true', dest='amsgrad',
                          help='For Adam and its variants, use amsgrad')
    optional.add_argument('--trans-args', dest='trans_args', required=False, default=None, type=json.loads,
                          help='A distionary for transformation [fitness selection]')
    optional.add_argument('--wdecay', help='Weight decay', required=False, type=float, default=0.0)
    optional.add_argument('--dataset-class', required=False, default='AnEnDatasetWithTimeWindow', dest='dataset_class',
                          help='One of [AnEnDatasetOneToMany, AnEnDatasetWithTimeWindow, AnEnDatasetSpatial]')
    optional.add_argument('--matching-forecast-station', required=False, default=-1, type=int, dest='matching_forecast_station',
                          help='The index of the forecast station to match the observation station [AnEnDatasetOneToMany]')


    # Parse arguments
    args = parser.parse_args()

    # Convert argument lists to numbers if they are numbers
    arg_list = getattr(args, 'fcst_variables')

    if arg_list is not None and all([arg_str.isdigit() for arg_str in arg_list]):
        setattr(args, 'fcst_variables', [int(val) for val in arg_list])

    # Expand user path
    for arg in ['out', 'forecast', 'observation', 'intermediate_file']:
        setattr(args, arg, os.path.expanduser(getattr(args, arg)))

    # Check for existence
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # Recognize date time characters to a UTC time object
    for arg in ['split', 'anchor_start', 'anchor_end', 'search_start', 'search_end']:
        datetime_str = getattr(args, arg)
        datetime_utc = datetime(*(time.strptime(datetime_str, args.datetime_format)[0:6]), tzinfo=timezone.utc)
        setattr(args, arg, datetime_utc)

    # Import a scaling method
    if args.scaler_type == 'MinMaxScaler':
        from DeepAnalogs.Scalers import MinMaxScaler as ScalerClass
    elif args.scaler_type == 'StandardScaler':
        from DeepAnalogs.Scalers import StandardScaler as ScalerClass
    else:
        raise Exception('The input scaler type {} is not supported!'.format(args.scaler_type))

    # Decide the type of network to train
    if args.use_conv_lstm:
        network_type = 'ConvLSTM'
    else:
        network_type = 'LSTM'

    # If only one dropout is specified, flatten the list to a scalar
    if len(args.dropout) == 1:
        args.dropout = args.dropout[0]

    print('Train deep network for Deep Analogs v {}'.format(__version__))
    print('Argument preview:')
    print(parser.format_values())
    print('Use the embedding network {}'.format(network_type))

    if not os.path.exists(args.intermediate_file):

        ############################
        # Generate reverse analogs #
        ############################

        # Read NetCDF files
        print('Reading observations and forecasts ...')

        observations = AnEnDict(args.observation, 'Observations', stations_index=args.obs_stations_index)
        print(observations)

        forecasts = AnEnDict(args.forecast, 'Forecasts', stations_index=args.fcst_stations_index)

        if args.fcst_variables is not None:
            forecasts.subset_variables(args.fcst_variables)
            assert isinstance(args.fcst_variables[0], int), "Fatal error!"

        # Remove any forecast times that contain NaN values
        nan_times_index = np.unique(np.where(np.isnan(forecasts['Data']))[2])
        forecasts.remove_times_index(nan_times_index)
        print('{} forecast times containing NaN have been removed. {} forecast times left.'.format(
            len(nan_times_index), forecasts['Data'].shape[2]))

        print(forecasts)
        print('')

        # Align observations
        aligned_obs = observations.align_observations(forecasts['Times'], forecasts['FLTs'])

        # Calculate anchor times index
        mask = [True if args.anchor_start <= time <= args.anchor_end else False for time in forecasts['Times']]
        anchor_times_index = np.where(mask)[0].tolist()

        # Calculate search times index
        mask = [True if args.search_start <= time <= args.search_end else False for time in forecasts['Times']]
        search_times_index = np.where(mask)[0].tolist()

        # Housekeeping
        del mask, nan_times_index

        # Calculate the distances of anchors and search using observations and then sort the members
        # based on the distances from lowest to highest (corresponding to most to least similar candidates).
        #
        sorted_members = sort_distance_mc(
            anchor_times_index, search_times_index, aligned_obs, scaler_type=args.scaler_type,
            parameter_weights=args.obs_weights, julian_weight=args.julian_weight, forecast_times=forecasts['Times'],
            max_workers=args.cpu_cores)

        #########################################
        # Data preprocessing for model training #
        #########################################

        # Sanity check for NaN
        assert np.array(np.where(np.isnan(forecasts['Data']))).size == 0, 'Forecasts must not have NaN values!'

        # Normalize forecast data
        original_data = torch.tensor(forecasts['Data']).float()
        scaler = ScalerClass()
        scaler.fit(original_data)
        forecasts['DataNorm'] = scaler.transform(original_data).numpy()

        # Create a dataset for training
        dataset_kwargs = {
            'lead_time_radius': args.lstm_radius,
            'forecasts': forecasts,
            'sorted_members': sorted_members,
            'num_analogs': args.analogs,
            'margin': args.dataset_margin,
            'positive_predictand_index': args.positive_index,
            'triplet_sample_prob': args.triplet_sample_prob,
            'triplet_sample_method': args.triplet_sample_method,
            'forecast_data_key': 'DataNorm',
            'to_tensor': True,
            'disable_pbar': False,
            'tqdm': tqdm,
            'fitness_num_negative': args.fitness_num_negative,
        }

        if network_type == 'ConvLSTM':
            dataset_kwargs['forecast_grid_file'] = '/Users/wuh20/tmp/GFS_1p00.txt'
            dataset_kwargs['obs_x'] = observations['Xs']
            dataset_kwargs['obs_y'] = observations['Ys']
            dataset_kwargs['metric_width'] = args.spatial_mask_width
            dataset_kwargs['metric_height'] = args.spatial_mask_height
            dataset = AnEnDatasetSpatial(**dataset_kwargs)

        else:
            dataset_kwargs['trans_args'] = args.trans_args

            if args.dataset_class == 'AnEnDatasetWithTimeWindow':
                dataset = AnEnDatasetWithTimeWindow(**dataset_kwargs)

            elif args.dataset_class == 'AnEnDatasetOneToMany':
                assert args.matching_forecast_station >= 0, 'Please set --matching-forecast-station for AnEnDatasetOneToMany!'
                dataset_kwargs['matching_forecast_station'] = args.matching_forecast_station
                dataset = AnEnDatasetOneToMany(**dataset_kwargs)
            else:
                raise Exception('Unknown dataset class {} for LSTM'.format(args.dataset_class))

        print(dataset)

        # Save samples
        sample_file = '{}/samples.pkl'.format(args.out)
        print('\nSaving samples to {} ...'.format(sample_file))
        dataset.save_samples(sample_file)
        print('Samples have been saved to {}!\n'.format(sample_file))

        # These variables have been pointed internally.
        # To ensure no unexpected changes to these, I remove the outer pointers from the global environment
        #
        del aligned_obs, anchor_times_index, search_times_index, forecasts, sorted_members, observations

        # Split the dataset into training and testing indices
        train_indices, test_indices = [], []

        for sample_index, sample_time in enumerate(dataset.anchor_sample_times):
            if sample_time < args.split:
                train_indices.append(sample_index)
            else:
                test_indices.append(sample_index)

        # Random shuffle training samples
        random.shuffle(train_indices)

        # Split the dataset based on the indices
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        print('{} samples in the training. {} samples in the testing.'.format(len(train_dataset), len(test_dataset)))
        assert len(train_dataset) > 0 and len(test_dataset) > 0, 'Train/Test datasets cannot be empty!'

        ##################
        # Model training #
        ##################

        num_forecast_variables = dataset.forecasts['Data'].shape[0]
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,
                                                   num_workers=args.load_workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch,
                                                  num_workers=args.test_load_workers)

        if args.intermediate_file:
            backup(args.intermediate_file, {
                'num_forecast_variables': num_forecast_variables,
                'scaler': scaler,
                'train_loader': train_loader,
                'test_loader': test_loader,
            })

    else:
        # If an intermediate file has been found
        num_forecast_variables, scaler, train_loader, test_loader = restore(args.intermediate_file).values()

    if network_type == 'LSTM':
        embedding_net = EmbeddingLSTM(
            input_features=num_forecast_variables,
            hidden_features=args.lstm_hidden,
            hidden_layers=args.lstm_layers,
            output_features=args.embeddings,
            scaler=scaler,
            dropout=args.dropout,
            subset_variables_index=args.fcst_variables)

    elif network_type == 'ConvLSTM':
        embedding_net = EmbeddingConvLSTM(
            input_features=num_forecast_variables,
            hidden_features=args.lstm_hidden,
            hidden_layers=args.lstm_layers,
            conv_kernel_size=args.conv_kernel_size,
            pool_kernel_size=args.pool_kernel_size,
            output_features=args.embeddings,
            dropout=args.dropout,
            scaler=scaler,
            subset_variables_index=args.fcst_variables)

    else:
        raise Exception('Unknown network type {}'.format(network_type))

    print('')
    summary_pytorch(embedding_net)
    print('')

    # Define the device
    device = torch.device('cpu') if args.use_cpu else torch.device('cuda')
    embedding_net.to(device)

    # Define training utilities
    loss_func = torch.nn.TripletMarginLoss(margin=args.train_margin)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(embedding_net.parameters(), lr=args.lr,
                                     amsgrad=args.amsgrad, weight_decay=args.wdecay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(embedding_net.parameters(), lr=args.lr,
                                      amsgrad=args.amsgrad, weight_decay=args.wdecay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(embedding_net.parameters(), lr=args.lr)
    else:
        raise Exception('Unknown optimizer {}'.format(args.optimizer))
    
    train_losses = {'mean': [], 'max': [], 'min': []}
    test_losses = {'mean': [], 'max': [], 'min': []}

    # The boolean variable for whether the process has been interrupted
    interrupt = False

    # Train the model
    try:
        for epoch in range(args.epochs):

            # Model training
            embedding_net.train()
            train_batch_losses = []

            for train_data in tqdm(train_loader, leave=False):
                a = embedding_net(train_data[0].to(device))
                p = embedding_net(train_data[1].to(device))
                n = embedding_net(train_data[2].to(device))

                train_loss = loss_func(a, p, n)
                train_batch_losses.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            train_losses['mean'].append(np.mean(train_batch_losses))
            train_losses['max'].append(np.max(train_batch_losses))
            train_losses['min'].append(np.min(train_batch_losses))

            # Model evaluation
            embedding_net.eval()
            test_batch_losses = []

            with torch.no_grad():
                for test_data in tqdm(test_loader, leave=False):
                    a = embedding_net(test_data[0].to(device))
                    p = embedding_net(test_data[1].to(device))
                    n = embedding_net(test_data[2].to(device))
                    test_loss = loss_func(a, p, n)
                    test_batch_losses.append(test_loss.item())

                test_losses['mean'].append(np.mean(test_batch_losses))
                test_losses['max'].append(np.max(test_batch_losses))
                test_losses['min'].append(np.min(test_batch_losses))
                
            print('Epoch {}/{} with index {}: train loss mean: {:.4f}; validate loss: {:.4f}'.format(
                epoch + 1, args.epochs, epoch, train_losses['mean'][-1], test_losses['mean'][-1]))

            # Model saving
            embedding_net.to(torch.device('cpu'))
            torch.jit.script(embedding_net).save("{}/embedding_epoch-{:05d}.pt".format(
                os.path.expanduser(args.out), epoch + 1))
            embedding_net.to(device)

    except KeyboardInterrupt:
        print("Keyboard interruption catched! I'm going to save the training loss.")
        interrupt = True

    # Save training progress information
    with open("{}/losses.pkl".format(os.path.expanduser(args.out)), 'wb') as f:
        pickle.dump((train_losses, test_losses), f)

    end_time = datetime.now()

    if interrupt:
        print('Model training interrupted after {}'.format(end_time - start_time))
    else:
        print('Model training complete in {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()

