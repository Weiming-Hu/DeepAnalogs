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
import torch
import pickle
import random
import argparse

import numpy as np

from tqdm import tqdm
from pprint import pprint
from datetime import datetime

from DeepAnalogs import __version__
from DeepAnalogs.AnEnDict import AnEnDict
from DeepAnalogs.EarlyStopping import EarlyStopping
from DeepAnalogs.utils import sort_distance_mc, summary_pytorch, read_yaml, validate_args
from DeepAnalogs.Embeddings import EmbeddingLSTM, EmbeddingConvLSTM, EmbeddingNaiveSpatialMask
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

    parser = argparse.ArgumentParser(description='Train an embedding network v {}'.format(__version__))
    parser.add_argument('yaml', metavar='YAML', type=str, help='A YAML file. Example files at https://github.com/Weiming-Hu/DeepAnalogs/tree/main/Examples')
    args = read_yaml(parser.parse_args().yaml)
    args = validate_args(args)

    print('Train deep network for Deep Analogs v {}'.format(__version__))
    print('Argument preview:')
    pprint(args)

    # Check for existence
    if not os.path.exists(args['io']['out']):
        os.mkdir(args['io']['out'])

    # Import a scaling method
    if args['train']['scaler_type'] == 'MinMaxScaler':
        from DeepAnalogs.Scalers import MinMaxScaler as ScalerClass
    elif args['train']['scaler_type'] == 'StandardScaler':
        from DeepAnalogs.Scalers import StandardScaler as ScalerClass
    else:
        raise Exception('The input scaler type {} is not supported!'.format(args.scaler_type))

    # Decide the type of network to train
    if args['model']['use_conv_lstm']:
        network_type = 'ConvLSTM'

        if args['model']['use_naive']:
            network_type = 'NaiveSpatialMask'

    else:
        network_type = 'LSTM'

    if not os.path.exists(args['data']['intermediate_file']):

        ############################
        # Generate reverse analogs #
        ############################

        # Read NetCDF files
        print('Reading observations and forecasts ...')

        observations = AnEnDict(args['io']['observation'], 'Observations',
                                stations_index=args['data']['obs_stations_index'])
        print(observations)

        forecasts = AnEnDict(args['io']['forecast'], 'Forecasts',
                             stations_index=args['data']['fcst_stations_index'])

        if args['data']['fcst_variables'] is not None:
            forecasts.subset_variables(args['data']['fcst_variables'])
            assert isinstance(args['data']['fcst_variables'][0], int), "Fatal error!"

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
        mask = [True if args['io']['anchor_start'] <= time <= args['io']['anchor_end']
                else False for time in forecasts['Times']]
        anchor_times_index = np.where(mask)[0].tolist()

        # Calculate search times index
        mask = [True if args['io']['search_start'] <= time <= args['io']['search_end']
                else False for time in forecasts['Times']]
        search_times_index = np.where(mask)[0].tolist()

        # Housekeeping
        del mask, nan_times_index

        # Calculate the distances of anchors and search using observations and then sort the members
        # based on the distances from lowest to highest (corresponding to most to least similar candidates).
        #
        sorted_members = sort_distance_mc(
            anchor_times_index, search_times_index, aligned_obs,
            scaler_type=args['train']['scaler_type'],
            parameter_weights=args['data']['obs_weights'],
            julian_weight=args['data']['julian_weight'],
            forecast_times=forecasts['Times'],
            max_workers=args['data']['preprocess_workers'])

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
            'lead_time_radius': args['model']['lstm_radius'],
            'forecasts': forecasts,
            'sorted_members': sorted_members,
            'num_analogs': args['data']['analogs'],
            'margin': args['data']['dataset_margin'],
            'positive_predictand_index': args['data']['positive_index'],
            'triplet_sample_prob': args['data']['triplet_sample_prob'],
            'triplet_sample_method': args['data']['triplet_sample_method'],
            'forecast_data_key': 'DataNorm',
            'to_tensor': True,
            'disable_pbar': False,
            'tqdm': tqdm,
            'fitness_num_negative': args['data']['fitness_num_negative'],
        }

        if network_type == 'ConvLSTM' or network_type == 'NaiveSpatialMask':
            dataset_kwargs['forecast_grid_file'] = args['model']['forecast_grid_file']
            dataset_kwargs['obs_x'] = observations['Xs']
            dataset_kwargs['obs_y'] = observations['Ys']
            dataset_kwargs['metric_width'] = args['model']['spatial_mask_width']
            dataset_kwargs['metric_height'] = args['model']['spatial_mask_height']
            dataset_class = AnEnDatasetSpatial

        else:
            if args['data']['dataset_class'] == 'AnEnDatasetWithTimeWindow':
                dataset_class = AnEnDatasetWithTimeWindow

            elif args['data']['dataset_class'] == 'AnEnDatasetOneToMany':
                dataset_kwargs['matching_forecast_station'] = args['data']['matching_forecast_station']
                dataset_class = AnEnDatasetOneToMany
            else:
                raise Exception('Unknown dataset class {} for LSTM'.format(args['data']['dataset_class']))

        dataset = dataset_class(**dataset_kwargs)
        print(dataset)

        # Save samples
        print('\nSaving AnEnDataset [samples, forecasts, sorted_members] ...')
        dataset.save(args['io']['out'])
        print('AnEnDataset has been saved to {}!\n'.format(args['io']['out']))

        # These variables have been pointed internally.
        # To ensure no unexpected changes to these, I remove the outer pointers from the global environment
        #
        del aligned_obs, anchor_times_index, search_times_index, forecasts, sorted_members, observations

        # Split the dataset into training and testing indices
        train_indices, test_indices = [], []

        for sample_index, sample_time in enumerate(dataset.anchor_sample_times):
            if sample_time < args['io']['split']:
                train_indices.append(sample_index)
            else:
                test_indices.append(sample_index)

        # Random shuffle training samples
        random.shuffle(train_indices)

        # Split the dataset based on the indices
        train_dataset = torch.utils.data.Subset(dataset, train_indices)

        # Create the complete test if enabled
        if args['data']['test_complete_sequence']:
            dataset_kwargs['triplet_sample_method'] = 'sequential'
            dataset_kwargs['triplet_sample_prob'] = 1
            dataset_test = dataset_class(**dataset_kwargs)
            test_dataset = torch.utils.data.Subset(dataset_test, test_indices)
        else:
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

        print('{} samples in the training. {} samples in the testing.'.format(len(train_dataset), len(test_dataset)))
        assert len(train_dataset) > 0 and len(test_dataset) > 0, 'Train/Test datasets cannot be empty!'

        ##################
        # Model training #
        ##################

        num_forecast_variables = dataset.forecasts['Data'].shape[0]

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args['train']['train_batch'],
            num_workers=args['train']['train_loaders'],
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args['train']['test_batch'],
            num_workers=args['train']['test_loaders'])

        if args['data']['intermediate_file']:
            backup(args['data']['intermediate_file'], {
                'num_forecast_variables': num_forecast_variables,
                'scaler': scaler,
                'train_loader': train_loader,
                'test_loader': test_loader})

    else:
        # If an intermediate file has been found
        num_forecast_variables, scaler, train_loader, test_loader = restore(
            args['data']['intermediate_file']).values()

    if network_type == 'LSTM':
        embedding_net = EmbeddingLSTM(
            input_features=num_forecast_variables,
            hidden_features=args['model']['lstm_hidden'],
            hidden_layers=args['model']['lstm_layers'],
            output_features=args['model']['lstm_output'],
            scaler=scaler,
            dropout=args['model']['dropout'],
            subset_variables_index=args['data']['fcst_variables'],
            fc_last=args['model']['linear_layer_last'])

    elif network_type == 'ConvLSTM':
        embedding_net = EmbeddingConvLSTM(
            input_width = args['model']['spatial_mask_width'],
            input_height = args['model']['spatial_mask_height'],
            input_features=num_forecast_variables,
            hidden_features=args['model']['lstm_hidden'],
            hidden_layers=args['model']['lstm_layers'],
            hidden_layer_types=args['model']['hidden_layer_types'],
            conv_kernel=args['model']['conv_kernel'],
            conv_padding=args['model']['conv_padding'],
            conv_stride=args['model']['conv_stride'],
            pool_kernel=args['model']['pool_kernel'],
            pool_padding=args['model']['pool_padding'],
            pool_stride=args['model']['pool_stride'],
            output_features=args['model']['lstm_output'],
            dropout=args['model']['dropout'],
            scaler=scaler,
            subset_variables_index=args['data']['fcst_variables'],
            fc_last=args['model']['linear_layer_last'])

    elif network_type == 'NaiveSpatialMask':
        embedding_net = EmbeddingNaiveSpatialMask(
            input_width = args['model']['spatial_mask_width'],
            input_height = args['model']['spatial_mask_height'],
            range_step=args['model']['range_step'],
            scaler=scaler,
            subset_variables_index=args['data']['fcst_variables'])

        print('Naive spatial mask does not have neural networks. No trainig needed.')
        print('Saving the embedding model ...')

        embedding_net.to(torch.device('cpu'))

        torch.jit.script(embedding_net).save("{}/embedding_epoch-{:05d}.pt".format(
            os.path.expanduser(args['io']['out']), 0))
        return

    else:
        raise Exception('Unknown network type {}'.format(network_type))

    print('')
    summary_pytorch(embedding_net)
    print('')

    # Define the device
    device = torch.device('cpu') if args['train']['use_cpu'] else torch.device('cuda')
    embedding_net.to(device)

    # Define training utilities
    loss_func = torch.nn.TripletMarginLoss(margin=args['train']['train_margin'])
    
    if args['train']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            embedding_net.parameters(),
            lr=args['train']['lr'],
            amsgrad=args['train']['use_amsgrad'],
            weight_decay=args['train']['wdecay'])

    elif args['train']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            embedding_net.parameters(),
            lr=args['train']['lr'],
            amsgrad=args['train']['use_amsgrad'],
            weight_decay=args['train']['wdecay'])

    elif args['train']['optimizer'] == 'Adagrad':
        optimizer = torch.optim.Adagrad(
            embedding_net.parameters(),
            lr=args['train']['lr'],
            lr_decay=args['train']['lr_decay'],
            weight_decay=args['train']['wdecay'])

    elif args['train']['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            embedding_net.parameters(),
            lr=args['train']['lr'],
            weight_decay=args['train']['wdecay'],
            momentum=args['train']['momentum'])

    else:
        raise Exception('Unknown optimizer {}'.format(args['train']['optimizer']))
    
    train_losses = {'mean': [], 'max': [], 'min': []}
    test_losses = {'mean': [], 'max': [], 'min': []}

    # The boolean variable for whether the process has been interrupted
    interrupt = False
    
    # A variable to control early stopping
    early_stopping = EarlyStopping(patience=args['train']['patience'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=5, verbose=True, min_lr=1e-5)

    # Train the model
    try:
        for epoch in range(args['train']['epochs']):

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
                
            print('Epoch {}/{}: train loss mean: {:.4f}; validate loss: {:.4f}'.format(
                epoch + 1, args['train']['epochs'], train_losses['mean'][-1], test_losses['mean'][-1]))

            # Model saving
            embedding_net.to(torch.device('cpu'))
            torch.jit.script(embedding_net).save("{}/embedding_epoch-{:05d}.pt".format(
                os.path.expanduser(args['io']['out']), epoch + 1))
            if args['io']['save_as_pure_python_module']:
                torch.save(embedding_net, "{}/embedding_epoch-{:05d}.pt_python".format(
                    os.path.expanduser(args['io']['out']), epoch + 1))
            embedding_net.to(device)
            
            # Check for early stopping
            if early_stopping(test_losses['mean'][-1]):
                break
            
            # Check for learning rate updating
            scheduler.step(test_losses['mean'][-1])

    except KeyboardInterrupt:
        print("Keyboard interruption catched! I'm going to save the training loss.")
        interrupt = True

    # Save training progress information
    with open("{}/losses.pkl".format(os.path.expanduser(args['io']['out'])), 'wb') as f:
        pickle.dump((train_losses, test_losses), f)

    end_time = datetime.now()

    if interrupt:
        print('Model training interrupted after {}'.format(end_time - start_time))
    else:
        print('Model training complete in {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
