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
# This file contains utility functions.
#

import os
import gc
import math
import yaml
import time
import numpy as np
import bottleneck as bn

from tqdm import tqdm
from functools import partial
from bisect import bisect_left
from sklearn import preprocessing
from prettytable import PrettyTable
from datetime import datetime, timezone
from tqdm.contrib.concurrent import process_map


def read_yaml(file):
    with open(file, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args


def add_default_values(args):
    if 'save_as_pure_python_module' not in args['io']:
        args['io']['save_as_pure_python_module'] = False
    
    if 'fcst_variables' not in args['data']:
        args['data']['fcst_variables'] = None
    if 'obs_weights' not in args['data']:
        args['data']['obs_weights'] = None
    if 'positive_index' not in args['data']:
        args['data']['positive_index'] = None
    if 'triplet_sample_prob' not in args['data']:
        args['data']['triplet_sample_prob'] = 1.0
    if 'dataset_margin' not in args['data']:
        args['data']['dataset_margin'] = np.nan
    if 'obs_stations_index' not in args['data']:
        args['data']['obs_stations_index'] = None
    if 'fcst_stations_index' not in args['data']:
        args['data']['fcst_stations_index'] = None
    if 'preprocess_workers' not in args['data']:
        args['data']['preprocess_workers'] = os.cpu_count()
    if 'intermediate_file' not in args['data']:
        args['data']['intermediate_file'] = ''
    if 'julian_weight' not in args['data']:
        args['data']['julian_weight'] = 0.0
    if 'dataset_class' not in args['data']:
        args['data']['dataset_class'] = 'AnEnDatasetWithTimeWindow'
    if 'test_complete_sequence' not in args['data']:
        args['data']['test_complete_sequence'] = False

    if 'use_conv_lstm' not in args['model']:
        args['model']['use_conv_lstm'] = False
    if 'conv_kernel' not in args['model']:
        args['model']['conv_kernel'] = 3
    if 'conv_padding' not in args['model']:
        args['model']['conv_padding'] = 1
    if 'conv_stride' not in args['model']:
        args['model']['conv_stride'] = 1
    if 'pool_kernel' not in args['model']:
        args['model']['pool_kernel'] = 2
    if 'pool_padding' not in args['model']:
        args['model']['pool_padding'] = 0
    if 'pool_stride' not in args['model']:
        args['model']['pool_stride'] = args['model']['pool_kernel']
    if 'forecast_grid_file' not in args['model']:
        args['model']['forecast_grid_file'] = 'Not specified'
    if 'spatial_mask_width' not in args['model']:
        args['model']['spatial_mask_width'] = 5
    if 'spatial_mask_height' not in args['model']:
        args['model']['spatial_mask_height'] = 5
    if 'hidden_layer_types' not in args['model']:
        args['model']['hidden_layer_types'] = 'conv_lstm'
    if 'use_naive' not in args['model']:
        args['model']['use_naive'] = False
    if 'range_step' not in args['model']:
        args['model']['range_step'] = 1

    if 'optimizer' not in args['train']:
        args['train']['optimizer'] = 'Adam'
    if 'lr_decay' not in args['train']:
        args['train']['lr_decay'] = 0
    if 'scaler_type' not in args['train']:
        args['train']['scaler_type'] = 'MinMaxScaler'
    if 'y_scaler_type' not in args['train']:
        args['train']['y_scaler_type'] = None
    if 'train_loaders' not in args['train']:
        args['train']['train_loaders'] = os.cpu_count()
    if 'test_loaders' not in args['train']:
        args['train']['test_loaders'] = os.cpu_count()
    if 'use_cpu' not in args['train']:
        args['train']['use_cpu'] = False
    if 'train_margin' not in args['train']:
        args['train']['train_margin'] = 0.9
    if 'use_amsgrad' not in args['train']:
        args['train']['use_amsgrad'] = False
    if 'wdecay' not in args['train']:
        args['train']['wdecay'] = 0
    if 'momentum' not in args['train']:
        args['train']['momentum'] = 0
    if 'patience' not in args['train']:
        args['train']['patience'] = None
        
    return args


def validate_args(args):
    # Check groups
    expected_groups = ['io', 'data', 'model', 'train']

    assert len(args.keys()) == len(expected_groups) and \
           all([k in expected_groups for k in args.keys()]), \
        'Allowed argument groups: {}'.format(expected_groups)

    args = add_default_values(args)

    # General check
    err_msg = []
    for group in expected_groups:
        for k, v in args[group].items():

            if v == '__REQUIRED__':
                err_msg.append('Please provide argument [{}] in the group [{}]!'.format(k, group))

            if v == 'np.nan':
                args[group][k] = np.nan

            if isinstance(v, str) and len(v) > 0 and v[0] == '~':
                args[group][k] = os.path.expanduser(args[group][k])

    # Specific check
    if args['model']['use_conv_lstm']:
        grid_file = args['model']['forecast_grid_file']
        if not os.path.exists(grid_file):
            err_msg.append('Forecast grid not found: {}'.format(grid_file))

    if args['data']['triplet_sample_method'] == 'fitness':
        num_negative = args['data']['fitness_num_negative']
        if not isinstance(num_negative, int):
            err_msg.append('Invalid fitness_num_negative: {}'.format(num_negative))

    if args['data']['dataset_class'] == 'AnEnDatasetOneToMany':
        matching_station = args['data']['matching_forecast_station']
        if isinstance(matching_station, int) and matching_station > 0:
            pass
        else:
            err_msg.append('Invalid matching_forecast_station: {}'.format(matching_station))

    if len(err_msg) != 0:
        err_msg = '\n'.join(err_msg)
        raise Exception('Failed during argument validation:\n' + err_msg)

    # Change from str to datetime
    for k in ['split', 'anchor_start', 'anchor_end', 'search_start', 'search_end']:
        dt = time.strptime(args['io'][k], '%Y/%m/%d %H:%M:%S')
        args['io'][k] = datetime(*(dt[0:6]), tzinfo=timezone.utc)

    return args


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def summary(d):
    """
    Generate a summary message of the dictionary.
    :param d: A dictionary to summary.
    :return: A string as summary message.
    """
    msg_list = []

    for key, value in d.items():
        msg = '- {}: '.format(key)

        if isinstance(value, np.ndarray):
            msg += 'shape {}'.format(value.shape)
        elif len(value) == 1:
            msg += 'value {}'.format(value)
        elif isinstance(value, list):
            msg += 'length {}'.format(len(value))
        else:
            msg += '** no preview **'

        msg_list.append(msg)

    return '\n'.join(msg_list)


def binary_search(seq, v):
    """
    Carries out a binary search on a list to find the index of the first occurrence.
    Code references from https://www.geeksforgeeks.org/binary-search-bisect-in-python/.

    :param seq: A sorted list.
    :param v: A value to search for the index.
    :return: The index of the value if found or -1 if not found.
    """
    i = bisect_left(seq, v)

    if i != len(seq) and seq[i] == v:
        return i
    else:
        return -1


def sort_distance(anchor_times, search_times, arr, scaler_type, parameter_weights=None, julian_weight=0,
                  forecast_times=None, disable_pbar=False, tqdm=tqdm, return_values=False, verbose=True):
    """
    Calculates the dissimilarity (distance) between each of the anchor times and the remaining times at each location.
    This function is capable of finding multi-variate distances by setting the parameter weights. The sorted indices
    and distances are saved in a dictionary.

    :param arr: A four dimensional numpy array with the dimensions [parameters, stations, times, lead times]
    :param anchor_times: A list of time indices from the observation array as anchor points. For each of the
    anchor time index, it is compared to the time indices in search_times and the search entries will be sorted based
    on the similarity from highest to lowest (or the distance from lowest to highest).
    :param search_times: A list of time indices from the observation array as search points.
    :param scaler_type: The normalization method to use. It can be either `MinMaxScaler` or `StandardScaler`.
    :param parameter_weights: A list of weights corresponding to each parameters in the input array.
    :param disable_pbar: Whether to disable the progress bar.
    :param tqdm: A tqdm progress bar object used to wrap around enumerate function call in the loop.
    :param return_values: Whether to return the values of the sorted array. It drastically increase the memory usage.
    :param verbose: Whether to print messages
    :param julian_weight: The weight for the Julian day as a variable

    :return: A dictionary with the sorted members, 'index' and 'distance'. The members are both four dimensional
    numpy arrays with the dimensions [stations, anchor times, lead times, remaining times]. They record the order and
    the values of the similarity between the remaining times and each of the anchor time.
    """

    # Sanity check
    assert isinstance(arr, np.ndarray), 'Argument arr should be a Numpy array!'
    assert isinstance(search_times, list), 'Argument search_times should be a list'
    assert len(arr.shape) == 4, 'Argument arr should have 4 dimensions [parameters, stations, times, lead times]!'
    assert len(parameter_weights) == arr.shape[0] if parameter_weights else True, 'Too many or too few weights!'

    if isinstance(anchor_times, list):
        pass
    elif isinstance(anchor_times, int):
        anchor_times = [anchor_times]
    else:
        raise Exception('Argument anchor_times should either be a list of integers or a single integer!')

    assert len(anchor_times) == len(set(anchor_times)), 'Anchor times must not have duplicates!'
    assert max(anchor_times) <= arr.shape[2], 'Anchor time index out of bound!'

    assert julian_weight >= 0
    if julian_weight > 0:
        anchor_julians = [int(forecast_times[i].strftime('%j')) for i in anchor_times]
        search_julians = [int(forecast_times[i].strftime('%j')) for i in search_times]

    # Extract dimensions
    num_parameters, num_stations, num_times, num_lead_times = arr.shape
    num_samples = num_stations * num_times * num_lead_times

    num_anchor_times = len(anchor_times)
    num_search_times = len(search_times)

    # Default parameter weights to one if not set
    if parameter_weights:
        parameter_weights = np.array(parameter_weights)
    else:
        parameter_weights = np.ones(num_parameters)

    # I need to decide the index that has non-zero weights so that NA for 0 weighted
    # variables would not have any effect when calculating the average.
    #
    obs_vars_index = np.where(parameter_weights != 0)[0]
    parameter_weights = parameter_weights[parameter_weights != 0]

    if verbose:
        print('Observation variable index {} with weights {}'.format(obs_vars_index, parameter_weights))

    # Initialize a dictionary for indices and distances.
    # For each anchor time, the remaining times will be used as search times.
    #
    new_dimensions = (num_stations, num_anchor_times, num_lead_times, num_search_times)
    sorted_members = {
        'index': np.empty(new_dimensions, int),
        'distance': np.full(new_dimensions, np.nan),
        'anchor_times_index': anchor_times,
        'search_times_index': search_times,
        'aligned_obs': arr
    }

    if return_values:
        value_dimensions = (num_parameters, num_stations, num_anchor_times, num_lead_times, num_search_times)
        sorted_members['value'] = np.full(value_dimensions, np.nan)

    # Normalization
    if scaler_type is None:
        arr_norm = arr.copy()
        
        assert julian_weight <= 0, 'Target values need to be scaled before combining with Julian dates'
    
    else:
        if scaler_type == 'MinMaxScaler':
            scaler = preprocessing.MinMaxScaler()

            if julian_weight > 0:
                julian_max = np.max((np.max(anchor_julians), np.max(search_julians)))
                anchor_julians /= julian_max
                search_julians /= julian_max

        elif scaler_type == 'StandardScaler':
            scaler = preprocessing.StandardScaler()

            if julian_weight > 0:
                julian_mean = np.mean((anchor_julians, search_julians))
                julian_std = np.std((anchor_julians, search_julians))
                anchor_julians = (anchor_julians - julian_mean) / julian_std
                search_julians = (search_julians - julian_mean) / julian_std

        else:
            raise Exception('Unknown scaler type {}'.format(scaler_type))

        arr_norm = np.transpose(arr.reshape((num_parameters, num_samples)))
        arr_norm = scaler.fit_transform(arr_norm)
        arr_norm = np.transpose(arr_norm).reshape(arr.shape)

    # Calculate distance and sort indices
    if verbose:
        print('Calculating sorting distances ...')

    for anchor_time_index, anchor_time in enumerate(tqdm(anchor_times, disable=disable_pbar, leave=True)):

        for station_index in range(num_stations):
            for lead_time_index in range(num_lead_times):

                # Initialize lists for distances
                distances = np.full(num_search_times, np.nan)

                for search_time_index, search_time in enumerate(search_times):

                    # Prevent comparing anchor with itself
                    if search_time == anchor_time:
                        continue

                    # This is the core code for calculating the differences between anchor and search
                    difference = arr_norm[obs_vars_index, station_index, anchor_time, lead_time_index] - \
                                 arr_norm[obs_vars_index, station_index, search_time, lead_time_index]

                    # Calculate the average distance. If any entry is NA, the result would be NA.
                    if bn.anynan(difference):
                        distances[search_time_index] = np.nan
                    else:
                        distances[search_time_index] = bn.nanmean(np.abs(difference) * parameter_weights)

                        if julian_weight > 0:
                            distances[search_time_index] += julian_weight * np.abs(anchor_julians[anchor_time_index] -
                                                                                   search_julians[search_time_index])

                # Sort
                distance_order = np.argsort(distances)
                sorted_indices = [search_times[index] for index in distance_order]

                # Save
                sorted_members['index'][station_index, anchor_time_index, lead_time_index, :] = sorted_indices

                sorted_members['distance'][station_index, anchor_time_index, lead_time_index, :] = \
                    distances[distance_order]

                if return_values:
                    sorted_members['value'][:, station_index, anchor_time_index, lead_time_index, :] = \
                        arr[:, station_index, sorted_indices, lead_time_index]

    sorted_members['aligned_obs_norm'] = arr_norm

    return sorted_members


def sort_distance_mc(anchor_times, search_times, arr, scaler_type, parameter_weights=None, julian_weight=0,
                     forecast_times=None, disable_pbar=False, return_values=False, max_workers=1):
    """
    This is simply a wrapper function for sort_distance using parallel processing.
    :param anchor_times: See sort_distance
    :param search_times: See sort_distance
    :param arr: See sort_distance
    :param scaler_type: See sort_distance
    :param parameter_weights: See sort_distance
    :param julian_weight: See sort_distance
    :param disable_pbar: See sort_distance
    :param return_values: See sort_distance
    :param max_workers: The number of parallel processes to create.
    :return: See sort_distance
    """

    # Define a wrapper function to run on parallel cores
    wrapper = partial(sort_distance, search_times=search_times, arr=arr, scaler_type=scaler_type,
                      parameter_weights=parameter_weights, julian_weight=julian_weight,
                      forecast_times=forecast_times, disable_pbar=True, return_values=return_values, verbose=False)

    # Run the algorithm in parallel
    print('Sorting observations in parallel ...')
    sorted_members_list = process_map(wrapper, anchor_times, max_workers=max_workers, leave=True, disable=disable_pbar,
                                      chunksize=math.ceil(len(anchor_times) / max_workers / 10))

    # Combine list members into a big dictionary
    print('Collect results from multiple processes ...')
    sorted_members = {
        'index': np.concatenate([element['index'] for element in sorted_members_list], axis=1),
        'distance': np.concatenate([element['distance'] for element in sorted_members_list], axis=1),
        'anchor_times_index': np.concatenate([element['anchor_times_index']
                                              for element in sorted_members_list], axis=0).tolist(),
        'search_times_index': sorted_members_list[0]['search_times_index'],
        'aligned_obs': sorted_members_list[0]['aligned_obs'],
        'aligned_obs_norm': sorted_members_list[0]['aligned_obs_norm'],
    }

    if return_values:
        sorted_members['value'] = np.concatenate([element['value'] for element in sorted_members_list], axis=2),

    # Garbage collection
    gc.collect()

    return sorted_members


def summary_pytorch(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if name[:2] == 'fc' and hasattr(model, 'fc_last') and not model.fc_last: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(model)
    print(table)
    print(f"Total Trainable Params: {total_params}")
