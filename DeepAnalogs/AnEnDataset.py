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
# This file defines a customized dataset for perfect analogs.
#

import torch
import pickle
import random
import itertools

import numpy as np
import bottleneck as bn

from tqdm import tqdm
from torch.utils.data import Dataset
from DeepAnalogs.AnEnDict import AnEnDict


class AnEnDataset(Dataset):
    """
    AnEnDataset is an abstract class for generating triplet samples for training the embedding network.
    This class only prepares the triplet sample indices of forecasts to avoid copying forecast values which
    might potentially be very memory and computationally expensive. The values should be copied when the forecasts
    are actually indexed.

    This class contains a pointer to the input forecasts and sorted members but this could be potentially dangerous
    if they are changed after this class and the corresponding indices have been created. So make sure they ARE NOT
    CHANGED after this class is created.

    The BEST PRACTICE is to remove them (del *) from the outer environment once you have created an object of this
    class to ensure that forecasts can only be access from objects of this class.
    """

    def __init__(self, forecasts, sorted_members, num_analogs,
                 margin=np.nan, positive_predictand_index=None,
                 triplet_sample_prob=1, triplet_sample_method='fitness',
                 forecast_data_key='Data', to_tensor=True, disable_pbar=False, tqdm=tqdm,
                 fitness_num_negative=1, add_lead_time_index=False, trans_args=None):
        """
        Initialize an AnEnDataset

        :param forecasts: An AnEnDict for forecasts
        :param sorted_members: A dictionary for sorted members
        :param num_analogs: The number of analogs to extract from search entries
        :param margin: The distance margin while creating the triplet. If the positive distance plus the margin is
        still smaller than the negative distance, this triplet is considered too easy and will be ignored.
        :param positive_predictand_index: If the positivity of the predictand is of concern, set this to the
        index that points to the predictand that should be positive in the key `aligned_obs` from the sorted members.
        :param triplet_sample_prob: The sample probability for whether to include a given triplet sample
        :param triplet_sample_method: The sample method
        :param forecast_data_key: The forecast key to use for querying data values
        :param to_tensor: Whether to convert results to tensors
        :param disable_pbar: Whether to be disable the progress bar
        :param tqdm: A tqdm progress bar
        :param fitness_num_negative: If the sample method is `fitness`, this argument specifies the number of
        negative candidates to select for each positive candidate. The selection will be sampling without replacement
        to ensure that a particular negative candidate is only selected once for a particular positive candidate.
        :param add_lead_time_index: Whether to add lead time index in the results of __get_item__
        :param trans_args Arguments as a dictionary for the transformation in fitness selection
        """

        # Sanity checks
        assert isinstance(forecasts, AnEnDict), 'Forecasts must be an object of AnEnDict!'
        assert isinstance(sorted_members, dict), 'Sorted members must be an object of dictionary!'

        expected_dict_keys = ['index', 'distance', 'anchor_times_index', 'search_times_index']
        assert all([key in sorted_members.keys() for key in expected_dict_keys]), \
            '{} are required in sorted members'.format(sorted_members)
        assert num_analogs <= sorted_members['index'].shape[3], 'Not enough search entries to select analogs from!'

        if positive_predictand_index is not None:
            assert 0 <= positive_predictand_index < sorted_members['aligned_obs'].shape[0]

        # Decide the triplet selection method
        if triplet_sample_method == 'fitness':
            select_func = self._select_fitness
        elif triplet_sample_method == 'sequential':
            select_func = self._select_sequential
        else:
            raise Exception('Unknown selection method {}!'.format(triplet_sample_method))

        # These variables will be used inside the for loops
        num_stations = sorted_members['index'].shape[0]
        num_lead_times = sorted_members['index'].shape[2]
        self.num_total_entries = sorted_members['index'].shape[3]

        # Initialization
        self.forecasts = forecasts
        self.sorted_members = sorted_members
        self.num_analogs = num_analogs
        self.margin = margin
        self.positive_predictand_index = positive_predictand_index
        self.triplet_sample_prob = triplet_sample_prob
        self.triplet_sample_method = triplet_sample_method
        self.forecast_data_key = forecast_data_key
        self.to_tensor = to_tensor
        self.fitness_num_negative = fitness_num_negative
        self.add_lead_time_index = add_lead_time_index
        self.tqdm = tqdm
        self.disable_pbar = disable_pbar

        self.samples = []
        self.anchor_sample_times = []
        self.positive_sample_times = []
        self.negative_sample_times = []

        # Decide the transformation function
        self.trans_args = trans_args

        if self.trans_args is None:
            self.trans_func = None
        else:
            assert isinstance(trans_args, dict), 'Transformation arguments must be a dictionary! Got {}'.format(type(trans_args))

            if self.trans_args['method'] == 'exponential':
                for k in ['a', 'b', 'c']:
                    self.trans_args[k] = float(self.trans_args[k])

                self.trans_func = AnEnDataset._transform_exponential
            else:
                raise Exception('No supported type: {}'.format(self.trans_args['method']))

        # Create index samples
        #
        # Each sample is a length-of-5 list containing the following information:
        # - the station index
        # - the lead time index
        # - the anchor time index
        # - the positive candidate time index
        # - the negative candidate time index
        #
        print('Generating triplet samples ...')

        with self.tqdm(total=num_stations * num_lead_times, disable=self.disable_pbar, leave=True) as pbar:
            for station_index in range(num_stations):
                for lead_time_index in range(num_lead_times):

                    for anchor_index, anchor_time_index in enumerate(sorted_members['anchor_times_index']):

                        # If the predictand should be positive, exclude NaN and non-positive cases
                        if positive_predictand_index is not None:
                            o = sorted_members['aligned_obs'][
                                positive_predictand_index, station_index, anchor_time_index, lead_time_index]

                            if np.isnan(o) or o <= 0:
                                continue

                        # Generate triplets for this [station, lead time, anchor] from all possible search entries
                        select_func(station_index, lead_time_index, anchor_index, anchor_time_index)

                    # Update the progress bar
                    pbar.update(1)

    def save_samples(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.samples, f)

    def _select_sequential(self, station_index, lead_time_index, anchor_index, anchor_time_index):
        """
        Sequential selection is defined as follows:

        1. Each search entry within the number of analogs from the start will be positive and the entry at the number
        of analogs is the negative.
        2. The entry at the number of the analogs is the positive and all entries left are considered negative compared
        to the positive entry.
        """

        for analog_index in range(self.num_analogs):

            positive_candidate_index = analog_index
            negative_candidate_index = self.num_analogs

            self._check_and_add(station_index, anchor_index, lead_time_index,
                                positive_candidate_index, negative_candidate_index, anchor_time_index)

        for index_offset in range(1, self.num_total_entries - self.num_analogs):

            positive_candidate_index = self.num_analogs
            negative_candidate_index = positive_candidate_index + index_offset

            self._check_and_add(station_index, anchor_index, lead_time_index,
                                positive_candidate_index, negative_candidate_index, anchor_time_index)

    def _select_fitness(self, station_index, lead_time_index, anchor_index, anchor_time_index):
        """
        Fitness selection is defined as follows:

        1. Search entries within the number of analogs are positive and every entries left are considered negative.
        2. For each positive entry, several negative entries can be selected to form a triplet.
        3. Negative entries are selected with a probability that is proportional to its normalized fitness.
        """

        # Get the distance for all negative entries
        distance = self.sorted_members['distance'][station_index, anchor_index, lead_time_index, self.num_analogs:]

        # Inverse distances
        distance_inverse = bn.nansum(distance) - distance

        # Replace NAN with 0
        distance_inverse[np.isnan(distance_inverse)] = 0

        if self.trans_func is not None:
            distance_inverse = self.trans_func(distance_inverse, **self.trans_args)

        for analog_index in range(self.num_analogs):
            positive_candidate_index = analog_index

            # Normalize the inverse distance to initialize fitness
            fitness = distance_inverse / bn.nansum(distance_inverse)

            for repetition in range(self.fitness_num_negative):
                # Calculate cumulative sum
                fitness_cumsum = np.cumsum(fitness)

                # Decide on the negative candidate
                negative_candidate_index = np.digitize(random.random(), fitness_cumsum)

                # Remove this negative candidate from future selection
                fitness[negative_candidate_index] = 0

                # Rescale the fitness to [0, 1]
                fitness = fitness / bn.nansum(fitness)

                self._check_and_add(station_index, anchor_index, lead_time_index, positive_candidate_index,
                                    negative_candidate_index + self.num_analogs, anchor_time_index)

    def _check_and_add(self, station_index, anchor_index, lead_time_index,
                       positive_candidate_index, negative_candidate_index, anchor_time_index):
        """
        Checks validity of the specified triplet and then add the valid triplet to the sample list.

        :param station_index: The station index
        :param anchor_index: The anchor index to be used with sorted members
        :param lead_time_index: The lead time index
        :param positive_candidate_index: The positive search entry index
        :param negative_candidate_index: The negative search entry index
        :param anchor_time_index: The anchor index to be used with forecasts
        """

        # Check for the probability of random sampling
        if random.random() > self.triplet_sample_prob:
            return

        # This is the distance between the positive candidate and the anchor
        d_p = self.sorted_members['distance'][station_index, anchor_index, lead_time_index, positive_candidate_index]

        # This is the distance between the negative candidate and the anchor
        d_n = self.sorted_members['distance'][station_index, anchor_index, lead_time_index, negative_candidate_index]

        # Distances should both be valid and they should be different. Otherwise, skip this pair.
        if np.isnan(d_p) or np.isnan(d_n) or d_p == d_n:
            return

        # The comparison must not be negative
        if d_p > d_n:
            raise Exception('I found a distance pair that is not sorted! This is fatal!')

        if d_p + self.margin < d_n:
            # This triplet is considered too easy, skip it
            return

        # This is the index of the positive candidate
        i_p = self.sorted_members['index'][station_index, anchor_index, lead_time_index, positive_candidate_index]

        # This is the index of the negative candidate
        i_n = self.sorted_members['index'][station_index, anchor_index, lead_time_index, negative_candidate_index]

        # Construct a triplet
        triplet = [station_index, lead_time_index, anchor_time_index, i_p, i_n]

        self.samples.append(triplet)
        current_lead_time = self.forecasts['FLTs'][lead_time_index]
        self.positive_sample_times.append(self.forecasts['Times'][i_p] + current_lead_time)
        self.negative_sample_times.append(self.forecasts['Times'][i_n] + current_lead_time)
        self.anchor_sample_times.append(self.forecasts['Times'][anchor_time_index] + current_lead_time)

    def _get_summary(self):
        """
        Generates a list of messages as a summary for the dataset.
        :return: A list of messages
        """

        assert len(self.samples) == len(self.positive_sample_times) == \
               len(self.negative_sample_times) == len(self.anchor_sample_times), \
            'Assertion failed during the summary: sample length mismatch (samples and sample times)!'

        msg = [
            '*************** A Customized Dataset for AnEn ***************',
            'Class name: {}'.format(type(self).__name__),
            'Number of analogs: {}'.format(self.num_analogs),
            'Triplet margin: {}'.format(self.margin),
            'Predictand being positive: {}'.format('No' if self.positive_predictand_index is None
                                                   else 'Yes (index: {})'.format(self.positive_predictand_index)),
            'Triplet sample probability: {}'.format(self.triplet_sample_prob),
            'Triplet sample method: {}'.format(self.triplet_sample_method),
        ]

        if self.triplet_sample_method == 'fitness':
            msg.append('Number of negative candidates (fitness selection): {}'.format(self.fitness_num_negative))

        msg.extend([
            'Forecast data key: {}'.format(self.forecast_data_key),
            'Convert to tensor: {}'.format(self.to_tensor),
            'Number of total triplets: {}'.format(len(self)),
            'Add lead time index for one hot coding: {}'.format(self.add_lead_time_index),
            '********************** End of messages **********************',
        ])

        return msg

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return '\n'.join(self._get_summary())

    def __getitem__(self, index):
        """
        Returns the triplet forecasts [anchor, positive, negative]. This operation actually copies the data values.

        :param index: A sample index
        :return: The returned sample is a list of 3 arrays, an anchor, a positive, and a negative forecast. All three
         forecasts have the exact same dimensions. The dimensions are [parameters, 1 station, 1 lead time].
        """

        assert isinstance(index, int), "Only support indexing using a single integer!"

        # Extract the triplet sample
        triplet = self.samples[index]

        # Get forecast values at a single station and a single lead time
        anchor = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[2], triplet[1]]
        positive = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[3], triplet[1]]
        negative = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[4], triplet[1]]

        # Fix dimensions to be [parameters, 1 station, 1 lead time]
        anchor = np.expand_dims(anchor, (1, 2))
        positive = np.expand_dims(positive, (1, 2))
        negative = np.expand_dims(negative, (1, 2))

        if self.to_tensor:
            anchor = torch.tensor(anchor, dtype=torch.float)
            positive = torch.tensor(positive, dtype=torch.float)
            negative = torch.tensor(negative, dtype=torch.float)

        ret = [anchor, positive, negative]

        if self.add_lead_time_index:
            lead_time_index = triplet[1]

            if self.to_tensor:
                lead_time_index = torch.tensor(lead_time_index, dtype=torch.long)

            ret.append(lead_time_index)

        return ret

    @staticmethod
    def _transform_exponential(y, a, b, c, method):
        """
        Multiply the input `y` with a exponential distribution transformation. The exponential distribution function
        is as follows:
            y_multiplier = a * np.exp(-b * x), where x is the rank of y normalized to `[0, c]`.
        """
        # An example:
        # a = 0.05, b = 0.02, c = 100

        x = np.arange(0, len(y), 1)
        x = x / bn.nanmax(x) * c

        multiplier = a * np.exp(-b * x)
        return y * multiplier


class AnEnDatasetWithTimeWindow(AnEnDataset):
    """
    AnEnDatasetWithTimeWindow is inherited from AnEnDataset. Instead of generating forecast triplets at a single
    lead time, this dataset prepares the triplets from nearby forecast lead times to form a small time window. This
    behavior mirrors how Analog Ensembles are typically generated with a small lead time window.
    """

    def __init__(self, lead_time_radius, **kw):
        """
        Initialize an AnEnDatasetWithTimeWindow class
        :param lead_time_radius: The radius of lead times to include. The lead time window at lead time t will be
        [t - lead_time_radius, t + lead_time_radius].
        :param kw: Additional arguments to `AnEnDataset`
        """
        super().__init__(**kw)

        self.lead_time_radius = lead_time_radius
        num_lead_times = self.forecasts[self.forecast_data_key].shape[3]

        # Calculate a mask for which samples to keep or to remove
        keep_samples = [True if self.samples[sample_index][1] - lead_time_radius >= 0 and
                                self.samples[sample_index][1] + lead_time_radius < num_lead_times
                        else False for sample_index in range(len(self))]

        # Copy samples and times
        self.samples = list(itertools.compress(self.samples, keep_samples))
        self.anchor_sample_times = list(itertools.compress(self.anchor_sample_times, keep_samples))
        self.positive_sample_times = list(itertools.compress(self.positive_sample_times, keep_samples))
        self.negative_sample_times = list(itertools.compress(self.negative_sample_times, keep_samples))

    def __str__(self):
        msg = super()._get_summary()
        msg.insert(-1, 'Lead time radius: {}'.format(self.lead_time_radius))
        return '\n'.join(msg)

    def __getitem__(self, index):
        assert isinstance(index, int), "Only support indexing using a single integer!"

        # Extract the triplet sample
        triplet = self.samples[index]

        # Determine the start and the end indices for the lead time window
        flt_left = triplet[1] - self.lead_time_radius
        flt_right = triplet[1] + self.lead_time_radius + 1

        # Get forecast values at a single station and from a lead time window
        anchor = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[2], flt_left:flt_right]
        positive = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[3], flt_left:flt_right]
        negative = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[4], flt_left:flt_right]

        # Fix dimensions
        anchor = np.expand_dims(anchor, 1)
        positive = np.expand_dims(positive, 1)
        negative = np.expand_dims(negative, 1)

        if self.to_tensor:
            anchor = torch.tensor(anchor, dtype=torch.float)
            positive = torch.tensor(positive, dtype=torch.float)
            negative = torch.tensor(negative, dtype=torch.float)

        ret = [anchor, positive, negative]

        if self.add_lead_time_index:
            lead_time_index = triplet[1]

            if self.to_tensor:
                lead_time_index = torch.tensor(lead_time_index, dtype=torch.long)

            ret.append(lead_time_index)

        return ret


class AnEnOneToMany(AnEnDatasetWithTimeWindow):
    """
    AnEnOneToMany is inherited from AnEnDatasetWithTimeWindow. It is mostly the same as AnEnDatasetWithTimeWindow
    except that AnEnOneToMany only accepts one stations in the observation dataset and multiple stations in the
    forecast dataset. Users need to specify which forecast stations is the matching station to the observation
    stations. However, when creating triplets, forecasts from all stations will be used to be compared to the forecasts
    at the matching station.
    """

    def __init__(self, matching_forecast_station, **kw):
        """
        Initialize an AnEnOneToMany class
        :param matching_forecast_station: The index of the forecast station is matches the observation station.
        :param kw: Additional arguments to `AnEnDatasetWithTimeWindow`
        """

        # Sanity check
        err_msg = 'Invalid matching station index (). The total number of forecast stations is {}'.format(
            matching_forecast_station, kw['forecasts'][kw['forecast_data_key']].shape[1])

        assert kw['forecasts'][kw['forecast_data_key']].shape[1] > matching_forecast_station, err_msg
        assert kw['sorted_members']['index'].shape[0] == 1, 'This class only supports having one observation station!'

        super().__init__(**kw)

        self.matching_forecast_station = matching_forecast_station

        # This is where AnEnOneToMany starts to differ from the base classes. Triplets will be duplicated with changing
        # the station indices. Because not only the matching station is going to be similar, all stations from forecasts
        # should be considered similar to the matching station.
        #

        assert len(np.unique([sample[0]] for sample in self.samples)) == 1, 'Fatal! There should be only 1 station!'

        # Create new samples with changing the station index
        print('Enumerating station indices with samples ...')
        new_samples = []
        num_stations = self.forecasts[self.forecast_data_key].shape[1]

        for sample in self.tqdm(self.samples, disable=self.disable_pbar, leave=True):
            for station_index in range(num_stations):
                sample[0] = station_index
                new_samples.append(sample.copy())

        del self.samples
        self.samples = new_samples

    def __str__(self):
        msg = [
            super().__str__(),
            'Matching forecast station index: {}'.format(self.matching_forecast_station),
            'Number of forecast stations: {}'.format(self.forecasts[self.forecast_data_key].shape[1]),
        ]
        return '\n'.join(msg)

    def __getitem__(self, index):
        assert isinstance(index, int), "Only support indexing using a single integer!"

        # Extract the triplet sample
        triplet = self.samples[index]

        # Determine the start and the end indices for the lead time window
        flt_left = triplet[1] - self.lead_time_radius
        flt_right = triplet[1] + self.lead_time_radius + 1

        # Get forecast values at a single station and from a lead time window
        anchor = self.forecasts[self.forecast_data_key][:, self.matching_forecast_station, triplet[2], flt_left:flt_right]
        positive = self.forecasts[self.forecast_data_key][:, triplet[0], triplet[3], flt_left:flt_right]
        negative = self.forecasts[self.forecast_data_key][:, self.matching_forecast_station, triplet[4], flt_left:flt_right]

        # Fix dimensions
        anchor = np.expand_dims(anchor, 1)
        positive = np.expand_dims(positive, 1)
        negative = np.expand_dims(negative, 1)

        if self.to_tensor:
            anchor = torch.tensor(anchor, dtype=torch.float)
            positive = torch.tensor(positive, dtype=torch.float)
            negative = torch.tensor(negative, dtype=torch.float)

        ret = [anchor, positive, negative]

        if self.add_lead_time_index:
            lead_time_index = triplet[1]

            if self.to_tensor:
                lead_time_index = torch.tensor(lead_time_index, dtype=torch.long)

            ret.append(lead_time_index)

        return ret
