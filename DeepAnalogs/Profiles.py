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
# This file includes several classes for variable manipulation. These classes should be
# compilable so that they can be used in C++.
#

import torch


@torch.jit.script
class VerticalPreprocessor:
    def __init__(self, total_variables, variables_each_layer, number_of_layers, 
                 general_variables_index, vertical_variables_index):
        
        self.total_variables = total_variables
        self.variables_each_layer = variables_each_layer
        self.number_of_layers = number_of_layers
        self.general_variables_index = general_variables_index
        self.vertical_variables_index = vertical_variables_index
        
    def do(self, arr):
        # Sanity check
        assert arr.size(1) == self.total_variables, 'Expect {} variables Input {}'.format(
            self.total_variables, arr.size(1))
        assert arr.ndim == 2, "Input array must have exact 2 dimensions [samples, variables]!"

        # Initialization
        split_variables = {'general': arr[:, self.general_variables_index],
                           'vertical': arr[:, self.vertical_variables_index]}

        # Fix the order of vertical variables
        split_variables['vertical'] = split_variables['vertical'].reshape(
            -1, self.variables_each_layer, self.number_of_layers)

        return split_variables


class VerticalProfile:
    """
    This is a class to align variables from different vertical layers into a vertical
    structure. Variables that are not specified for the vertical structure or not
    recognized will be considered as general variables to be added to the features after
    the convolution layers.
    """

    def __init__(self, parameter_names,
                 vertical_names, vertical_values,
                 vertical_type='isobaricInhPa',
                 split_by='_', name_index=3,
                 vertical_value_index=1,
                 vertical_type_index=2,
                 general_names=None,
                 verbose=False):

        # Define dimensions
        self.total_variables = len(parameter_names)
        self.number_of_layers = len(vertical_values)
        self.variables_each_layer = len(vertical_names)

        if verbose:
            print('Expect {} vertical layers each with {} variables.'.format(
                self.number_of_layers, self.variables_each_layer))

        # Split variables
        self.vertical_variables = []
        self.general_variables = []

        for index, name in enumerate(parameter_names):

            str_split = name.split(split_by)
            info = [index, str_split[name_index], int(str_split[vertical_value_index]), str_split[vertical_type_index]]

            if info[3] == vertical_type and info[1] in vertical_names and info[2] in vertical_values:
                self.vertical_variables.append(info)
            else:
                info.append(name)
                self.general_variables.append(info)

        # Remove general variables that are not specified
        if general_names:
            index_to_keep = [parameter_names.index(name) for name in general_names]
            
            general_variables_subset = []
            for variable in self.general_variables:
                if variable[0] in index_to_keep:
                    general_variables_subset.append(variable)
                    
            self.general_variables = general_variables_subset

        if verbose:
            print('{} variables identified as vertical and {} as general from a total of {} variables.'.format(
                len(self.vertical_variables), len(self.general_variables), self.total_variables))

        ###################################################################################
        # Convert vertical variable info to indices                                       #
        #                                                                                 #
        # Variables are sorted first by names and then by vertical layers to ensure that, #
        # when reshaped to an array of form (channels, length), the order maintains.      #
        ###################################################################################

        # Calculate the number of vertical layers and the number variables in each layer

        msg = "Expect {} layers each with {} variables but got {} vertical variables. Impossible to align them!"
        assert self.variables_each_layer * self.number_of_layers == len(self.vertical_variables), \
            msg.format(self.number_of_layers, self.variables_each_layer, len(self.vertical_variables))

        # Convert vertical variable info to indices, sorted by vertical layers and names
        self.vertical_variables.sort(key=VerticalProfile._double_keys)
        self.vertical_variables_index = [info[0] for info in self.vertical_variables]

        #############################################################################
        # Convert general variable info to indices.                                 #
        #                                                                           #
        # At this point, general variables have been recognized and defined.        #
        # Later on in the function, these indices will be used to split variables.  #
        #############################################################################

        self.general_variables.sort(key=VerticalProfile._single_key)
        self.general_variables_index = [info[0] for info in self.general_variables]
        
        ###########################################################
        # Finally, convert numpy to tensors for C++ compatibility #
        ###########################################################
        
        self.general_variables_index = torch.tensor(self.general_variables_index)
        self.vertical_variables_index = torch.tensor(self.vertical_variables_index)
        self.total_variables = torch.tensor(self.total_variables)
        self.number_of_layers = torch.tensor(self.number_of_layers)
        self.variables_each_layer = torch.tensor(self.variables_each_layer)
    
    @staticmethod
    def _double_keys(x):
        return (x[1], x[2])
    
    @staticmethod
    def _single_key(x):
        return x[1]
    
    def get_preprocessor(self):
        return VerticalPreprocessor(self.total_variables, self.variables_each_layer, self.number_of_layers, 
                                    self.general_variables_index, self.vertical_variables_index)
