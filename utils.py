#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified by brune for minigrid
@author: jacobb
"""

import torch
import numpy as np
import os
import datetime
import logging


def from_bit_to_hot(x):
    # takes in a step_numx25x4 array and turns it into a step_numx25x9 with 9 being the one-hot representation
    batch_size, dim1, dim2 = x.shape

    # Convert binary array bits to decimal values
    decimal_values = np.dot(x, 1 << np.arange(dim2-1, -1, -1))
    decimal_values = decimal_values.astype(int)

    # Initialize the one-hot array
    one_hot_encoded = np.zeros((batch_size, dim1, 13), dtype=int)

    # Efficiently assign the one-hot encoding
    rows = np.repeat(np.arange(batch_size), dim1)
    cols = np.tile(np.arange(dim1), batch_size)
    one_hot_encoded[rows, cols, decimal_values.ravel()] = 1

    # shape = batch_size x 25 x 9
    return one_hot_encoded.astype(np.float32)

def inv_var_weight(mus, sigmas):
    '''
    Accepts lists batches of row vectors of means and standard deviations, with batches along dim 0
    Return tensors of inverse-variance weighted averages and tensors of inverse-variance weighted standard deviations
    '''
    # Stack vectors together along first dimension
    mus = torch.stack(mus, dim = 0)
    sigmas = torch.stack(sigmas, dim = 0)
    # Calculate inverse variance weighted variance from sum over reciprocal of squared sigmas
    inv_var_var = 1.0 / torch.sum(1.0 / (sigmas**2), dim = 0)
    # Calculate inverse variance weighted average
    inv_var_avg = torch.sum(mus / (sigmas**2), dim = 0) * inv_var_var
    # Convert weigthed variance to sigma
    inv_var_sigma = torch.sqrt(inv_var_var)
    # And return results
    return inv_var_avg, inv_var_sigma

def softmax(x):
    '''
    Applies softmax to tensors of inputs, using torch softmax funcion
    Assumes x is a 1D vector, or batches of row vectors with the batches along dim 0
    '''
    # Return torch softmax
    return torch.nn.Softmax(dim=-1)(x)

def normalise(x):
    '''
    Normalises vector of input to unit norm, using torch normalise funcion
    Assumes x is a 1D vector, or batches of row vectors with the batches along dim 0
    '''
    # Return torch normalise with p=2 for L2 norm
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def relu(x):
    '''
    Applies rectified linear activation unit to tensors of inputs, using torch relu funcion
    '''
    # Return torch relu
    return torch.nn.functional.relu(x)

def leaky_relu(x):
    '''
    Applies leaky (meaning small negative slope instead of zeros) rectified linear activation unit to tensors of inputs, using torch leaky relu funcion
    '''
    # Return torch leaky relu [torch.nn.functional.leaky_relu(val) for val in x] if type(x) is list else
    return torch.nn.functional.leaky_relu(x)

def squared_error(value, target):
    '''
    Calculates mean squared error (L2 norm) between (list of) tensors value and target by using torch MSE loss
    Include a factor 0.5 to squared error by convention
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    '''
    # Return torch MSE loss
    if type(value) is list:
        loss = [0.5 * torch.sum(torch.nn.MSELoss(reduction='none')(value[i], target[i]),dim=-1) for i in range(len(value))]
    else:
        loss = 0.5 * torch.sum(torch.nn.MSELoss(reduction='none')(value, target),dim=-1)
    return loss

def cross_entropy(value, target):
    '''
    Calculate cross-entropy loss for each of the 25 locations individually,
    maintaining their structure within the batch.

    Returns:
    - List[torch.Tensor]: A list of loss values for each location, each tensor in the list with shape [batch_size].
    '''

    losses = []
    for i in range(value.shape[1]):  # Iterate over locations
        # Select the logits and targets for the current location across all batches
        logits = value[:, i, :]
        labels = target[:, i]

        # Calculate cross-entropy loss for the current location
        loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
        losses.append(loss)

    return losses  # A list where each element is a tensor of losses for each batch at each location

def downsample(value, target_dim):
    '''
    Does downsampling by taking the an input vector, then averaging chunks to make it of requested dimension
    Assumes x is a 1D vector, or batches of row vectors with the batches along dim 0
    '''
    # Get input dimension
    value_dim = value.size()[-1]
    # Set places to break up input vector into chunks
    edges = np.append(np.round(np.arange(0, value_dim, float(value_dim) / target_dim)),value_dim).astype(int)
    # Create downsampling matrix
    downsample = torch.zeros((value_dim,target_dim), dtype = torch.float)
    # Fill downsampling matrix with chunks
    for curr_entry in range(target_dim):
        downsample[edges[curr_entry]:edges[curr_entry+1],curr_entry] = torch.tensor(1.0/(edges[curr_entry+1]-edges[curr_entry]), dtype=torch.float)
    # Do downsampling by matrix multiplication
    return torch.matmul(value,downsample)

def make_directories():
    '''
    Creates directories for storing data during a model training run
    '''
    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = '../Summaries/' + date + '/run' + str(run) + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        script_path = run_path + 'script'
        envs_path = script_path + '/envs'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(script_path)
            os.makedirs(envs_path)
            dir_check = False
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path

def set_directories(date, run):
    '''
    Returns directories for storing data during a model training run from a given previous training run
    '''
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    run_path = '../Summaries/' + date + '/run' + str(run) + '/'
    train_path = run_path + 'train'
    model_path = run_path + 'model'
    save_path = run_path + 'save'
    script_path = run_path + 'script'
    envs_path = script_path + '/envs'
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path

def make_logger(run_path):
    '''
    Creates logger so output during training can be stored to file in a consistent way
    '''
    # Create new logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + 'report.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger

if __name__ == '__main__':
    array = np.array([[[0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 1]],

           [[0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0]]])

    converted_data = from_bit_to_hot(array)
    print(converted_data.shape)