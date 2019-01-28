# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2

import numpy
import pandas
import network
import multiprocessing

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"

# Import data
train_data = pandas.read_csv(TRAIN_FILE,header=None)
test_data  = pandas.read_csv(TEST_FILE ,header=None)

# Preprocess data 
train_data.sample(frac=1)      # shuffle training data
train_target = train_data[0].values # Save targets as a separate dataframe/array
train_data.drop(columns=0)     # Remove column with target info
train_data = train_data.values # convert to numpy array
train_data = numpy.divide(train_data, INPUT_MAX) # scale inputs between 0 and 1 by dividing by input max value

test_target = test_data[0].values # Save targets as a separate dataframe/array
test_data.drop(columns=0)    # Remove column with target info
test_data = test_data.values # convert to numpy array
test_data = numpy.divide(test_data, INPUT_MAX)

input_size = len(train_data[0]) # how many inputs are there

