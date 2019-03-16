# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#

import numpy as np 
import pandas, pickle
import network

np.seterr(all='print')
TITLE      = "e1_100"

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"
TITLE_TRAIN= 'train'
TITLE_TEST = 'test'
DELIMITER  = ','

MAX_EPOCH = 2 
INPUT_MAX = 255
increment = 1000

hidden_layer_size = 100
output_layer_size = 10
momentum = 0.9
learning_rate = 0.1

# Pandas version of Importing and processing
## Import data
#traind = pandas.read_csv(TRAIN_FILE,header=None)
#testd  = pandas.read_csv(TEST_FILE ,header=None)
#
## Preprocess data 
#traind.sample(frac=1)      # shuffle training data
#trainl = traind[0].values # Save targets as a separate dataframe/array
#traind = traind.drop(columns=0)     # Remove column with target info
#traind = traind.values # convert to np array
#traind = np.divide(traind, INPUT_MAX) # scale inputs between 0 and 1 by dividing by input max value
#
#testl = testd[0].values # Save targets as a separate dataframe/array
#testd = testd.drop(columns=0)    # Remove column with target info
#testd = testd.values # convert to np array
#testd = np.divide(testd, INPUT_MAX)

# Import data
traind = np.loadtxt(TRAIN_FILE,delimiter=DELIMITER)
testd  = np.loadtxt(TEST_FILE,delimiter=DELIMITER)

# Pre-process data
# Shuffle, separate labels, and scale
np.random.shuffle(traind)
trainl = traind[:,0]
traind = np.delete(traind,0,axis=1)
traind = np.divide(traind, INPUT_MAX)

np.random.shuffle(testd)
testl  = testd[:,0]
testd  = np.delete(testd,0,axis=1)
testd  = np.divide(testd,INPUT_MAX)

input_size = len(traind[0]) # how many inputs are in one row
net = network.NeuralNet(input_size,hidden_layer_size,output_layer_size,momentum,learning_rate)

# Observe inital epoch 0 accuracy and train for 50 epochs
# Observe accuracy after each epoch
for e in range(MAX_EPOCH):
    net.evaluate(TITLE_TRAIN,traind,trainl)
    net.evaluate(TITLE_TEST,testd,testl)
    net.train(traind,trainl)

# Observe 50th epoch accuracy results
# Create confusion matrix for test data testing
net.evaluate(TITLE_TRAIN,traind,trainl)
net.evaluate(TITLE_TEST,testd,testl,True)

net.report_accuracy(TITLE)
net.report_confusion_matrix(TITLE)

