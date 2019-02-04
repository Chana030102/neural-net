# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#

import numpy, pandas, pickle
import network

numpy.seterr(all='print')

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"
TITLE      = "e1_100"
TITLE_TRAIN= 'train'
TITLE_TEST = 'test'

MAX_EPOCH = 50 
INPUT_MAX = 255
increment = 1000

hidden_layer_size = 100
output_layer_size = 10
momentum = 0.9
learning_rate = 0.1

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
net = network.NeuralNet(input_size,hidden_layer_size,output_layer_size,momentum,learning_rate)

# Observe inital epoch 0 accuracy and train for 50 epochs
# Observe accuracy after each epoch
for e in range(MAX_EPOCH):
    net.evaluate(TITLE_TRAIN,train_data,train_target)
    net.evaluate(TITLE_TEST,test_data,test_target)
    net.train(train_data,train_target)

# Observe 50th epoch accuracy results
# Create confusion matrix for test data testing
net.evaluate(TITLE_TRAIN,train_data,train_target)
net.evaluate(TITLE_TEST,test_data,test_target,True)

net.report_accuracy(TITLE)
net.report_confusion_matrix(TITLE)

