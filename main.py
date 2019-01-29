# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2

import numpy, pandas, pickle
import network

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"
TITLE      = "test"

MAX_EPOCH = 2
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
train_files = len(train_data)/increment
test_files = len(test_data)/increment

# break files up into batches, 1000 per file
for i in range(0,train_files):
    pickle.dump(train_data[(i*increment):(increment)*(i+1)], open('train_data_'+str(i),'wb'))

for i in range(0,test_files):
    pickle.dump(train_data[(i*increment):(increment)*(i+1)], open('test_data_'+str(i),'wb'))

del train_data, test_data

# initialize network
net = network.NeuralNet(input_size, hidden_layer_size, output_layer_size, momentum, learning_rate)

# observe initial epoch 0 accuracy
# then train and observe accuracy for 49 epochs
for e in range(0,MAX_EPOCH):
    net.evaluate('train_data_',train_files,increment,train_target,"train")
    net.evaluate('test_data_',test_files,increment,train_target,"test")
    net.train('test_data_',test_files,increment,train_target)

# observe 50th epoch training
# create confusion matrix using test data
net.evaluate('train_data_',train_files,increment,train_target,"train")
net.final_evaluate('test_data_',test_files,increment,train_target,"test")

net.report_accuracy(TITLE)
net.report_confusion_matrix(TITLE)
