# network.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#
#

import numpy
import myperceptron

class NeuralNet:

    def __init__(self, num_inputs, num_hidden_nodes, num_outputs, momentum, learning_rate):
        self.num_inputs       = num_inputs
        self.size_hidden_layer = num_hidden_nodes
        self.size_output_layer = num_outputs
        
        # set up layers for network
        self.hidden_layer = [myperceptron.Perceptron(num_inputs) for i in range(0,num_hidden_nodes)]
        self.output_layer = [myperceptron.Perceptron(num_hidden_nodes) for i in range(0,num_outputs)]
        
        # buffer for evaluation of node outputs
        self.hid_buffer = [None]*num_hidden_nodes
        self.out_buffer = [None]*num_outputs

        # table to keep track of accuracy for evalutations
        self.accuracy = pandas.DataFrame(columns=range['test_c','test_i','train_c','train_i'])
        self.epoch = 0 # count number of times trained

        # confusion matrix for evaluation 
        self.c_matrix = pandas.DataFrame(0,index=range(0,num_outputs),columns=range(0,num_outputs))

    # run network through provided data
    # set name will be "train" or "test" for data set identification
    def evaluate(self, input_data, targets, set_name):
        # loop for each row of data
        for data_index in range(0,len(input_data)):
            # loop for each node in hidden layer
            for node_index in range(0,self.size_hidden_layer):
                self.hid_buffer[node_index] = self.hidden_layer[node_index].evaluate(input_data[data_index])

            # loop for each node in output layer
            for node_index in range(0,self.size_output_layer):
                self.out_buffer[node_index] = self.output_layer[node_index].evaluate(self.hid_buffer)

            # update accuracy table
            if (targets[data_index] == self.out_buffer.index(max(self.out_buffer))):
                self.accuracy[set_name + '_c'][self.epoch] += 1
            else:
                self.accuracy[set_name + '_i'][self.epoch] += 1
