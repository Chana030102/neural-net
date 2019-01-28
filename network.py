# network.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#
#

import numpy
import myperceptron
import pandas

class NeuralNet:

    def __init__(self, num_inputs, num_hidden_nodes, num_outputs, momentum, learning_rate):
        self.num_inputs       = num_inputs
        self.size_hidden_layer = num_hidden_nodes
        self.size_output_layer = num_outputs
        
        # set up layers for network
        self.hidden_layer = [myperceptron.Perceptron(num_inputs, momentum, learning_rate) for i in range(0,num_hidden_nodes)]
        self.output_layer = [myperceptron.Perceptron(num_hidden_nodes, momentum, learning_rate) for i in range(0,num_outputs)]
        
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

    def train(self, input_data, targets):
         # loop through each row of data
        for data_index in range(0,len(input_data)):
            # set up expected target array for row of data
            t = [None]*self.size_output_layer
            for i in range(0,self.size_output_layer):
                if(i == targets[data_index]):
                    t[i] = 0.9
                else:
                    t[i] = 0.1

            # Compute activation for nodes in hiden layer
            for hidden_index in range(0,self.size_hidden_layer):
                self.hid_buffer[hidden_index] = self.hidden_layer[hidden_index].evaluate(input_data[data_index])

            # compute activation for nodes in output layer
            # then update weights for output layer nodes
            for out_index in range(0,self.size_output_layer):
                self.out_buffer[out_index] = self.output_layer[out_index].evaluate(self.hid_buffer)
                self.output_layer[out_index].updateWeights(t[out_index],self.out_buffer[out_index],input_data[data_index])

            # update weights for hidden layer nodes
            for hidden_index in range(0,self.size_hidden_layer):
                WES = 0
                
                # calculate sum of products of weight and output node error term
                for out_index in range(0,self.size_output_layer):
                    WES += self.output_layer[out_index].get_WE(hidden_index)
                
                self.hidden_layer[hidden_index].updateWeightsHidden(WES,self.hid_buffer[hidden_index],input_data)
        
        # increment epoch count
        self.epoch += 1

    def report_accuracy(self):
        

    def report_confusion_matrix(self):

