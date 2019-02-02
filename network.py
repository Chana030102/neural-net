# network.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#
# 

import math, numpy, pandas

WEIGHT_LOW = -0.05
WEIGHT_HIGH = 0.05

sigmoid = lambda x : 1/(1+math.exp(-x))

class NeuralNet:

    # Input number should not include bias
    def __init__(self, num_inputs, num_hidden_nodes, num_outputs, momentum, learn_rate):
        self.size_inputs = num_inputs + 1
        self.size_hidden = num_hidden_nodes + 1
        self.size_output = num_outputs

        self.momentum = momentum
        self.learn_rate = learn_rate
        self.epoch = 0

        # weights will be organized as a matrix. The first row is always bias node
        # row index = inputs
        # column index = hidden node
        self.weight_input_to_hidden = numpy.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH,size=(self.size_inputs,self.size_hidden))
        self.weight_input_to_hidden_prevdelta = numpy.zeros((self.size_inputs,self.size_hidden))

        # - row index = hidden node (weights for all output nodes)
        # - column index = output node (weights for all inputs for each output node)
        self.weight_hidden_to_out = numpy.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH,size=(self.size_hidden,self.size_output))
        self.weight_hidden_to_out_prevdelta = numpy.zeros((self.size_hidden,self.size_output))

        self.accuracy = pandas.DataFrame(0,index=range(0,51),columns=['test_c','test_i','train_c','train_i'])
        self.c_matrix = pandas.DataFrame(0,index=range(0,self.size_output),columns=range(0,self.size_output))

    # Forward propagation activation for network
    def activation(self, input_data, return_hidden=False):
        i_to_h_dot = numpy.dot(self.weight_input_to_hidden,[1]+input_data.transpose())
        hidden_activations = numpy.asarray(list(map(sigmoid,i_to_h_dot)))

        h_to_o_dot = numpy.dot(self.weight_hidden_to_out,[1]+hidden_activations)
        out_activations = numpy.asarray(list(map(sigmoid,h_to_o_dot)))

        if(return_hidden == False):
            return [hidden_activations,out_activations]
        else:
            return out_activations

    # Back propagation to update weights in network
    def updateWeights(self, input_data, hidden_activations, out_activations, target):
        

    # Calculate activation for inputs and record accuracy
    def evaluate(self, input_data):

    # Train network
    def train(self, input_data):
    
    # Output accuracy table to CSV file
    def report_accuracy(self, file_name):

    # Output confusion matrix
    def report_matrix(self, file_name):
