# network.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#
# 

import math, numpy, pandas

# Weight initialization range
WEIGHT_LOW    = -0.05
WEIGHT_HIGH   = 0.05

# expected values of output nodes for activation
EXPECTED_LOW  = 0.1
EXPECTED_HIGH = 0.9

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
        i_to_h_dot = numpy.dot(self.weight_input_to_hidden.transpose(),[1]+input_data)
        hidden_activations = list(map(sigmoid,i_to_h_dot))

        h_to_o_dot = numpy.dot(self.weight_hidden_to_out.transpose(),[1]+hidden_activations)
        out_activations = list(map(sigmoid,h_to_o_dot))

        if(return_hidden == False):
            return [hidden_activations,out_activations]
        else:
            return out_activations

    # Back propagation to update weights in network
    def updateWeights(self, input_data, hidden_activations, out_activations, target):
        # Calculate output error terms 
        out_a = numpy.subtract(target,out_activations)
        out_b = numpy.subtract(1,out_activations)
        error_out = numpy.multiply(out_a,out_b)
        error_out = numpy.multiply(error_out,out_activations)

        # calculate hidden error terms
        hidden_a = numpy.subtract(1,hidden_activations)
        hidden_b = numpy.dot(numpy.transpose(self.weight_hidden_to_out),error_out)
        error_hidden = numpy.multiply(hidden_a,hidden_b)
        error_hidden = numpy.multiply(error_hidden,hidden_activations)

        # calculate hidden-to-out weight deltas
        out_a = numpy.multiply(error_out,self.learn_rate)
        out_a = numpy.multiply(out_a,hidden_activations)
        out_b = numpy.multiply(self.momentum,self.weight_hidden_to_out_prevdelta)
        self.weight_hidden_to_out_prevdelta = numpy.add(out_a,out_b)

        # calculate input-to-hidden weight deltas
        hidden_a = numpy.multiply(error_hidden,self.learn_rate)
        hidden_a = numpy.multiply(hidden_a,input_data)
        hidden_b = numpy.multiply(self.momentum,self.weight_input_to_hidden_prevdelta)
        self.weight_input_to_hidden_prevdelta = numpy.add(hidden_a,hidden_b)

        # apply weight deltas to current weights
        self.weight_hidden_to_out = numpy.add(self.weight_hidden_to_out,self.weight_hidden_to_out_prevdelta)
        self.weight_input_to_hidden = numpy.add(self.weight_input_to_hidden,self.weight_input_to_hidden_prevdelta)

    # Calculate activation for inputs and record accuracy
    def evaluate(self, set_name, input_data, targets, cmatrix=False):
        # loop through each row of data
        for data_index in (numpy.shape(input_data)[0]):
            activation = self.activation(input_data[data_index])
            prediction = activation.index(max(activation))

            # Update accuracy table
            if(prediction == targets[data_index]):
                self.accuracy[set_name+'_c'][self.epoch] += 1
            else:
                self.accuracy[set_name+'_i'][self.epoch] += 1

            # Update confusion matrix if requested
            if(cmatrix == True):
                self.c_matrix[targets[data_index]][prediction] += 1

    # Train network
    # targets will be an array of digits for MNIST and needs to be converted to matrix of 0.1s and 0.9s
    def train(self, input_data, targets):

        # loop through each row of data
        for data_index in (numpy.shape(input_data)[0]):
            t = [EXPECTED_LOW]*self.size_output
            t[targets[data_index]] = EXPECTED_HIGH

            out, hidden = self.activation(input_data[data_index],return_hidden=True)
            self.updateWeights(input_data,hidden,out,t)

    # output accuracy table to CSV file
    def report_accuracy(self,name):
        file_name = name + '_accuracy.csv'
        self.accuracy.to_csv(file_name)
        print(file_name + "has been created\n")

    # output confusion matrix to CSV file
    def report_confusion_matrix(self,name):
        file_name = name + '_cmatrix.csv'
        self.c_matrix.to_csv(file_name)
        print(file_name + "has been created\n")
