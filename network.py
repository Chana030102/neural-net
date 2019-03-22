# network.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#
# 

import math, pandas
import numpy as np

# Weight initialization range
WEIGHT_LOW    = -0.05
WEIGHT_HIGH   = 0.05

# expected values of output nodes for activation
EXPECTED_LOW  = 0.1
EXPECTED_HIGH = 0.9

sigmoid = lambda x : 1/(1+np.exp(-x))

class NeuralNet:

    # Input number should not include bias
    def __init__(self, num_inputs, num_hidden_nodes, num_outputs, momentum, learn_rate):
        self.size_inputs = num_inputs
        self.size_hidden = num_hidden_nodes
        self.size_output = num_outputs

        self.momentum = momentum
        self.learn_rate = learn_rate
        self.epoch = 0

        # weights will be organized as a matrix. The first row is always bias node
        # row index = inputs
        # column index = hidden node
        self.weight_input_to_hidden = np.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH,size=(self.size_inputs+1,self.size_hidden))
        self.weight_input_to_hidden_prevdelta = np.zeros((self.size_inputs+1,self.size_hidden))

        # - row index = hidden node (weights for all output nodes)
        # - column index = output node (weights for all inputs for each output node)
        self.weight_hidden_to_out = np.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH,size=(self.size_hidden+1,self.size_output))
        self.weight_hidden_to_out_prevdelta = np.zeros((self.size_hidden+1,self.size_output))

        self.accuracy = pandas.DataFrame(0,index=range(0,51),columns=['test_c','test_i','train_c','train_i'])
        self.c_matrix = np.zeros((num_outputs,num_outputs))

    # Forward propagation activation for network
    def activation(self, input_data, return_hidden=False):
        i_to_h_dot = np.dot(self.weight_input_to_hidden.transpose(),np.insert(input_data,0,1))
        hidden_activations = list(map(sigmoid,i_to_h_dot))

        h_to_o_dot = np.dot(self.weight_hidden_to_out.transpose(),np.append(hidden_activations,1))
        out_activations = list(map(sigmoid,h_to_o_dot))

        if(return_hidden == True):
            return [out_activations,hidden_activations]
        else:
            return out_activations

    # Back propagation to update weights in network
    def updateWeights(self, input_data, hidden_activations, out_activations, target):
        # Calculate output error terms 
        out_a = np.subtract(target,out_activations)
        out_b = np.subtract(1,out_activations)
        error_out = np.multiply(out_a,out_b)
        error_out = np.multiply(error_out,out_activations)

        # calculate hidden error terms
        hidden_a = np.subtract(1,hidden_activations)
        # indexing [1:] to ignore bias weight because bias does not need error term
        hidden_b = np.dot(self.weight_hidden_to_out[:-1],error_out)
        error_hidden = np.multiply(hidden_a,hidden_b)
        error_hidden = np.multiply(error_hidden,hidden_activations)

        # calculate hidden-to-out weight deltas
        out_a = np.multiply(error_out,self.learn_rate)
        out_a = np.outer(np.append(hidden_activations,1),out_a)
        out_b = np.multiply(self.momentum,self.weight_hidden_to_out_prevdelta)
        ho_delta = np.add(out_a,out_b)

        # calculate input-to-hidden weight deltas
        hidden_a = np.multiply(error_hidden,self.learn_rate)
        hidden_a = np.outer(hidden_a,np.append(input_data,1))
        hidden_b = np.multiply(self.momentum,self.weight_input_to_hidden_prevdelta)
        ih_delta = np.add(hidden_a.transpose(),hidden_b)

        # apply weight deltas to current weights and save new deltas as previous
        self.weight_hidden_to_out = np.add(self.weight_hidden_to_out,ho_delta)
        self.weight_input_to_hidden = np.add(self.weight_input_to_hidden,ih_delta)
        self.weight_hidden_to_out_prevdelta = ho_delta
        self.weight_input_to_hidden_prevdelta = ih_delta

    # Calculate activation for inputs and record accuracy
    def evaluate(self, set_name, input_data, targets, cmatrix=False):
        # loop through each row of data
        for data_index in range(np.shape(input_data)[0]):
            print("Epoch {} - evaluate: {} data entry {}".format(str(self.epoch),set_name,str(data_index)))
            activation = self.activation(input_data[data_index])
            prediction = activation.index(max(activation))

            # Update accuracy table
            if(prediction == targets[data_index]):
                self.accuracy[set_name+'_c'][self.epoch] += 1
            else:
                self.accuracy[set_name+'_i'][self.epoch] += 1

            if(cmatrix == True):
                # Update cmatrix
                self.c_matrix[int(targets[data_index]),prediction] += 1

    # Train network
    # targets will be an array of digits for MNIST and needs to be converted to matrix of 0.1s and 0.9s
    def train(self, input_data, targets):

        # loop through each row of data
        for data_index in range(np.shape(input_data)[0]):
            print("Epoch {} - train: data entry {}".format(str(self.epoch),str(data_index)))

            t = [EXPECTED_LOW]*self.size_output
            t[int(targets[data_index])] = EXPECTED_HIGH

            out, hidden = self.activation(input_data[data_index],return_hidden=True)
            self.updateWeights(input_data[data_index],hidden,out,t)
        
        #self.epoch += 1

    # output accuracy table to CSV file
    def report_accuracy(self,name):
        file_name = name + '_accuracy.csv'
        self.accuracy.to_csv(file_name)
        print(file_name + " has been created\n")

    # output confusion matrix to CSV file
    def report_confusion_matrix(self,name):
        file_name = name + '_cmatrix.csv'
        np.savetxt(file_name,self.c_matrix.astype(int),delimiter=',')
        print(file_name + " has been created\n")
