# network.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#
#

import pandas, pickle, numpy
import myperceptron

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
        self.accuracy = pandas.DataFrame(0,index=range(0,50),columns=['test_c','test_i','train_c','train_i'])
        self.epoch = 0 # count number of times trained

        # confusion matrix for evaluation 
        self.c_matrix = pandas.DataFrame(0,index=range(0,num_outputs),columns=range(0,num_outputs))

    # run network through provided data
    # set name will be "train" or "test" for data set identification
    def evaluate(self, file_name_base, num_files, increments, targets, set_name):

        # loop for each row of data
        for file_index in range(0,num_files):
            print('Evaluating ' + file_name_base + str(file_index))
            input_data = pickle.load(open(file_name_base + str(file_index),'rb'))

            for data_index in range(0,increments):    
                # loop for each node in hidden layer
                for node_index in range(0,self.size_hidden_layer):
                    self.hid_buffer[node_index] = self.hidden_layer[node_index].evaluate(input_data[data_index])

                # loop for each node in output layer
                for node_index in range(0,self.size_output_layer):
                    self.out_buffer[node_index] = self.output_layer[node_index].evaluate(self.hid_buffer)

                # update accuracy table
                if (targets[(file_index*increments) + data_index] == self.out_buffer.index(max(self.out_buffer))):
                    self.accuracy[set_name + '_c'][self.epoch] += 1
                else:
                    self.accuracy[set_name + '_i'][self.epoch] += 1
            
            del input_data
        
    # run network through provided data and add to confusion matrix
    # set name will be "train" or "test" for data set identification
    def final_evaluate(self, file_name_base, num_files, increments, targets, set_name):
        
        for file_index in range(0,num_files):
            print('Evaluating ' + file_name_base + str(file_index))
            input_data = pickle.load(open(file_name_base + str(file_index),'rb'))

            for data_index in range(0,increments):  
                # loop for each node in hidden layer
                for node_index in range(0,self.size_hidden_layer):
                    self.hid_buffer[node_index] = self.hidden_layer[node_index].evaluate(input_data[data_index])

                # loop for each node in output layer
                for node_index in range(0,self.size_output_layer):
                    self.out_buffer[node_index] = self.output_layer[node_index].evaluate(self.hid_buffer)

                # update accuracy table
                if (targets[(file_index*increments) + data_index] == self.out_buffer.index(max(self.out_buffer))):
                    self.accuracy[set_name + '_c'][self.epoch] += 1
                else:
                    self.accuracy[set_name + '_i'][self.epoch] += 1
                
                self.c_matrix[targets[file_index + data_index]][self.out_buffer.index(max(self.out_buffer))] += 1

    def train(self, file_name_base, num_files, increments, targets):
        # increment epoch count
        self.epoch += 1

        for file_index in range(0,num_files):
            print('Epoch '+ str(self.epoch) + ': ' + file_name_base + str(file_index))
            input_data = pickle.load(open(file_name_base + str(file_index),'rb'))

            for data_index in range(0,increments): 
                print(str((file_index*increments) + data_index))
                # set up expected target array for row of data
                t = [None]*self.size_output_layer
                for i in range(0,self.size_output_layer):
                    if(i == targets[(file_index*increments) + data_index]):
                        t[i] = 0.9
                    else:
                        t[i] = 0.1

                # Compute activation for nodes in hiden layer
                for hidden_index in range(0,self.size_hidden_layer):
                    self.hid_buffer[hidden_index] = self.hidden_layer[hidden_index].evaluate(input_data[data_index])

                WES = 0
                # compute activation for nodes in output layer
                # then update weights for output layer nodes
                for out_index in range(0,self.size_output_layer):
                    self.out_buffer[out_index] = self.output_layer[out_index].evaluate(self.hid_buffer)
                    self.output_layer[out_index].updateWeights(t[out_index],self.out_buffer[out_index],self.hid_buffer)

                    # get sum of product of weights and error term for each node
                    temp = numpy.multiply(self.output_layer[out_index].weights, self.output_layer[out_index].error_term)
                    WES = numpy.add(WES, temp)

                # update weights for hidden layer nodes
                for hidden_index in range(0,self.size_hidden_layer):                    
                    self.hidden_layer[hidden_index].updateWeightsHidden(WES[hidden_index],self.hid_buffer[hidden_index],input_data[data_index])
                        
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
