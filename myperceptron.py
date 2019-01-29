# myperceptron.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
# 
# Perceptron will be initialized with a specified number of inputs.
# Bias should not be included in input count argument. Bias will always be provided.

import numpy 
WEIGHT_LOW = -0.05
WEIGHT_HIGH = 0.05

class Perceptron:
    # Initialize weights randomly for each input
    def __init__(self, num_inputs, momentum, learn_rate):
        self.size = num_inputs
        self.weights = numpy.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH,size=(num_inputs,))
        self.bias_weight = numpy.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH)
        
        self.prev_delta = [0.0] * num_inputs
        self.prev_bias_delta = 0.0

        self.momentum = momentum
        self.learn_rate = learn_rate
        self.error_term = 0.0

    # Evaluates inputs by summing the products of inputs and weights
    # Return -1 if size of inputs doesn't match initialized input size for Perceptron
    # Returns 1 if evaluates greater than 0
    # Returns 0 otherwise
    def evaluate(self,inputs):
        if len(inputs) != self.size:
            return -1
        
        return numpy.sum(numpy.multiply(self.weights,inputs)) + self.bias_weight
    
    # Weight update method for output layer nodes
    def updateWeights(self,target,output,inputs):
        # delta_weight = ((learn_rate)*(error_terms)*(input)) + ((momentum)*(prev_delta))
        self.error_term = output*(1-output)*(target-output)
        
        delta = ((self.learn_rate)*(self.error_term)) 
        a = numpy.multiply(delta,inputs)
        b = numpy.multiply(self.momentum,self.prev_delta
        delta_weights = numpy.add(a,b)
        
        self.weights = numpy.add(self.weights,delta_weights)
        self.bias_weight += delta + (self.momentum*self.prev_bias_delta)

    # Weight update method for hidden layer nodes
    # WES = Weight Error Sum = sum of the products of this node's weight 
    #       to corresponding nodes of next layer
    def updateWeightsHidden(self, WES, output, inputs):
        self.error_term = output*(1-output)*WES
        
        # delta_weight = ((learn_rate)*(error_terms)*(input)) + ((momentum)*(prev_delta))
        delta = ((self.learn_rate)*(self.error_term)) 
        a = numpy.multiply(delta,inputs)
        b = numpy.multiply(self.momentum,self.prev_delta
        delta_weights = numpy.add(a,b)
        
        self.weights = numpy.add(self.weights,delta_weights)
        self.bias_weight += delta + (self.momentum*self.prev_bias_delta)

    # Provide product of error term and a specific weight
    def get_WE(self, weight_index):
        return self.error_term*self.weights[weight_index]
