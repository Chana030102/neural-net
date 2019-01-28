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
        
        self.prev_delta = [0] * num_inputs
        self.prev_bias_delta = 0

        self.momentum = momentum
        self.learn_rate = learn_rate

    # Evaluates inputs by summing the products of inputs and weights
    # Return -1 if size of inputs doesn't match initialized input size for Perceptron
    # Returns 1 if evaluates greater than 0
    # Returns 0 otherwise
    def evaluate(self,inputs):
        if len(inputs) != self.size:
            return -1
        
        return numpy.sum(numpy.multiply(self.weights,inputs)) + self.bias_weight
    
    # delta = ((learn_rate)*(error_terms)*(output)) + ((momentum)*(prev_delta))
    def updateWeights(self,target,output,inputs,error_terms):
        delta = self.learn_rate*(target-output)
        delta_weights = numpy.multiply(delta,inputs)
        self.weights = numpy.add(self.weights,delta_weights)
        self.bias_weight += delta