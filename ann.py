import numpy
import getopt
import os.path
import sys


def sigmoid(input):
    #sigmoid function
    return 1/(1+numpy.exp(-input))


def sigmoidDeriv(input):
    #derivative of the sigmoid function
    return sigmoid(input)*(1-sigmoid(input))


def main(argv=None):
    #check arguments & set defaults
    print len(argv)
    str(len(argv))
    if len(argv) < 1:
        print "The correct input format is as follows: python ann.py <filename> [h <number of hidden nodes> | p <holdout percentage>]"
        return
    inputData = []
    outputData = []
    inputWeights = []
    outputWeights = []
    
    try:
        options, args = getopt.getopt(argv, "h:p:")
    except getopt.GetoptError:
        print "The correct input format is as follows: python ann.py <filename> [h <number of hidden nodes> | p <holdout percentage>]"
        sys.exit(0)
    
    for option, arg in options:
        if option == "h":
            numHidden = arg
        elif option == "p":
            numPercent = arg
    
    #try and open file
    if os.path.isfile(argv[1]):
        inputData = numpy.loadtxt(argv[1],usecols=(0,1))
        outputData = numpy.loadtxt(argv[1],usecols=(2))
    else:
        print "This file either doesn't exist or the name was misspelled"
        sys.exit(0)
    
    #read the file into the input and output arrays two inputs and one output
    #separate the data into the input and training data based on either the given percentage or the default value of 20%
    
    
    #initialize the input and hidden layer with random weights
    
    
    #loops over the training data 1000 times
    for i in xrange(1000):
        #set the input layer layer values form the input array
        #forwardprop the input to the hidden layers
        #forwardprop hidden layer outputs to the output layer
        #calculate output layer error using the given data.
        #calculate the output layer delta.
        #calc the hidden layer error using the output delta value and weights.
        #calculate the hidden layer delta by using the errors
        #modify input weights and hidden layer weights depending on the delta values and the output of the neurons.
	

