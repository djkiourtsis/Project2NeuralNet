import numpy
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
    if len(argv) > 6 or len(argv) < 2 or len(argv)%2==1:
        print "The correct input format is as follows: python ann.py <filename> [h <number of hidden nodes> | p <holdout percentage>]"
        return
    inputData = []
    outputData = []
    inputWeights = []
    outputWeights = []
    numHidden = 5
    percentTest = .2
    
    for i in xrange(len(argv)):
        if argv[i] == "h":
            numHidden = int(argv[i+1])
        elif argv[i] == "p":
            percentTest = float(argv[i+1])
    
    #try and open file
    if os.path.isfile(argv[1]):
        inputData = numpy.loadtxt(argv[1],usecols=(0,1))
        outputData = numpy.loadtxt(argv[1],usecols=(2,))
        outputData = numpy.reshape(outputData, (-1, 1))
    else:
        print "This file either doesn't exist or the name was misspelled"
        sys.exit(0)
    
    #read the file into the input and output arrays two inputs and one output
    #separate the data into the input and training data based on either the given percentage or the default value of 20%
    
    
    #initialize the input and hidden layer with random weights
    
    
    #loops over the training data 1000 times
    for i in xrange(inputData.shape[0]):
        sys.exit(0)
        #set the input layer layer values form the input array
        inputLayer = inputData[i]
        #forwardprop the input to the hidden layers
        hiddenLayerIN = numpy.dot(inputLayer, inputWeights)
        hiddenLayer = sigmoid(hiddenLayerIN)
        #forwardprop hidden layer outputs to the output layer
        outputLayerIN = numpy.dot(hiddenLayer, hiddenWeights)
        outputLayer = sigmoid(outputLayerIN)
        #calculate output layer error using the given data.
        outputError = outputData[i]-outputLayer
        #calculate the output layer delta.
        outputDelta = outputError*sigmoidDeriv(outputLayerIN)
        #calc the hidden layer error using the output delta value and weights.
        hiddenError = outputDelta.dot(hiddenWeights.T)
        #calculate the hidden layer delta by using the errors
        hiddenDelta = hiddenError*sigmoidDeriv(hiddenLayerIN)
        #modify input weights and hidden layer weights depending on the delta values and the output of the neurons.
        inputWeights = inputWeights+inputLayer.dot(hiddenDelta)
        hiddenWeights = hiddenWeights+hiddenLayer.dot(outputDelta)


main(sys.argv)
