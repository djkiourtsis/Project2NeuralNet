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
        #read the file into the input and output arrays two inputs and one output
        inputData = numpy.loadtxt(argv[1],usecols=(0,1))
        outputData = numpy.loadtxt(argv[1],usecols=(2,))
        outputData = numpy.reshape(outputData, (-1, 1))
    else:
        print "This file either doesn't exist or the name was misspelled"
        sys.exit(0)
    
    #initialize the input and hidden layer with random weights
    numpy.random.seed(1)
    inputWeights = 2*(numpy.random.random((inputData.shape[1], numHidden)))-1
    hiddenWeights = 2*(numpy.random.random((numHidden, 1)))-1
    
    #loops over the training data 1000 times
    for i in xrange(inputData.shape[0]):
        #copy input data to input array
        inputLayer = inputData[i]
        inputLayer = numpy.reshape(inputLayer, (-1, 1)).T
        #forward propegate the input to the hidden layers
        #dot product produces sum of node i output times weight Wi,j for all nodes i,j
        hiddenLayerIN = numpy.dot(inputLayer, inputWeights)
        hiddenLayer = sigmoid(hiddenLayerIN)
        #forward propegate hidden layer outputs to the output layer
        #dot product produces sum of node i output times weight Wi,j for all nodes i,j
        outputLayerIN = numpy.dot(hiddenLayer, hiddenWeights)
        outputLayer = sigmoid(outputLayerIN)
        #calculate output layer error using the given data.
        outputError = outputData[i]-outputLayer
        #calculate the output layer delta.
        outputDelta = outputError*sigmoidDeriv(outputLayerIN)
        #calculate the hidden layer error using the output delta value and weights.
        #dot product produces sum of delta[j] times weights Wi,j for all i,j (in this case i are hidden nodes and j are output nodes)
        hiddenError = numpy.dot(outputDelta, hiddenWeights.T)
        #calculate the hidden layer delta by using the errors
        hiddenDelta = hiddenError*sigmoidDeriv(hiddenLayerIN)
        #modify input weights and hidden layer weights depending on the delta values and the output of the neurons.
        #dot product produces a(i) * delta(j) for every node i connected to node j with weight weight[i][j]
        inputWeights = inputWeights+numpy.dot(inputLayer.T, hiddenDelta)
        hiddenWeights = hiddenWeights+numpy.dot(hiddenLayer.T, outputDelta)


main(sys.argv)
