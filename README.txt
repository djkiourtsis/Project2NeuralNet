Python version: 2.4.3
External libraries: numpy
Python standard libraries: os.path, math, sys

Usage: python ann.py <filename> [h <number of hidden nodes> | p <holdout
percentage>]
example: python ann.py hw5data.txt h 10 p 0.1

<filename> must be file separated into 3 columns where the first two columns
are input data and the third column is output data.

<number of hidden nodes> must be an integer greater than 0. (default 5)

<holdout percentage> must be a decimal between 0 and 1. (default 0.2)



ann.py will simulate a 3 layer neural network with 2 input nodes, 1 output
node, and a variable number of hidden nodes.  For each line of training data,
the neural network will use back propegation to modify the weights and train
the network.  This process is repeated 100 times so that smaller data sets
will still train the neural network fairly well.
