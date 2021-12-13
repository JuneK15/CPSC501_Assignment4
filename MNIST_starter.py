#CPSC501 Fall 2021
#Junehyuk Kim 30020861


#Running the environment
#tutorial-env\Scripts\activate.bat

import numpy as np
import idx2numpy
import network

import time

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    images = idx2numpy.convert_from_file(imagefile) 
    
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    # CODE GOES HERE
    flattenImages = [img.reshape(784,1) for img in images]
    
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    # CODE GOES HERE
    features = [flattenImg/255 for flattenImg in flattenImages]

    return features


# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training labels in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    # CODE GOES HERE
    ntrain, train_labels = getLabels(trainingLabelFile)

    #train_labels = onehot(ntrain, 10)    #size= 10 for oneshot
    trainLabels = [onehot(label, 10) for label in train_labels]

    ntest, test_labels = getLabels(testingLabelFile)

    trainingFeatures = getImgData(trainingImageFile)
    testingFeatures = getImgData(testingImageFile)

    trainingData = zip(trainingFeatures, trainLabels)
    testingData = zip(testingFeatures, test_labels)

       
    return (trainingData, testingData)
    

###################################################


trainingData, testingData = prepData()

#Base network
# net = network.Network([784,10,10])
# net.SGD(trainingData, 10, 10, .1, test_data = testingData)

#net.SGD(trainingData, 10, 10, 1, test_data = testingData)

# net = network.Network([784,30,10])
# net.SGD(trainingData, 10, 10, 1, test_data = testingData)

start_time = time.time()

net = network.Network([784,30,10])
net.SGD(trainingData, 20, 10, 3.5, test_data = testingData)

print("--- %s seconds ---" % (time.time() - start_time))

network.saveToFile(net, "part1.pkl")
