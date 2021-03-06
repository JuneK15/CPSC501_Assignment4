#CPSC501 Fall 2021
#Junehyuk Kim 30020861

import numpy as np
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

    
#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    # CODE GOES HERE

    flattenTraining = [f.reshape(784,1) for f in train_features]
    flattenTest = [f.reshape(784,1) for f in test_features]
    
    trainingFeatures = [feature/255 for feature in flattenTraining]
    testFeatures = [feature/255 for feature in flattenTest]
    
    trainLabels = [onehot(label, 10) for label in train_labels]

    trainingData = zip(trainingFeatures, trainLabels)
    testingData = zip(testFeatures, test_labels)

    
       
    return (trainingData, testingData)
    
###################################################################


trainingData, testingData = prepData()

#Base network
# net = network.Network([784,10])
# net.SGD(trainingData, 10, 10, 10, test_data = testingData)

# Better network
# net = network.Network([784,30,10])
# net.SGD(trainingData, 10, 10, 1, test_data = testingData)

start_time = time.time()

net = network.Network([784,30,10])
net.SGD(trainingData, 20, 10, 3.5, test_data = testingData)

print("--- %s seconds ---" % (time.time() - start_time))

#network.saveToFile(net, "part2.pkl")