import csv
import numpy as np
import network

import statistics


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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

sbpList =[]
tabaccoList = []
ldlList = []
adiList = []
typeaList = []
obesityList = []
alocoholList = []
ageList = []

with open('data/heart.csv', newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        n = 0
        features = []
        labels = []

        for row in reader:
            sbpList.append(round(float(row[1]),2)) 
            tabaccoList.append(round(float(row[2]),2)) 
            ldlList.append(round(float(row[3]),2)) 
            adiList.append(round(float(row[4]),2)) 
            typeaList.append(round(float(row[6]),2))
            obesityList.append(round(float(row[7]),2)) 
            alocoholList.append(round(float(row[8]),2)) 
            ageList.append(round(float(row[9]),2)) 
            n = n + 1
    
print(tabaccoList)

sbpMean = sum(sbpList) / len(sbpList)
sbpSD = statistics.stdev(sbpList)

tabaccoMean = sum(tabaccoList)/len(tabaccoList)
tabaccoSD = statistics.stdev(tabaccoList)

ldlMean = sum(ldlList)/len(ldlList)
ldlSD = statistics.stdev(ldlList)

adiMean = sum(adiList)/len(adiList)
adiSD = statistics.stdev(adiList)

typeaMean = sum(typeaList)/len(typeaList)
typeaSD = statistics.stdev(typeaList)
    
obesityMean = sum(obesityList)/len(obesityList)
obesitySD = statistics.stdev(obesityList)

alcoholMean = sum(alocoholList)/len(alocoholList)
alcoholSD = statistics.stdev(alocoholList)

ageMean = sum(ageList)/len(ageList)
ageSD = statistics.stdev(ageList)
ageMax = ageList[np.argmax(ageList)]

# given a list with the features and label for a sample (row of the csv),
# converts it to a numeric feature vector and an integer label
# returns the tuple (feature, label)
def getDataFromSample(sample):
    #sbp(Systolic Blood Pressure) -> Standardize
    #looking at heartmeta.txt for mean and std.dev

    print("sbpMean is : ", str(sbpMean))
    print("sbpSD is : ", str(sbpSD))
    print("tabaccoMean is : ", str(tabaccoMean))
    print("tabaccoSD is : ", str(tabaccoSD))
    print("ldlMean is : ", str(ldlMean))
    print("ldlSD is : ", str(ldlSD))
    print("adiMean is : ", str(adiMean))
    print("adiSD is : ", str(adiSD))
    print("typeaMean is : ", str(typeaMean))
    print("typeaSD is : ", str(typeaSD))
    print("obesityMean is : ", str(obesityMean))
    print("obesitySD is : ", str(obesitySD))
    print("alcoholMean is : ", str(alcoholMean))
    print("alcoholSD is : ", str(alcoholSD))
    print("ageMean is : ", str(ageMean))
    print("ageSD is : ", str(ageSD))
    print("ageMax is : ", str(ageMax))
    

    #sbpMean = 138.3
    #sbpSD = 20.5
    sbpValue = cv([standardize(float(sample[1]), sbpMean, sbpSD)])

    #tabacco -> Standardize
    # tabaccoMean = 3.64
    # tabaccoSD = 4.59   
    tabaccoValue = cv([standardize(float(sample[2]), tabaccoMean, tabaccoSD)])

    #ldl(Low Density Lipoprotein cholesterol) -> Standardize
    # ldlMean = 4.74
    # ldlSD = 2.07
    ldlValue = cv([standardize(float(sample[3]), ldlMean, ldlSD)])

    #adiposity -> standardize
    # adiMean = 25.4
    # adiSD = 7.77
    adiValue = cv([standardize(float(sample[4]), adiMean, adiSD)])


    #famhist(Family history of heart disease) -> Boolean
    if (sample[5] == "Present"):
        famValue = cv([1])    
    elif (sample[5] == "Absent"):
        famValue = cv([0])
    else:
        print("Data processing error. Exiting program.")
        quit()


    #typea(type-A behaviour) -> standardize
    # typeaMean = 53.1
    # typeaSD = 9.81
    typeaValue = cv([standardize(float(sample[6]), typeaMean, typeaSD)])

    #obesity -> standardize
    # obesityMean = 26.0
    # obesitySD = 4.21
    obesityValue = cv([standardize(float(sample[7]), obesityMean, obesitySD)])

    #alcohol -> standardize
    # alcoholMean = 17
    # alcoholSD = 24.5
    alcoholValue = cv([standardize(float(sample[8]), alcoholMean, alcoholSD)])

    #age -> rescale to have 0 -10
    # ageMean = 42.8
    # ageSD = 14.6
    # ageMax = 64
    ageScale = float(sample[9])/ageMax
    ageValue = cv([standardize(ageScale, alcoholMean, alcoholSD)])


    features = np.concatenate((sbpValue, tabaccoValue, ldlValue, adiValue, famValue, typeaValue, obesityValue, alcoholValue, ageValue), axis=0)


    #chd(response, coronarytheart disease) -> value we're trying to predict
    label = int(sample[10])
    
    # return as a tuple
    return (features, label)

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    # CODE GOES HERE
    #filename = heart.csv
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        n = 0
        features = []
        labels = []

        for row in reader:
            featureVec, label = getDataFromSample(row)
            features.append(featureVec)
            labels.append(label)
            n = n + 1

    #print(features)
    #print(labels)
    
    return (n, features, labels)

def getData(filename):


    #filename = heart.csv
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        n = 0
        features = []
        labels = []

        for row in reader:
            sbpList.append(round(float(row[1]),2)) 
            tabaccoList.append(round(float(row[2]),2)) 
            ldlList.append(round(float(row[3]),2)) 
            adiList.append(round(float(row[4]),2)) 
            typeaList.append(round(float(row[6]),2))
            obesityList.append(round(float(row[7]),2)) 
            alocoholList.append(round(float(row[8]),2)) 
            ageList.append(round(float(row[9]),2)) 
            n = n + 1
    
    print(tabaccoList)

    sbpMean = sum(sbpList) / len(sbpList)
    sbpSD = statistics.stdev(sbpList)

    tabaccoMean = sum(tabaccoList)/len(tabaccoList)
    tabaccoSD = statistics.stdev(tabaccoList)


    ldlMean = sum(ldlList)/len(ldlList)
    ldlSD = statistics.stdev(ldlList)

    adiMean = sum(adiList)/len(adiList)
    adiSD = statistics.stdev(adiList)

    typeaMean = sum(typeaList)/len(typeaList)
    typeaSD = statistics.stdev(typeaList)
    
    obesityMean = sum(obesityList)/len(obesityList)
    obesitySD = statistics.stdev(obesityList)

    alcoholMean = sum(alocoholList)/len(alocoholList)
    alcoholSD = statistics.stdev(alocoholList)

    ageMean = sum(ageList)/len(ageList)
    ageSD = statistics.stdev(ageList)
    ageMax = ageList[np.argmax(ageList)]
    
    return (sbpMean, sbpSD, tabaccoMean, tabaccoSD, ldlMean, ldlSD, adiMean, adiSD, typeaMean, typeaSD, obesityMean, obesitySD, alcoholMean, alcoholSD, ageMean,ageSD, ageMax)

################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')

    # CODE GOES HERE
    ntrain = int(n * 5/6)    
    ntest = n - ntrain

    #print(features[0])
    #print(labels[0])

    # split into training and testing data
    trainingFeatures = features[:ntrain]
    trainingLabels = [onehot(label, 2) for label in labels[:ntrain]]    # training labels should be in onehot form

    testingFeatures = features[ntrain:]
    testingLabels = labels[ntrain:]

    trainingData = zip(trainingFeatures, trainingLabels)
    testingData = zip(testingFeatures, testingLabels)

    return (trainingData, testingData)


###################################################

n, features, labels = readData('data/heart.csv')

trainingData, testingData = prepData()