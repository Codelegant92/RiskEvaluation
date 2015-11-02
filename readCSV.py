__author__ = 'yangchen'

import csv
from commonFunction import *
from svm_classification import svmclassifier

import numpy as np
def generateRichnessDataset():
    richNess = []
    sampleNum = np.zeros(50)
    featureNum = 26
    for num in xrange(1, 51):
        filePath = 'Data/trainingData/' + str(num) + '.csv'
        f1 = open(filePath, 'rb')
        nullNum = np.zeros(featureNum)
        for row in csv.reader(f1):
            j = 0
            sampleNum[num-1] += 1      #count the total number of samples of each platform
            for rows in row:
                if(j < featureNum):
                    if(rows == '\\N' or rows == ''):
                        nullNum[j] += 1
                    j += 1
        richNess.append((sampleNum[num - 1] - nullNum) * 1.0 / sampleNum[num - 1])
        f1.close()
    #print(sampleNum)
    testLabel = np.zeros(50)
    for index in range(20):
        testLabel[index] = 1
    return(richNess, testLabel)

dataFeature, dataLabel = generateRichnessDataset()
featureFolder, labelFolder = crossValidation(np.array(dataFeature), dataLabel, 5)
accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, 2.0, 0.0625)

print(accu1, accu2)



