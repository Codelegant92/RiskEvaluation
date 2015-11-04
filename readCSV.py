__author__ = 'yangchen'

import csv
from commonFunction import *
import numpy as np

from decisionTree import decision_Tree
from svm_classification import svmclassifier, baggingSVM
from ensemble import adboostDT, bagging_adboostDT
from regression import logistic_regression
from KNN import knn

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
    return(np.array(richNess), np.array(testLabel))

dataFeature, dataLabel = generateRichnessDataset()

featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, 5)

#knn
#accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, knn, 1)
#logistic regression
#accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
#decision tree
#accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
#adboost decision tree
#accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
#bagging adboost decision tree
accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 50, 1.0)
#svm
#accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, 2.0, 0.0625)
#bagging svm
#accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, baggingSVM, 2.0, 0.0625)

print(accu1, accu2, (accu1+accu2)/2)


'''
t = gainRatio(dataFeature, dataLabel)
ratioDict = []
for i in range(26):
    ratioDict.append((i, t[i]))
print(ratioDict)
ratioDict = sorted(ratioDict, key = lambda x : -x[1])
print(ratioDict)
'''
