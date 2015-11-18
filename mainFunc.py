import csv
from commonFunction import *
import numpy as np

from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer
from svm_classification import svmclassifier, baggingSVM
from regression import logistic_regression
from KNN import knn

def readFeatureCSV():
    with open('feature.csv') as f:
        Feature = []
        csvreader = csv.reader(f)
        i = 0
        for rows in csvreader:
            if(i == 0 or i == 8 or i == 10 or i == 11 or i == 14 or i == 38 or i == 45 or i == 46):
                i += 1
                continue
            elif(i < 51):
                print(i)
                try:
                    Feature.append([float(item) for item in rows])
                except ValueError, e:
                    print("error", e, "on line", i)
                i += 1
            else:
                break
        f.close()
    Feature = np.array(Feature)
    Label = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    return(Feature, Label)

if(__name__ == "__main__"):
    feature, label = readFeatureCSV()
    featureFolder, labelFolder = crossValidation(feature, label, 5)

    #logistic regression
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
    #decision tree
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
    #bagging adboost decision tree
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 50, 1.0)
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)
    #svm
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, 2.0, 0.0625)
    #bagging svm
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, baggingSVM, 2.0, 0.0625)
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, knn, 1)
    print(accu1, accu2, (accu1+accu2)/2)