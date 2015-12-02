import csv
from commonFunction import *
import numpy as np

from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer
from svm_classification import svmclassifier, baggingSVM, svm_GridSearch_creditScore
from regression import logistic_regression, bagging_LR, bagging_twoLayer_LR
from KNN import knn, bagging_KNN
from classifierComparison import bagging_classifierComparison


def readFeatureCSV():
    with open('feature/feature.csv') as f:
        Feature = []
        csvreader = csv.reader(f)
        i = 0
        for rows in csvreader:
            if(i == 0 or i == 8 or i == 10 or i == 11 or i == 13 or i == 14 or i == 22 or i == 38 or i == 45 or i == 46):
                i += 1
                continue
            elif(i < 51):
                try:
                    Feature.append([float(item) for item in rows])
                except ValueError, e:
                    print("error", e, "on line", i)
                i += 1
            else:
                break
        f.close()
    Feature = np.array(Feature)[:, :56]
    Feature = (Feature - np.min(Feature, axis = 0))/ (np.max(Feature, axis = 0) - np.min(Feature, axis = 0))
    print(Feature)
    Label = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    return(Feature, Label)

if(__name__ == "__main__"):
    feature, label = readFeatureCSV()
    featureFolder, labelFolder = crossValidation(feature, label, 2)
    #crossValidationFunc(featureFolder, labelFolder, bagging_classifierComparison)


    #knn
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, knn, 5)
    #bagging_KNN
    accu11, accu21 = crossValidationFunc(featureFolder, labelFolder, bagging_KNN, 5)

    #logistic regression
    accu12, accu22 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)

    #bagging LR
    accu13, accu23 = crossValidationFunc(featureFolder, labelFolder, bagging_LR)

    #bagging_twoLayer_LR
    accu1a, accu2a = crossValidationFunc(featureFolder, labelFolder, bagging_twoLayer_LR)


    #decision tree
    accu14, accu24 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    accu15, accu25 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
    #bagging adboost decision tree
    accu16, accu26 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 50, 1.0)
    accu17, accu27 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)

    #svm
    accu18, accu28 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, 1.0, 0.015625)
    #bagging svm
    accu19, accu29 = crossValidationFunc(featureFolder, labelFolder, baggingSVM, 1.0, 0.015625)


    print("knn: accu1-%f, accu2-%f, average accu_%f") % (accu1, accu2, (accu1+accu2)/2)
    print("bagging knn: accu1-%f, accu2-%f, average accu_%f") % (accu11, accu21, (accu11+accu21)/2)

    print("svm: accu1-%f, accu2-%f, average accu_%f") % (accu18, accu28, (accu18+accu28)/2)

    print("bagging svm: accu1-%f, accu2-%f, average accu_%f") % (accu19, accu29, (accu19+accu29)/2)
    print("DT: accu1-%f, accu2-%f, average accu_%f") % (accu14, accu24, (accu14+accu24)/2)
    print("AdaDT: accu1-%f, accu2-%f, average accu_%f") % (accu15, accu25, (accu15+accu25)/2)
    print("bagging AdaDT: accu1-%f, accu2-%f, average accu_%f") % (accu16, accu26, (accu16+accu26)/2)
    print("Random Forest: accu1-%f, accu2-%f, average accu_%f") % (accu17, accu27, (accu17+accu27)/2)


    print("LR: accu1-%f, accu2-%f, average accu_%f") % (accu12, accu22, (accu12+accu22)/2)
    print("bagging LR: accu1-%f, accu2-%f, average accu_%f") % (accu13, accu23, (accu13+accu23)/2)
    print("bagging two_layer LR: accu1-%f, accu2-%f, average accu_%f") % (accu1a, accu2a, (accu1a+accu2a)/2)
