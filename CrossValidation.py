#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@Time      : 4/22/16 10:30 AM
#@Author    : Robin
#@File      : CrossValidation.py
#@Software  : PyCharm

from commonFunction import *
from featureSupplement import *
from decisionTree import decision_Tree, adboostDT, bagging_DT, RandomForest_Classifer, GBDT
from regression import logistic_regression, bagging_LR
from svm_classification import baggingSVM, svmclassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if(__name__ == "__main__"):
    '''
    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 5, 0.6, 20, 0.6, 5, 0.6, 20)
    #featureFolder, labelFolder = crossValidation(trainFeature, trainLabel, 4)
    #crossValidationFunc(featureFolder, labelFolder, bagging_classifierComparison)

    randomFolders = [[27, 8, 31, 3, 43, 25, 11, 2, 4, 45, 28, 30, 21, 24, 17, 26], [15, 33, 19, 18, 35, 37, 5, 44, 29, 32, 38, 42, 46, 20, 13, 1], [41, 0, 10, 39, 48, 23, 47, 6, 9, 22, 7, 34, 40, 12, 16, 14, 49, 36]]
    featureFolder = [np.array([list(list(trainFeature)[j]) for j in folderList]) for folderList in randomFolders]
    labelFolder = [np.array([list(trainLabel)[k] for k in folderList]) for folderList in randomFolders]
    '''

    old_accuracy = 0
    para = []

    x = 0
    for freq_i in range(5, 6):
        for life_i in range(10, 11):
            for amount_i in range(15, 16):
                for rate_i in range(5, 6):
                    for neighbor_i in range(1, 2):

                            x += 1
                            trainFeature, trainLabel, testFeature, testPlatform = readFeature(freq_i, 0.5, life_i, 0.6, amount_i, 0.6, rate_i, 0.6, neighbor_i)
                            testLabel = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                            AllFeature = np.concatenate([trainFeature, testFeature])[:, :]
                            AllLabel = np.concatenate((trainLabel, testLabel))
                            TruePositive = []
                            TrueNegative = []
                            #3-folder cross validation
                            #randomFolders = [[27, 8, 31, 3, 43, 25, 11, 2, 4, 45, 28, 30, 21, 24, 17, 26], [15, 33, 19, 18, 35, 37, 5, 44, 29, 32, 38, 42, 46, 20, 13, 1], [41, 0, 10, 39, 48, 23, 47, 6, 9, 22, 7, 34, 40, 12, 16, 14, 49, 36]]
                            #featureFolder = [np.array([list(list(AllFeature)[j]) for j in folderList]) for folderList in randomFolders]
                            #labelFolder = [np.array([list(AllLabel)[k] for k in folderList]) for folderList in randomFolders]

                            featureFolder, labelFolder = crossValidation(AllFeature, AllLabel, 3)
                            #logistic regression
                            for baggingSize in range(39, 139, 2):
                                accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, bagging_LR, baggingSize)
                                TruePositive.append(accu1)
                                TrueNegative.append(accu2)

                            print(TruePositive)
                            print(TrueNegative)



    '''
    #LR
    accu12, accu22 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
    #bagging LR
    accu13, accu23 = crossValidationFunc(featureFolder, labelFolder, bagging_LR, 11)

    #bagging_twoLayer_LR
    accu1a, accu2a = crossValidationFunc(featureFolder, labelFolder, bagging_twoLayer_LR, 11)

    #decision tree
    accu14, accu24 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    accu15, accu25 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
    #bagging adboost decision tree
    accu16, accu26 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 11, 50, 1.0)
    accu17, accu27 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)
    accu18, accu28 = crossValidationFunc(featureFolder, labelFolder, GBDT)

    print("==================================different classifiers================================")
    print("classifier               type1 accuracy      type2 accuracy      average accuracy")
    print("    DT                       %f            %f            %f") % (accu14, accu24, (accu14+accu24)/2)
    print(" AdaDT                       %f            %f            %f") % (accu15, accu25, (accu15+accu25)/2)
    print("Random Forest                %f            %f            %f") % (accu17, accu27, (accu17+accu27)/2)
    print(" GBDT                        %f            %f            %f") % (accu18, accu28, (accu18+accu28)/2)
    print("bagging AdaDT                %f            %f            %f") % (accu16, accu26, (accu16+accu26)/2)
    print("   LR                        %f            %f            %f") % (accu12, accu22, (accu12+accu22)/2)
    print("bagging LR                   %f            %f            %f") % (accu13, accu23, (accu13+accu23)/2)
    print("bagging two_layer LR         %f            %f            %f") % (accu1a, accu2a, (accu1a+accu2a)/2)
    '''
