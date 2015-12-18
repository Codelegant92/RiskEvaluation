from commonFunction import *
from featureSupplement import *
import numpy as np
from sklearn import linear_model

from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer, GBDT
from svm_classification import svmclassifier, baggingSVM, svm_GridSearch_creditScore
from regression import logistic_regression, bagging_LR, bagging_twoLayer_LR, Ad_LR
from KNN import knn, bagging_KNN
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

from classifierComparison import bagging_classifierComparison



if(__name__ == "__main__"):
    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 10, 0.6, 15, 0.6, 5, 0.6, 2)

    '''
    clf = linear_model.LogisticRegression(dual=False, class_weight='auto')
    clf.fit(trainFeature, trainLabel)
    predictedProbList = clf.predict_proba(testFeature)
    predictedProbArray = [item[0] for item in clf.predict_proba(testFeature)[:, 1:]]
    result = list(predictedProbArray)
    resultTuple = sorted(zip(testPlatform, result), key = lambda x: x[0])
    '''

    riskList = []
    for ii in range(1000):
        #bagging Logistic Regression
        folderNum = 23
        #print(len(testFeature))
        predictedProbList = []
        posNum = list(trainLabel).count(1)
        negNum = list(trainLabel).count(0)
        print("positive number: %d, negative number: %d") % (posNum, negNum)
        trainFeature = list(trainFeature)
        posFeature = []
        negFeature = []
        for i in range(trainLabel.shape[0]):
            try:
                if(trainLabel[i] == 0):
                    negFeature.append(trainFeature[i])
                else:
                    posFeature.append(trainFeature[i])
            except ValueError, e:
                print("error", e, "on line", i)

        if(posNum < negNum):
            negFeatureFolders = []
            sequence = range(negNum)
            for i in range(folderNum):
                random.shuffle(sequence)
                negFeatureFolders.append([negFeature[j] for j in sequence[:posNum]])
                #print('random sequence:')
                #print(sequence[:posNum])

        #print(np.array(negFeatureFolders).shape)
        for i in range(folderNum):
            subTrainFeature = negFeatureFolders[i]
            subTrainFeature.extend(posFeature)
            subTrainFeature = np.array(subTrainFeature)
            subTrainLabel = list(np.zeros(posNum))
            subTrainLabel.extend(list(np.ones(posNum)))
            subTrainLabel = np.array(subTrainLabel)
            print("=====%dst Bagging=====") % (i+1)
            print("Positive: %d, Negative: %d") % (list(subTrainLabel).count(1), list(subTrainLabel).count(0))
            #print(subTrainFeature.shape)
            #print(subTrainLabel)
            clf = linear_model.LogisticRegression(penalty='l2', dual=False, class_weight='auto')
            clf.fit(subTrainFeature, subTrainLabel)
            predictedProb_temp = [item[0] for item in clf.predict_proba(testFeature)[:, 1:]]
            predictedProbList.append(predictedProb_temp)
            print("%dst predicted probability:") % (i+1)
            print(predictedProb_temp)
            print(clf.predict(testFeature))
        predictedProbArray = np.array(predictedProbList)
        result = list(np.mean(predictedProbArray, axis = 0))
        resultTuple = sorted(zip(testPlatform, result), key = lambda x: x[0])
        temp = []
        for i in range(len(testPlatform)):
            temp.append(1-resultTuple[i][1])
        riskList.append(temp)

    riskList = np.array(riskList)
    answer = np.mean(riskList, axis=0)

    print("===platform==================risk=========")
    for i in range(len(testPlatform)):
        #print("     %d               %f") % (resultTuple[i][0], 1-resultTuple[i][1])
        print("     %d               %f") % (resultTuple[i][0], answer[i])