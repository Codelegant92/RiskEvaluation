from commonFunction import *
from featureSupplement import *
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn import tree
from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer, GBDT
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from regression import bagging_LR, logistic_regression
from svm_classification import svmclassifier, baggingSVM

def bagging_twoLayer_LR1(trainFeature, trainLabel, testFeature, folderNum=5):
    newTrainFeature = []
    newTestFeature = []
    posNum = list(trainLabel).count(1)
    negNum = list(trainLabel).count(0)
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
    #print(np.array(negFeatureFolders).shape)
    for i in range(folderNum):
        newTrainFeature_temp = []
        newTestFeature_temp = []
        subTrainFeature = negFeatureFolders[i]
        subTrainFeature.extend(posFeature)
        subTrainFeature = np.array(subTrainFeature)
        subTrainLabel = list(np.zeros(posNum))
        subTrainLabel.extend(list(np.ones(posNum)))
        subTrainLabel = np.array(subTrainLabel)
        print("=====%dst Bagging=====") % (i+1)
        print("Positive: %d, Negative: %d") % (list(subTrainLabel).count(1), list(subTrainLabel).count(0))

        clf = AdaBoostRegressor()
        clf.fit(subTrainFeature, subTrainLabel)
        predictedTrainProb = clf.predict(trainFeature)
        predictedTestProb = clf.predict(testFeature)

        for item in predictedTrainProb:
            newTrainFeature_temp.append(item)
        for item in predictedTestProb:
            newTestFeature_temp.append(item)
        newTrainFeature.append(newTrainFeature_temp)
        newTestFeature.append(newTestFeature_temp)

    newTrainFeature = np.array(newTrainFeature).T
    newTestFeature = np.array(newTestFeature).T
    clf = linear_model.LogisticRegression(penalty='l2', dual=False, class_weight='auto')
    clf.fit(newTrainFeature, trainLabel)
    predictedLabel = clf.predict_proba(newTestFeature)
    return(predictedLabel[:, 0])

if(__name__ == "__main__"):
    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 10, 0.6, 15, 0.6, 5, 0.6, 1)
    '''
    selectFeature = SelectKBest(chi2, k = 55)
    selectFeature.fit(trainFeature, trainLabel)
    trainFeature_new = selectFeature.transform(trainFeature)
    testFeature_new = selectFeature.transform(testFeature)
    '''
    trainFeature_new = trainFeature[:, :]
    testFeature_new = testFeature[:, :]
    '''
    trainFeature_new = trainFeature[:, :26]
    testFeature_new = testFeature[:, :26]
    '''
    '''
    pca = PCA(n_components=61)
    pca.fit(np.concatenate([trainFeature, testFeature]))
    trainFeature_new = pca.transform(trainFeature)
    testFeature_new = pca.transform(testFeature)
    '''

    riskList = []
    for i in range(1):
        riskList.append(bagging_LR(trainFeature_new, trainLabel, testFeature_new, 51))
        #riskList.append(baggingSVM(trainFeature_new, trainLabel, testFeature_new, 128, 0.00048828125))
    riskList = np.array(riskList)
    answer = np.mean(riskList, axis=0)

    resultTuple = sorted(zip(testPlatform, answer), key = lambda x: x[0])
    result = []
    print("===platform==================risk=========")
    for i in range(len(testPlatform)):
        print("     %d               %f") % (resultTuple[i][0], resultTuple[i][1])
        result.append(resultTuple[i][1])
    goodResult = result[:10]
    badResult = result[10:]
    print("bad2good: %d")%(goodResult.count(0))
    print("good2bad: %d")%(badResult.count(1))

    '''
    para = []
    for i in range(1000):
        clf = ExtraTreesRegressor()
        clf.fit(trainFeature_new, trainLabel)
        predictedLabel = clf.predict(testFeature_new)
        para.append(predictedLabel)
    para = np.array(para)
    predictedLabel = np.mean(para, axis = 0)
    resultTuple = sorted(zip(testPlatform, predictedLabel), key = lambda x: x[0])

    print("===platform==================risk=========")
    for i in range(len(testPlatform)):
        #print("     %d               %f") % (resultTuple[i][0], 1-resultTuple[i][1])
        print("     %d               %f") % (resultTuple[i][0], 1 - resultTuple[i][1])
    '''

    '''
    clf = linear_model.LogisticRegression(dual=False, class_weight='auto')
    clf.fit(trainFeature_new, trainLabel)
    predictedProbList = clf.predict_proba(testFeature_new)
    predictedProbArray = [item[0] for item in clf.predict_proba(testFeature_new)[:, 1:]]
    result = list(predictedProbArray)
    resultTuple = sorted(zip(testPlatform, result), key = lambda x: x[0])
    print("===platform==================risk=========")
    for i in range(len(testPlatform)):
        #print("     %d               %f") % (resultTuple[i][0], 1-resultTuple[i][1])
        print("     %d               %f") % (resultTuple[i][0], 1-resultTuple[i][1])
    '''

    '''
    riskList = []
    for ii in range(100):
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
                    negFeature.append(trainFeature_new[i])
                else:
                    posFeature.append(trainFeature_new[i])
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
            temp = []
            for x in range(10):
                clf = ExtraTreesRegressor()
                clf.fit(subTrainFeature, subTrainLabel)
                predictedLabel = clf.predict(testFeature_new)
                temp.append(predictedLabel)
            temp = np.array(temp)
            temp = np.mean(temp, axis=0)
            predictedProb_temp = [item for item in temp]
            predictedProbList.append(predictedProb_temp)
            print("%dst predicted probability:") % (i+1)
            print(predictedProb_temp)
            print(clf.predict(testFeature_new))
        predictedProbArray = np.array(predictedProbList)
        result = list(np.mean(predictedProbArray, axis = 0))
        resultTuple = sorted(zip(testPlatform, result), key = lambda x: x[0])
        temp = []
        for i in range(len(testPlatform)):
            temp.append(1-resultTuple[i][1])
        riskList.append(temp)
    riskList = np.array(riskList)
    answer = np.mean(riskList, axis=0)
    print(trainFeature_new.shape)
    print(testFeature_new.shape)
    print("===platform==================risk=========")
    for i in range(len(testPlatform)):
        #print("     %d               %f") % (resultTuple[i][0], 1-resultTuple[i][1])
        print("     %d               %f") % (resultTuple[i][0], answer[i])
    '''


