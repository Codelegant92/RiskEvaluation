#This is a series of classification algorithms implementation
#SVM for Classification

from sklearn import svm, grid_search
import numpy as np
import time
from functools import wraps
from commonFunction import *

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % (function.func_name, str(t1-t0)))
        return result
    return function_timer()

#@fn_timer
def svm_GridSearch_creditScore(dataFeature, dataLabel):
    start = time.clock()
    C_s = np.logspace(-5, 12, 18, True, 2)
    gamma_para = np.logspace(-12, 5, 18, True, 2)
    parameters = {'C':C_s, 'gamma':gamma_para}
    svc = svm.SVC(cache_size=600)
    clf = grid_search.GridSearchCV(svc, parameters, cv = 5)
    clf.fit(dataFeature, dataLabel)
    scoreList = clf.grid_scores_[0]
    print(scoreList)
    print('best score:')
    print(clf.best_score_)
    print('best estimator:')
    print(clf.best_estimator_)
    print('best_params:')
    print(clf.best_params_)
    end = time.clock()
    print("Time consuming: %f" % (end-start))
#German data best parameters: C=512, gamma=0.000244140625
#Australian data best parameters: C = 128, gamma = 0.00048828125
#function: SVM algorithm - training and testing data, parameter C and gamma are given, when the output is the predicted class
#def svm_algorithm(trainingFeature, traningLabel, testingFeature, testingLabel, kernelType, paraC, paraGamma):
#     clf = svm.SVC(C = paraC, kernel = kernelType, )

def svmclassifier(trainFeature, trainLabel, testFeature, para_C, para_gamma):
    clf = svm.SVC(C = para_C, gamma = para_gamma)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

def baggingSVM(trainFeature, trainLabel, testFeature, para_C, para_gamma):
    folderNum = 39
    predictedLabel_voting = []
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
    negFeatureFolders = []
    if(posNum < negNum):

        sequence = range(negNum)
        for i in range(folderNum):
            random.shuffle(sequence)
            negFeatureFolders.append([negFeature[j] for j in sequence[:posNum]])
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
        predictedLabel_temp = svmclassifier(subTrainFeature, subTrainLabel, testFeature, para_C, para_gamma)
        predictedLabel_voting.append(predictedLabel_temp)
        print("%dst predicted labels:") % (i+1)
        print(predictedLabel_temp)
    predictedLabel_voting = np.array(predictedLabel_voting).T
    predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
    print(predictedLabel)
    return(predictedLabel)
    '''
    randomFeatureFolder, randomLabelFolder = crossValidation(trainFeature, trainLabel, folderNum)
    print("========bagging SVM========")
    for i in range(folderNum):
        subTrainFeature = []
        subTrainLabel = []
        for j in range(folderNum):
            if(j != i):
                subTrainFeature.extend(list(randomFeatureFolder[j]))
                subTrainLabel.extend(list(randomLabelFolder[j]))
        subTrainFeature = np.array(subTrainFeature)
        subTrainLabel = np.array(subTrainLabel)
        print("=====%dst Bagging=====") % (i+1)
        print("Positive: %d, Negative: %d") % (list(subTrainLabel).count(1), list(subTrainLabel).count(0))
        clf = svm.SVC(C = para_C, gamma = para_gamma)
        clf.fit(subTrainFeature, subTrainLabel)
        predictedLabel_temp = clf.predict(testFeature)
        predictedLabel_voting.append(predictedLabel_temp)
        print("%dst predicted labels:") % (i+1)
        print(predictedLabel_temp)
    predictedLabel_voting = np.array(predictedLabel_voting).T
    predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
    print(predictedLabel)
    return(predictedLabel)
    '''

if(__name__ == "__main__"):
    '''
    folderNum = 5
    para_C = 128
    para_gamma = 0.00048828125
    dataFeature, dataLabel = generateRichnessDataset()
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, para_C, para_gamma)
    print(accu1, accu2)
    '''
    pass