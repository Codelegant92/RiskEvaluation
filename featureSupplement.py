import csv
from commonFunction import *
import numpy as np
from sklearn import linear_model

from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer, GBDT
from svm_classification import svmclassifier, baggingSVM, svm_GridSearch_creditScore
from regression import logistic_regression, bagging_LR, bagging_twoLayer_LR, Ad_LR
from KNN import knn, bagging_KNN
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

from classifierComparison import bagging_classifierComparison

def readFeature(para_k):
    with open('feature/trainingRichness_clean.csv') as f:
        Feature = []
        csvreader = csv.reader(f)
        for rows in csvreader:
            try:
                Feature.append([float(item) for item in rows])
            except ValueError, e:
                print("error", e, "on line", i)
        f.close()
    Feature = np.array(Feature)[:, :]
    Feature = (Feature - np.min(Feature, axis = 0))/ (np.max(Feature, axis = 0) - np.min(Feature, axis = 0))
    trainFeature = Feature[:50, :]
    testFeature = Feature[50:, :]
    #print(Feature)
    trainLabel = [1 for i in range(20)]
    trainLabel.extend([0 for i in range(30)])
    trainLabel = np.array(trainLabel)
    #trainLabel = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    #Feature = SelectKBest(chi2, k=para_k).fit_transform(Feature, Label)
    #Feature = PCA(n_components=para_k).fit_transform(Feature, Label)
    return(trainFeature, trainLabel, testFeature)

if(__name__ == "__main__"):
    trainFeature, trainLabel, testFeature = readFeature(59)
    folderNum = 21

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
            print('random sequence:')
            print(sequence[:posNum])

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
        clf = linear_model.LogisticRegression(penalty='l2', dual=False)
        clf.fit(subTrainFeature, subTrainLabel)
        predictedProb_temp = [item[0] for item in clf.predict_proba(testFeature)[:, 1:]]
        predictedProbList.append(predictedProb_temp)
        print("%dst predicted probability:") % (i+1)
        print(predictedProb_temp)
        print(clf.predict(testFeature))
    predictedProbArray = np.array(predictedProbList)
    print(np.mean(predictedProbArray, axis = 0))