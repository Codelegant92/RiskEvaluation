from commonFunction import *
import numpy as np
import time
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals.six import StringIO
import os
#import pydot


def decision_Tree(trainFeature, trainLabel, testFeature):
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    print("feature importance:")
    print(clf.feature_importances_)
    '''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('./Data/test.pdf')
    '''
    return(predictedLabel)

def RandomForest_Classifer(trainFeature, trainLabel, testFeature):
    clf = RandomForestClassifier()
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    '''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('./Data/test.pdf')
    '''
    return(predictedLabel)

def adboostDT(trainFeature, trainLabel, testFeature, estimatorNum = 50, learningRate = 1.0):
    clf = AdaBoostClassifier(n_estimators = estimatorNum, learning_rate = learningRate)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

def bagging_DT(trainFeature, trainLabel, testFeature, folderNum, estimatorNum = 50, learningRate = 1.0):
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
    posFeatureFolders = []
    if(posNum <= negNum):
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
            predictedLabel_temp = decision_Tree(subTrainFeature, subTrainLabel, testFeature)
            predictedLabel_voting.append(predictedLabel_temp)
            print("%dst predicted labels:") % (i+1)
            print(predictedLabel_temp)
        predictedLabel_voting = np.array(predictedLabel_voting).T
        predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
        print(predictedLabel)
        return(predictedLabel)

    if(posNum > negNum):
        sequence = range(posNum)
        for i in range(folderNum):
            random.shuffle(sequence)
            posFeatureFolders.append([posFeature[j] for j in sequence[:negNum]])

    #print(np.array(negFeatureFolders).shape)

        for i in range(folderNum):
            subTrainFeature = posFeatureFolders[i]
            subTrainFeature.extend(negFeature)
            subTrainFeature = np.array(subTrainFeature)
            subTrainLabel = list(np.zeros(negNum))
            subTrainLabel.extend(list(np.ones(negNum)))
            subTrainLabel = np.array(subTrainLabel)
            print("=====%dst Bagging=====") % (i+1)
            print("Positive: %d, Negative: %d") % (list(subTrainLabel).count(1), list(subTrainLabel).count(0))
        #print(subTrainFeature.shape)
        #print(subTrainLabel)
            predictedLabel_temp = decision_Tree(subTrainFeature, subTrainLabel, testFeature)
        #print("predicted probability:")
        #print(predictedProb)
            predictedLabel_voting.append(predictedLabel_temp)
            print("%dst predicted labels:") % (i+1)
            print(predictedLabel_temp)
        predictedLabel_voting = np.array(predictedLabel_voting).T
        predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
        print(predictedLabel)
        return(predictedLabel)


def GBDT(trainFeature, trainLabel, testFeature):
    clf = GradientBoostingClassifier()
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

if __name__ == "__main__":
    pass