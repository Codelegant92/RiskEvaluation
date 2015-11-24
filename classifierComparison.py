
from commonFunction import *
import numpy as np
from KNN import knn
from decisionTree import decision_Tree, adboostDT, RandomForest_Classifer
from svm_classification import svmclassifier
from regression import logistic_regression

def bagging_classifierComparison(trainFeature, trainLabel, testFeature):
    folderNum = 5
    predictedLabel_voting1 = []
    predictedLabel_voting2 = []
    predictedLabel_voting3 = []
    predictedLabel_voting4 = []
    predictedLabel_voting5 = []
    predictedLabel_voting6 = []
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
        predictedLabel_temp1 = knn(subTrainFeature, subTrainLabel, testFeature, 5)
        predictedLabel_temp2 = decision_Tree(subTrainFeature, subTrainLabel, testFeature)
        predictedLabel_temp3 = adboostDT(subTrainFeature, subTrainLabel, testFeature)
        predictedLabel_temp4 = RandomForest_Classifer(subTrainFeature, subTrainLabel, testFeature)
        predictedLabel_temp5 = svmclassifier(subTrainFeature, subTrainLabel, testFeature, 1.0, 0.015625)
        predictedLabel_temp6 = logistic_regression(subTrainFeature, subTrainLabel, testFeature)

        predictedLabel_voting1.append(predictedLabel_temp1)
        predictedLabel_voting2.append(predictedLabel_temp2)
        predictedLabel_voting3.append(predictedLabel_temp3)
        predictedLabel_voting4.append(predictedLabel_temp4)
        predictedLabel_voting5.append(predictedLabel_temp5)
        predictedLabel_voting6.append(predictedLabel_temp6)
        print("KNN=====%dst predicted labels:") % (i+1)
        print(predictedLabel_temp1)
        print("DT=====%dst predicted labels:") % (i+1)
        print(predictedLabel_temp2)
        print("AdaDT=====%dst predicted labels:") % (i+1)
        print(predictedLabel_temp3)
        print("RandomForest=====%dst predicted labels:") % (i+1)
        print(predictedLabel_temp4)
        print("SVM=====%dst predicted labels:") % (i+1)
        print(predictedLabel_temp5)
        print("LR=====%dst predicted labels:") % (i+1)
        print(predictedLabel_temp6)

    predictedLabel_voting1 = np.array(predictedLabel_voting1).T
    predictedLabel_voting2 = np.array(predictedLabel_voting2).T
    predictedLabel_voting3 = np.array(predictedLabel_voting3).T
    predictedLabel_voting4 = np.array(predictedLabel_voting4).T
    predictedLabel_voting5 = np.array(predictedLabel_voting5).T
    predictedLabel_voting6 = np.array(predictedLabel_voting6).T
    predictedLabel1 = [1 if(list(predictedLabel_voting1[i]).count(1) > list(predictedLabel_voting1[i]).count(0)) else 0 for i in range(predictedLabel_voting1.shape[0])]
    predictedLabel2 = [1 if(list(predictedLabel_voting2[i]).count(1) > list(predictedLabel_voting2[i]).count(0)) else 0 for i in range(predictedLabel_voting2.shape[0])]
    predictedLabel3 = [1 if(list(predictedLabel_voting3[i]).count(1) > list(predictedLabel_voting3[i]).count(0)) else 0 for i in range(predictedLabel_voting3.shape[0])]
    predictedLabel4 = [1 if(list(predictedLabel_voting4[i]).count(1) > list(predictedLabel_voting4[i]).count(0)) else 0 for i in range(predictedLabel_voting4.shape[0])]
    predictedLabel5 = [1 if(list(predictedLabel_voting5[i]).count(1) > list(predictedLabel_voting5[i]).count(0)) else 0 for i in range(predictedLabel_voting5.shape[0])]
    predictedLabel6 = [1 if(list(predictedLabel_voting6[i]).count(1) > list(predictedLabel_voting6[i]).count(0)) else 0 for i in range(predictedLabel_voting6.shape[0])]
    print(predictedLabel1)
    print(predictedLabel2)
    print(predictedLabel3)
    print(predictedLabel4)
    print(predictedLabel5)
    print(predictedLabel6)
    return([predictedLabel1, predictedLabel2, predictedLabel3, predictedLabel4, predictedLabel5, predictedLabel6])