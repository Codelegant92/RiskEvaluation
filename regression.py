#This is for logistic regression
from commonFunction import *
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesRegressor
#from unbalanced_dataset.under_sampling import UnderSampler
#from unbalanced_dataset.unbalanced_dataset import UnbalancedDataset

def logistic_regression(trainFeature, trainLabel, testFeature):
    clf = linear_model.LogisticRegression(class_weight='auto')
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    predictedProb = clf.predict_proba(testFeature)
    return(predictedLabel)

def bagging_LR(trainFeature, trainLabel, testFeature, folderNum):
    predictedLabel_voting = []
    negFeatureFolders = []
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
        predictedLabel_temp = logistic_regression(subTrainFeature, subTrainLabel, testFeature)
        #print("predicted probability:")
        #print(predictedProb)
        predictedLabel_voting.append(predictedLabel_temp)
        print("%dst predicted labels:") % (i+1)
        print(predictedLabel_temp)
    predictedLabel_voting = np.array(predictedLabel_voting).T
    predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
    print(predictedLabel)
    return(predictedLabel)

def bagging_LR1(trainFeature, trainLabel, testFeature, folderNum):
    predictedLabel_voting = []

    for i in range(folderNum):
        print("Random under sampling")
        US = UnderSampler(verbose=True)
        subTrainFeature, subTrainLabel = US.fit(trainFeature, trainLabel)
        print("=====%dst Bagging=====") % (i+1)
        print("Positive: %d, Negative: %d") % (list(subTrainLabel).count(1), list(subTrainLabel).count(0))
        #print(subTrainFeature.shape)
        #print(subTrainLabel)
        predictedLabel_temp = logistic_regression(subTrainFeature, subTrainLabel, testFeature)
        #print("predicted probability:")
        #print(predictedProb)
        predictedLabel_voting.append(predictedLabel_temp)
        print("%dst predicted labels:") % (i+1)
        print(predictedLabel_temp)
    predictedLabel_voting = np.array(predictedLabel_voting).T
    predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
    print(predictedLabel)
    return(predictedLabel)

def bagging_twoLayer_LR(trainFeature, trainLabel, testFeature, folderNum=5):
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
        clf = linear_model.LogisticRegression(penalty='l2', dual=False, class_weight='auto')
        clf.fit(subTrainFeature, subTrainLabel)
        predictedTrainProb = clf.predict_proba(trainFeature)
        predictedTestProb = clf.predict_proba(testFeature)
        for item in predictedTrainProb:
            newTrainFeature_temp.append(item[1])
        for item in predictedTestProb:
            newTestFeature_temp.append(item[1])
        newTrainFeature.append(newTrainFeature_temp)
        newTestFeature.append(newTestFeature_temp)
    newTrainFeature = np.array(newTrainFeature).T
    newTestFeature = np.array(newTestFeature).T
    clf.fit(newTrainFeature, trainLabel)
    predictedLabel = clf.predict(newTestFeature)

    print(predictedLabel)
    return(predictedLabel)

def Ad_LR(trainFeature, trainLabel, testFeature):
    temp = []
    for i in range(1000):
        clf = ExtraTreesRegressor()
        clf.fit(trainFeature, trainLabel)
        predictedLabel = clf.predict(testFeature)
        temp.append()
    temp = np.array(temp)
    label = np.mean(temp, axis=0)
    predictLabel = []
    for i in range(label.shape[0]):
        if(label[i] >= 0.5):
            predictLabel.append(1)
        else:
            predictLabel.append(0)
    return(predictLabel)

def Extra(trainFeature, trainLabel, testFeature):
    clf = AdaBoostRegressor(linear_model.LogisticRegression(penalty='l2', dual=False))
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

if(__name__ == "__main__"):
    pass