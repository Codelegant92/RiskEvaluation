#This is for logistic regression
from commonFunction import *
from sklearn import linear_model

def logistic_regression(trainFeature, trainLabel, testFeature):
    clf = linear_model.LogisticRegression(penalty='l2', dual=False)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

def bagging_LR(trainFeature, trainLabel, testFeature):
    folderNum = 9
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
        predictedLabel_temp = logistic_regression(subTrainFeature, subTrainLabel, testFeature)
        predictedLabel_voting.append(predictedLabel_temp)
        print("%dst predicted labels:") % (i+1)
        print(predictedLabel_temp)
    predictedLabel_voting = np.array(predictedLabel_voting).T
    predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
    print(predictedLabel)
    return(predictedLabel)


if(__name__ == "__main__"):
    folderNum = 5
    dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
    #dataFeature, dataLabel = read_GermanData('./Data/german/german.data-numeric')
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
    print(accu1, accu2)