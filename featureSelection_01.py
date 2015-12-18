#This program defines negative samples as 0 and positive samples as 1

from mainFunc import *
import matplotlib.pyplot as plt
from sklearn import linear_model


def bagging_LR_prob(trainFeature, trainLabel, testFeature):
    folderNum = 9
    print(len(testFeature))
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
        clf = linear_model.LogisticRegression(penalty='l2', dual=False)
        clf.fit(subTrainFeature, subTrainLabel)
        predictedProb_temp = [item[0] for item in clf.predict_proba(testFeature)[:, 1:]]
        predictedProbList.append(predictedProb_temp)
        print("%dst predicted probability:") % (i+1)
        print(predictedProb_temp)
        print(clf.predict(testFeature))
    predictedProbArray = np.array(predictedProbList)
    result = list(np.mean(predictedProbArray, axis = 0))
    return(result)


if(__name__ == "__main__"):

    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 5, 1, 20, 1, 5, 0.6, 20)

    #featureFolder, labelFolder = crossValidation(trainFeature, trainLabel, 3)
    randomFolders = [[15, 13, 21, 39, 34, 24, 46, 41, 0, 49, 2, 31, 29, 12, 32, 9], [22, 35, 10, 26, 1, 6, 48, 30, 8, 28, 16, 23, 42, 7, 25, 14], [45, 3, 19, 17, 44, 18, 20, 11, 36, 33, 27, 38, 4, 47, 5, 37, 43, 40]]
    featureFolder = [np.array([list(list(trainFeature)[j]) for j in folderList]) for folderList in randomFolders]
    labelFolder = [np.array([list(trainLabel)[k] for k in folderList]) for folderList in randomFolders]

    average = []
    accuracy = []
    #CV1
    trainFeature = np.concatenate((featureFolder[1], featureFolder[2]))
    testFeature = featureFolder[0]
    trainLabel = list(np.concatenate((labelFolder[1], labelFolder[2])))
    testLabel = list(labelFolder[0])
    #test = gainRatio(trainFeature, trainLabel)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("<=======CV1=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(0))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(0))
    print("test label:")
    print(testLabel)


    '''
    predictedProb1 = bagging_LR_prob(trainFeature, np.array(trainLabel), testFeature)
    print("test label1:")
    print(testLabel)
    print("result1:")
    print(predictedProb1)
    '''
    #predictedLabel_CV1 = bagging_LR(trainFeature, np.array(trainLabel), testFeature, 11)
    #predictedLabel_CV1 = bagging_twoLayer_LR(trainFeature, np.array(trainLabel), testFeature, 9)
    predictedLabel_CV1 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = svmclassifier(trainFeature, np.array(trainLabel), testFeature, 1, 0.015625)
    #predictedLabel_CV1 = knn(trainFeature, np.array(trainLabel), testFeature, 5)
    #predictedLabel_CV1 = decision_Tree(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = adboostDT(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = RandomForest_Classifer(trainFeature, np.array(trainLabel), testFeature)

    diff = np.array(testLabel) - np.array(predictedLabel_CV1)
    type1_error = list(diff).count(1)/float(testLabel.count(1))
    type2_error = list(diff).count(-1)/float(testLabel.count(0))
    print("classifying result:")
    print("true positive: %d / %d; type1_accuracy: %f") % (testLabel.count(1)-list(diff).count(1), testLabel.count(1), 1-type1_error)
    print("true negative: %d / %d; type2_accuracy: %f") % (testLabel.count(0)-list(diff).count(-1), testLabel.count(0), 1-type2_error)
    #print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    accuracy.append({"type-1-accuracy" : 1-type1_error, "type-2-accuracy": 1-type2_error})
    average.append(1 - (type1_error+type2_error)/2)

    #CV2
    trainFeature = np.concatenate((featureFolder[0], featureFolder[2]))
    testFeature = featureFolder[1]
    trainLabel = list(np.concatenate((labelFolder[0], labelFolder[2])))
    testLabel = list(labelFolder[1])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("<=======CV2=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(0))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(0))
    print(testLabel)
    #predictedLabel_CV2 = bagging_LR(trainFeature, np.array(trainLabel), testFeature, 11)
    #predictedLabel_CV2 = bagging_twoLayer_LR(trainFeature, np.array(trainLabel), testFeature, 9)
    predictedLabel_CV2 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = svmclassifier(trainFeature, np.array(trainLabel), testFeature, 1, 0.015625)
    #predictedLabel_CV2 = knn(trainFeature, np.array(trainLabel), testFeature, 5)
    #predictedLabel_CV2 = decision_Tree(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = adboostDT(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = RandomForest_Classifer(trainFeature, np.array(trainLabel), testFeature)
    '''
    predictedProb2 = bagging_LR_prob(trainFeature, np.array(trainLabel), testFeature)
    print("test label2:")
    print(testLabel)
    print("result2:")
    print(predictedProb2)
    '''

    diff = np.array(testLabel) - np.array(predictedLabel_CV2)
    type1_error = list(diff).count(1)/float(testLabel.count(1))
    type2_error = list(diff).count(-1)/float(testLabel.count(0))
    print("classifying result:")
    print("true positive: %d / %d; type1_accuracy: %f") % (testLabel.count(1)-list(diff).count(1), testLabel.count(1), 1-type1_error)
    print("true negative: %d / %d; type2_accuracy: %f") % (testLabel.count(0)-list(diff).count(-1), testLabel.count(0), 1-type2_error)
    #print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    accuracy.append({"type-1-accuracy" : 1-type1_error, "type-2-accuracy": 1-type2_error})
    average.append(1 - (type1_error+type2_error)/2)

    #CV3
    trainFeature = np.concatenate((featureFolder[0], featureFolder[1]))
    testFeature = featureFolder[2]
    trainLabel = list(np.concatenate((labelFolder[0], labelFolder[1])))
    testLabel = list(labelFolder[2])
    print("<=======CV3=======>")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(0))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(0))
    print(testLabel)
    #predictedLabel_CV3 = bagging_LR(trainFeature, np.array(trainLabel), testFeature, 11)
    #predictedLabel_CV3 = bagging_twoLayer_LR(trainFeature, np.array(trainLabel), testFeature, 9)
    predictedLabel_CV3 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = svmclassifier(trainFeature, np.array(trainLabel), testFeature, 1, 0.015625)
    #predictedLabel_CV3 = knn(trainFeature, np.array(trainLabel), testFeature, 5)
    #predictedLabel_CV3 = decision_Tree(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = adboostDT(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = RandomForest_Classifer(trainFeature, np.array(trainLabel), testFeature)
    '''
    predictedProb3 = bagging_LR_prob(trainFeature, np.array(trainLabel), testFeature)
    print("test label3:")
    print(testLabel)
    print("result3:")
    print(predictedProb3)
    '''

    diff = np.array(testLabel) - np.array(predictedLabel_CV3)
    type1_error = list(diff).count(1)/float(testLabel.count(1))
    type2_error = list(diff).count(-1)/float(testLabel.count(0))
    print("classifying result:")
    print("true positive: %d / %d; type1_accuracy: %f") % (testLabel.count(1)-list(diff).count(1), testLabel.count(1), 1-type1_error)
    print("true negative: %d / %d; type2_accuracy: %f") % (testLabel.count(0)-list(diff).count(-1), testLabel.count(0), 1-type2_error)
    #print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    accuracy.append({"type-1-accuracy" : 1-type1_error, "type-2-accuracy": 1-type2_error})
    average.append(1 - (type1_error+type2_error)/2)


    print(accuracy)
    print("average accuracy: %f") % (np.mean(np.array(average)))
