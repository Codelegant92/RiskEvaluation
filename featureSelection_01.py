#This program defines negative samples as 0 and positive samples as 1

from mainFunc import *
import matplotlib.pyplot as plt

if(__name__ == "__main__"):

    trainFeature, trainLabel, testFeature = readFeature(10, 0.5, 10, 1, 10, 1, 3, 0.6, 20)
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