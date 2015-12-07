from mainFunc import *

if(__name__ == "__main__"):
    feature, label = readFeatureCSV()
    #featureFolder, labelFolder = crossValidation(feature, label, 2)
    #crossValidationFunc(featureFolder, labelFolder, bagging_classifierComparison)
    #randomFolders = [[14, 7, 22, 33, 23, 31, 2, 4, 40, 34, 1, 17, 35, 19, 6, 36, 0, 16, 25, 20], [39, 3, 27, 15, 18, 30, 29, 5, 10, 9, 11, 37, 32, 8, 21, 28, 12, 38, 26, 24, 13]]
    randomFolders = [[8, 21, 19, 39, 6, 25, 23, 15, 26, 3, 34, 7, 30], [14, 36, 38, 20, 10, 4, 35, 37, 31, 40, 27, 5, 12], [0, 1, 16, 29, 2, 9, 28, 32, 24, 22, 18, 13, 17, 33, 11]]

    featureFolder = [np.array([list(list(feature)[j]) for j in folderList]) for folderList in randomFolders]
    labelFolder = [np.array([list(label)[k] for k in folderList]) for folderList in randomFolders]
    average1 = []
    average2 = []
    #CV1
    trainFeature = np.concatenate((featureFolder[1], featureFolder[2]))
    testFeature = featureFolder[0]
    trainLabel = list(np.concatenate((labelFolder[1], labelFolder[2])))
    testLabel = list(labelFolder[0])
    print("<=======CV1=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(0))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(0))

    predictedLabel_CV1 = bagging_LR(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
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
    print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    average1.append(1 - (type1_error+type2_error)/2)

    #CV2
    trainFeature = np.concatenate((featureFolder[0], featureFolder[2]))
    testFeature = featureFolder[1]
    trainLabel = list(np.concatenate((labelFolder[0], labelFolder[2])))
    testLabel = list(labelFolder[1])
    print("<=======CV2=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(0))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(0))

    predictedLabel_CV2 = bagging_LR(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
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
    print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    average1.append(1 - (type1_error+type2_error)/2)

    #CV3
    trainFeature = np.concatenate((featureFolder[0], featureFolder[1]))
    testFeature = featureFolder[2]
    trainLabel = list(np.concatenate((labelFolder[0], labelFolder[1])))
    testLabel = list(labelFolder[2])
    print("<=======CV3=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(0))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(0))

    predictedLabel_CV3 = bagging_LR(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
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
    print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    average1.append(1 - (type1_error+type2_error)/2)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #CV1
    trainFeature = np.concatenate((featureFolder[1], featureFolder[2]))
    testFeature = featureFolder[0]
    trainLabel = np.concatenate((labelFolder[1], labelFolder[2]))
    testLabel = labelFolder[0]
    trainLabel = [1 if(item == 1) else -1 for item in trainLabel]
    testLabel = [1 if(item == 1) else -1 for item in testLabel]
    print("<=======CV1=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(-1))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(-1))

    #predictedLabel_CV1 = bagging_LR(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = svmclassifier(trainFeature, np.array(trainLabel), testFeature, 1, 0.015625)
    #predictedLabel_CV1 = knn(trainFeature, np.array(trainLabel), testFeature, 5)
    #predictedLabel_CV1 = decision_Tree(trainFeature, np.array(trainLabel), testFeature)
    predictedLabel_CV1 = adboostDT(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV1 = RandomForest_Classifer(trainFeature, np.array(trainLabel), testFeature)
    diff = np.array(testLabel) - np.array(predictedLabel_CV1)
    type1_error = list(diff).count(2)/float(testLabel.count(1))
    type2_error = list(diff).count(-2)/float(testLabel.count(-1))
    print("classifying result:")
    print("true positive: %d / %d; type1_accuracy: %f") % (testLabel.count(1)-list(diff).count(2), testLabel.count(1), 1-type1_error)
    print("true negative: %d / %d; type2_accuracy: %f") % (testLabel.count(-1)-list(diff).count(-2), testLabel.count(-1), 1-type2_error)
    print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    average2.append(1 - (type1_error+type2_error)/2)

    #CV2
    trainFeature = np.concatenate((featureFolder[0], featureFolder[2]))
    testFeature = featureFolder[1]
    trainLabel = np.concatenate((labelFolder[0], labelFolder[2]))
    testLabel = labelFolder[1]
    trainLabel = [1 if(item == 1) else -1 for item in trainLabel]
    testLabel = [1 if(item == 1) else -1 for item in testLabel]
    print("<=======CV2=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(-1))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(-1))

    #predictedLabel_CV2 = bagging_LR(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = svmclassifier(trainFeature, np.array(trainLabel), testFeature, 1.0, 0.015625)
    #predictedLabel_CV2 = knn(trainFeature, np.array(trainLabel), testFeature, 5)
    #predictedLabel_CV2 = decision_Tree(trainFeature, np.array(trainLabel), testFeature)
    predictedLabel_CV2 = adboostDT(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV2 = RandomForest_Classifer(trainFeature, np.array(trainLabel), testFeature)
    diff = np.array(testLabel) - np.array(predictedLabel_CV2)
    type1_error = list(diff).count(2)/float(testLabel.count(1))
    type2_error = list(diff).count(-2)/float(testLabel.count(-1))
    print("classifying result:")
    print("true positive: %d / %d; type1_accuracy: %f") % (testLabel.count(1)-list(diff).count(2), testLabel.count(1), 1-type1_error)
    print("true negative: %d / %d; type2_accuracy: %f") % (testLabel.count(-1)-list(diff).count(-2), testLabel.count(-1), 1-type2_error)
    print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    average2.append(1 - (type1_error+type2_error)/2)

    #CV3
    trainFeature = np.concatenate((featureFolder[0], featureFolder[1]))
    testFeature = featureFolder[2]
    trainLabel = np.concatenate((labelFolder[0], labelFolder[1]))
    testLabel = labelFolder[2]
    trainLabel = [1 if(item == 1) else -1 for item in trainLabel]
    testLabel = [1 if(item == 1) else -1 for item in testLabel]
    print("<=======CV3=======>")
    print("training sample:")
    print("positive: %d, negative: %d") % (trainLabel.count(1), trainLabel.count(-1))
    print("testing sample:")
    print("positive: %d, negative: %d") % (testLabel.count(1), testLabel.count(-1))

    #predictedLabel_CV3 = bagging_LR(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = logistic_regression(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = svmclassifier(trainFeature, np.array(trainLabel), testFeature, 1.0, 0.015625)
    #predictedLabel_CV3 = knn(trainFeature, np.array(trainLabel), testFeature, 5)
    #predictedLabel_CV3 = decision_Tree(trainFeature, np.array(trainLabel), testFeature)
    predictedLabel_CV3 = adboostDT(trainFeature, np.array(trainLabel), testFeature)
    #predictedLabel_CV3 = RandomForest_Classifer(trainFeature, np.array(trainLabel), testFeature)
    diff = np.array(testLabel) - np.array(predictedLabel_CV3)
    type1_error = list(diff).count(2)/float(testLabel.count(1))
    type2_error = list(diff).count(-2)/float(testLabel.count(-1))
    print("classifying result:")
    print("true positive: %d / %d; type1_accuracy: %f") % (testLabel.count(1)-list(diff).count(2), testLabel.count(1), 1-type1_error)
    print("true negative: %d / %d; type2_accuracy: %f") % (testLabel.count(-1)-list(diff).count(-2), testLabel.count(-1), 1-type2_error)
    print("average accuracy: %f") % (1 - (type1_error+type2_error)/2)
    average2.append(1 - (type1_error+type2_error)/2)

    print(average1, np.mean(np.array(average1)))
    print(average2, np.mean(np.array(average2)))