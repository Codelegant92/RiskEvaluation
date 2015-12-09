from commonFunction import *
from featureSupplement import *
from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer, GBDT
from svm_classification import svmclassifier, baggingSVM, svm_GridSearch_creditScore
from regression import logistic_regression, bagging_LR, bagging_twoLayer_LR, Ad_LR
from KNN import knn, bagging_KNN
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

from classifierComparison import bagging_classifierComparison



if(__name__ == "__main__"):
    trainFeature, trainLabel, testFeature = readFeature()
    #featureFolder, labelFolder = crossValidation(trainFeature, trainLabel, 3)
    randomFolders = [[15, 13, 21, 39, 34, 24, 46, 41, 0, 49, 2, 31, 29, 12, 32, 9], [22, 35, 10, 26, 1, 6, 48, 30, 8, 28, 16, 23, 42, 7, 25, 14], [45, 3, 19, 17, 44, 18, 20, 11, 36, 33, 27, 38, 4, 47, 5, 37, 43, 40]]
    featureFolder = [np.array([list(list(trainFeature)[j]) for j in folderList]) for folderList in randomFolders]
    labelFolder = [np.array([list(trainLabel)[k] for k in folderList]) for folderList in randomFolders]

    '''
    #knn
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, knn, 5, 9)
    #bagging_KNN
    accu11, accu21 = crossValidationFunc(featureFolder, labelFolder, bagging_KNN, 5, 9)
    '''
    #logistic regression
    accu12, accu22 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)

    #bagging LR
    accu13, accu23 = crossValidationFunc(featureFolder, labelFolder, bagging_LR, 11)

    #bagging_twoLayer_LR
    accu1a, accu2a = crossValidationFunc(featureFolder, labelFolder, bagging_twoLayer_LR, 11)

    '''
    #decision tree
    accu14, accu24 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    accu15, accu25 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
    #bagging adboost decision tree
    accu16, accu26 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 9, 50, 1.0)
    accu17, accu27 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)

    #svm
    accu18, accu28 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, 1.0, 0.015625)
    #bagging svm
    accu19, accu29 = crossValidationFunc(featureFolder, labelFolder, baggingSVM, 1.0, 0.015625)


    print("knn: accu1-%f, accu2-%f, average accu_%f") % (accu1, accu2, (accu1+accu2)/2)
    print("bagging knn: accu1-%f, accu2-%f, average accu_%f") % (accu11, accu21, (accu11+accu21)/2)

    print("svm: accu1-%f, accu2-%f, average accu_%f") % (accu18, accu28, (accu18+accu28)/2)

    print("bagging svm: accu1-%f, accu2-%f, average accu_%f") % (accu19, accu29, (accu19+accu29)/2)
    print("DT: accu1-%f, accu2-%f, average accu_%f") % (accu14, accu24, (accu14+accu24)/2)
    print("AdaDT: accu1-%f, accu2-%f, average accu_%f") % (accu15, accu25, (accu15+accu25)/2)
    print("bagging AdaDT: accu1-%f, accu2-%f, average accu_%f") % (accu16, accu26, (accu16+accu26)/2)
    print("Random Forest: accu1-%f, accu2-%f, average accu_%f") % (accu17, accu27, (accu17+accu27)/2)

    '''
    print("LR: accu1-%f, accu2-%f, average accu_%f") % (accu12, accu22, (accu12+accu22)/2)
    print("bagging LR: accu1-%f, accu2-%f, average accu_%f") % (accu13, accu23, (accu13+accu23)/2)
    print("bagging two_layer LR: accu1-%f, accu2-%f, average accu_%f") % (accu1a, accu2a, (accu1a+accu2a)/2)
