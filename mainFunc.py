from commonFunction import *
from featureSupplement import *
from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer, GBDT
from regression import logistic_regression, bagging_LR, bagging_twoLayer_LR, Ad_LR
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

if(__name__ == "__main__"):
    old_accuracy = 0
    para = []
    x = 0
    for freq_i in range(5, 21):
        for life_i in range(5, 21):
            for amount_i in range(5, 21):
                for rate_i in range(3, 10):
                    x += 1
                    trainFeature, trainLabel, testFeature, testPlatform = readFeature(freq_i, 0.5, life_i, 0.6, amount_i, 0.6, rate_i, 0.6, 20)
    #featureFolder, labelFolder = crossValidation(trainFeature, trainLabel, 3)
    #crossValidationFunc(featureFolder, labelFolder, bagging_classifierComparison)

                    randomFolders = [[27, 8, 31, 3, 43, 25, 11, 2, 4, 45, 28, 30, 21, 24, 17, 26], [15, 33, 19, 18, 35, 37, 5, 44, 29, 32, 38, 42, 46, 20, 13, 1], [41, 0, 10, 39, 48, 23, 47, 6, 9, 22, 7, 34, 40, 12, 16, 14, 49, 36]]
                    featureFolder = [np.array([list(list(trainFeature)[j]) for j in folderList]) for folderList in randomFolders]
                    labelFolder = [np.array([list(trainLabel)[k] for k in folderList]) for folderList in randomFolders]

    #logistic regression
                    accu12, accu22 = crossValidationFunc(featureFolder, labelFolder, bagging_LR, 11)
                    new_accuracy = (accu12+accu22)/2.0
                    if(new_accuracy >= old_accuracy):
                        para.append([freq_i, life_i, amount_i, rate_i, new_accuracy])
                        old_accuracy = new_accuracy
    for item in para:
        print(item)
    '''
    #bagging LR
    accu13, accu23 = crossValidationFunc(featureFolder, labelFolder, bagging_LR, 11)

    #bagging_twoLayer_LR
    accu1a, accu2a = crossValidationFunc(featureFolder, labelFolder, bagging_twoLayer_LR, 11)

    #decision tree
    accu14, accu24 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    accu15, accu25 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
    #bagging adboost decision tree
    accu16, accu26 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 11, 50, 1.0)
    accu17, accu27 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)
    accu18, accu28 = crossValidationFunc(featureFolder, labelFolder, GBDT)

    print("==================================different classifiers================================")
    print("classifier               type1 accuracy      type2 accuracy      average accuracy")
    print("    DT                       %f            %f            %f") % (accu14, accu24, (accu14+accu24)/2)
    print(" AdaDT                       %f            %f            %f") % (accu15, accu25, (accu15+accu25)/2)
    print("Random Forest                %f            %f            %f") % (accu17, accu27, (accu17+accu27)/2)
    print(" GBDT                        %f            %f            %f") % (accu18, accu28, (accu18+accu28)/2)
    print("bagging AdaDT                %f            %f            %f") % (accu16, accu26, (accu16+accu26)/2)
    print("   LR                        %f            %f            %f") % (accu12, accu22, (accu12+accu22)/2)
    print("bagging LR                   %f            %f            %f") % (accu13, accu23, (accu13+accu23)/2)
    print("bagging two_layer LR         %f            %f            %f") % (accu1a, accu2a, (accu1a+accu2a)/2)
    '''
