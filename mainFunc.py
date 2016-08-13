from commonFunction import *
from featureSupplement import *
from decisionTree import decision_Tree, adboostDT, bagging_DT, RandomForest_Classifer, GBDT
from regression import logistic_regression, bagging_LR, bagging_twoLayer_LR, Ad_LR
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from KNN import knn, bagging_KNN
from svm_classification import svmclassifier, baggingSVM, svm_GridSearch_creditScore
from NeuralNetwork import MLP

if(__name__ == "__main__"):

    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 10, 0.6, 15, 0.6, 5, 0.6, 20)

    testLabel = [1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 0,0,0,0,0]
    #testLabel = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
    label = list(trainLabel)
    label.extend(testLabel)
    feature1 = np.concatenate([trainFeature, testFeature])[:, :]
    label = np.array(label)
    acc_dt = []
    acc_svm = []
    acc_lr = []
    acc_mlp = []
    bagging_size = 21
    for n in range(5, 62, 5):
        pca = PCA(n_components=n)
        feature = pca.fit_transform(feature1)
        acc_dt1 = []
        acc_svm1 = []
        acc_lr1 = []
        acc_mlp1 = []
        for i in range(15):

            featureFolder, labelFolder = crossValidation(feature, label, 3)
            svm_C = 128
            svm_gamma = 0.00390625

    #svm_GridSearch_creditScore(trainFeature, trainLabel)


    #LR
    #accu12, accu22 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
    #bagging LR
            accu13, accu23 = crossValidationFunc(featureFolder, labelFolder, bagging_LR, bagging_size)
            acc_lr1.append((accu13+accu23)/2)
    #bagging_twoLayer_LR
    #accu1a, accu2a = crossValidationFunc(featureFolder, labelFolder, bagging_twoLayer_LR, 11)

    #decision tree
    #accu14, accu24 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    #accu15, accu25 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)

    #bagging adboost decision tree
            accu16, accu26 = crossValidationFunc(featureFolder, labelFolder, bagging_DT, bagging_size, 50, 1.0)
            acc_dt1.append((accu16+accu26)/2)
    #accu17, accu27 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)
    #accu18, accu28 = crossValidationFunc(featureFolder, labelFolder, GBDT)

    #accu1b, accu2b = crossValidationFunc(featureFolder, labelFolder, svmclassifier, svm_C, svm_gamma)
            accu1c, accu2c = crossValidationFunc(featureFolder, labelFolder, baggingSVM, bagging_size, svm_C, svm_gamma)
            acc_svm1.append((accu1c+accu2c)/2)

    #accu1d, accu2d = crossValidationFunc(featureFolder, labelFolder, knn, 1)
    #accu1e, accu2e = crossValidationFunc(featureFolder, labelFolder, bagging_KNN, 1, 11)

            accu1n, accu2n = crossValidationFunc(featureFolder, labelFolder, MLP)
            acc_mlp1.append((accu1n+accu2n)/2)

        acc_dt1 = np.array(acc_dt1)
        acc_svm1 = np.array(acc_svm1)
        acc_lr1 = np.array(acc_lr1)
        acc_mlp1 = np.array(acc_mlp1)
        print("<<<<<<========pca dimension %d========>>>>>>" % (n))
        print("decision tree:")
        print(acc_dt1)
        print(np.mean(acc_dt1))
        print("support vector machine:")
        print(acc_svm1)
        print(np.mean(acc_svm1))
        print("logistic regression:")
        print(acc_lr1)
        print(np.mean(acc_lr1))
        print("neural network:")
        print(acc_mlp1)
        print(np.mean(acc_mlp1))
        acc_dt.append((n, np.mean(acc_dt1)))
        acc_svm.append((n, np.mean(acc_svm1)))
        acc_lr.append((n, np.mean(acc_lr1)))
        acc_mlp.append((n, np.mean(acc_mlp1)))
    '''
    print()
    print("==================================different classifiers================================")
    print("classifier               type1 accuracy      type2 accuracy      average accuracy")
    #print("    DT                       %f            %f            %f") % (accu14, accu24, (accu14+accu24)/2)
    #print(" AdaDT                       %f            %f            %f") % (accu15, accu25, (accu15+accu25)/2)
    #print("Random Forest                %f            %f            %f") % (accu17, accu27, (accu17+accu27)/2)
    #print(" GBDT                        %f            %f            %f") % (accu18, accu28, (accu18+accu28)/2)
    print("bagging AdaDT                %f            %f            %f") % (accu16, accu26, (accu16+accu26)/2)
    #print("   LR                        %f            %f            %f") % (accu12, accu22, (accu12+accu22)/2)
    print("bagging LR                   %f            %f            %f") % (accu13, accu23, (accu13+accu23)/2)
    #print("bagging two_layer LR         %f            %f            %f") % (accu1a, accu2a, (accu1a+accu2a)/2)
    #print("SVM                          %f            %f            %f") % (accu1b, accu2b, (accu1b+accu2b)/2)
    print("bagging SVM                  %f            %f            %f") % (accu1c, accu2c, (accu1c+accu2c)/2)
    #print("knn                          %f            %f            %f") % (accu1d, accu2d, (accu1d+accu2d)/2)
    #print("bagging knn                  %f            %f            %f") % (accu1e, accu2e, (accu1e+accu2e)/2)
    print("MLP                            %f            %f            %f") % (accu1n, accu2n, (accu1n+accu2n)/2)
    '''
    print("dt")
    print(acc_dt)
    print("svm")
    print(acc_svm)
    print("lr")
    print(acc_lr)
    print("nn")
    print(acc_mlp)
    print(feature.shape)