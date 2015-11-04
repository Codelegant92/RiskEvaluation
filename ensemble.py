from commonFunction import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
'''
def svmclassifier(trainFeature, trainLabel, testFeature, para_C, para_gamma):
    clf = svm.SVC(C = para_C, gamma = para_gamma)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)
'''
def adboostDT(trainFeature, trainLabel, testFeature, estimatorNum = 50, learningRate = 1.0):
    clf = AdaBoostClassifier(n_estimators = estimatorNum, learning_rate = learningRate)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

def bagging_adboostDT(trainFeature, trainLabel, testFeature, estimatorNum = 50, learningRate = 1.0):
    folderNum = 5
    predictedLabel_voting = []
    randomFeatureFolder, randomLabelFolder = crossValidation(trainFeature, trainLabel, folderNum)
    print("========bagging SVM========")
    for i in range(folderNum):
        subTrainFeature = []
        subTrainLabel = []
        for j in range(folderNum):
            if(j != i):
                subTrainFeature.extend(list(randomFeatureFolder[j]))
                subTrainLabel.extend(list(randomLabelFolder[j]))
        subTrainFeature = np.array(subTrainFeature)
        subTrainLabel = np.array(subTrainLabel)
        print("=====%dst Bagging=====") % (i+1)
        print("Positive: %d, Negative: %d") % (list(subTrainLabel).count(1), list(subTrainLabel).count(0))
        clf = AdaBoostClassifier(n_estimators = estimatorNum, learning_rate = learningRate)
        clf.fit(trainFeature, trainLabel)
        predictedLabel_temp = clf.predict(testFeature)
        predictedLabel_voting.append(predictedLabel_temp)
        print("%dst predicted labels:") % (i+1)
        print(predictedLabel_temp)
    predictedLabel_voting = np.array(predictedLabel_voting).T
    predictedLabel = [1 if(list(predictedLabel_voting[i]).count(1) > list(predictedLabel_voting[i]).count(0)) else 0 for i in range(predictedLabel_voting.shape[0])]
    print(predictedLabel)
    return(predictedLabel)

'''
#dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
trainFeature = dataFeature[:800, :]
trainLabel = dataLabel[:800]
testFeature = dataFeature[800:, :]
testLabel = dataLabel[800:]
clf1 = DecisionTreeClassifier()
clf2 = KNeighborsClassifier(n_neighbors=10)
clf3 = LogisticRegression(penalty = 'l2', dual = False)
clf4 = SVC(C=500, gamma = 0.000244140625, cache_size=600)
clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm = 'SAMME', n_estimators = 200)

clf1.fit(trainFeature, trainLabel)
clf2.fit(trainFeature, trainLabel)
clf3.fit(trainFeature, trainLabel)
clf4.fit(trainFeature, trainLabel)
clf5.fit(trainFeature, trainLabel)
predictedLabel1 = clf1.predict(testFeature)
predictedLabel2 = clf2.predict(testFeature)
predictedLabel3 = clf3.predict(testFeature)
predictedLabel4 = clf4.predict(testFeature)
predictedLabel5 = clf5.predict(testFeature)

predictedLabel = []
sampleNum = testLabel.shape[0]
print(sampleNum)
i = 0
while(i < sampleNum):
    if((predictedLabel1[i]+predictedLabel2[i]+predictedLabel3[i]+predictedLabel4[i]+predictedLabel5[i])>2):
        predictedLabel.append(1)
    else:
        predictedLabel.append(0)
    i += 1
diff = list(testLabel - predictedLabel)
type_one_error = diff.count(1)/float(list(testLabel).count(1))
type_two_error = diff.count(-1)/float(list(testLabel).count(0))
print(1-type_one_error ,1-type_two_error)
'''