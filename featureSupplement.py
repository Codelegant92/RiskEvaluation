from richness_feature import *
from tradingFreq_feature import *
from lifeLoan_feature import *
from amount_feature import *
from moneyRate_feature import *

def readFeature():
    print("********************************Combine All the Features***************************************")

    with open('cache/feature.csv') as f:
        fullFeature = []
        missFeature = []
        csvreader = csv.reader(f)
        i = 0
        for rows in csvreader:
            if(i == 0):
                i += 1
                continue
            featureTemp = [float(item) if item != 'nan' else item for item in rows]
            if('nan' in featureTemp):
                missIndices = [j for j, x in enumerate(featureTemp) if(x == 'nan')]
                featureTemp = [0 if(x == 'nan') else x for x in featureTemp]
                missFeature.append((featureTemp, missIndices))
            else:
                fullFeature.append(featureTemp)
        f.close()
    print("++++++++++++++++++++++++++++begin supplementing missing features++++++++++++++++++++++++++++++++++++++++")
    #a parameter of supplementFeature is the number of nearest neighbors of the missing samples
    newFullFeature, newMissFeature = supplementFeature(fullFeature, missFeature, 20)
    trainSamples = []
    testFeature = []
    for item in newFullFeature:
        if(item[-1] == 9):
            testFeature.append(item[:-1])
        else:
            trainSamples.append(item)
    for item in newMissFeature:
        if(item[-1] == 9):
            testFeature.append(item[:-1])
        else:
            trainSamples.append(item)
    trainFeature = np.array(trainSamples)[:, :-1]
    trainLabel = np.array(trainSamples)[:, -1]
    print("++++++++++++++++++++++++++++end supplementing missing features++++++++++++++++++++++++++++++++++++++++")
    return(trainFeature, trainLabel, testFeature)

def featureCombination(tradingFreq_dim, tradingFreq_tolerance, lifeLoan_dim, lifeLoan_tolerance, amount_dim, amount_tolerance, moneyRate_dim, moneyRate_tolerance):
    richNess = generateRichnessFeature()
    tradingFreq = generateFreqFeature(tradingFreq_dim, tradingFreq_tolerance)
    lifeLoan = generateLifeFeature(lifeLoan_dim, lifeLoan_tolerance)
    amount = generateAmountFeature(amount_dim, amount_tolerance)
    moneyRate = generateMoneyRateFeature(moneyRate_dim, moneyRate_tolerance)

    Label = np.array([1 if(i < 20) else 0 if(i < 50) else 9 for i in range(70)])
    Feature = np.concatenate([richNess.T, tradingFreq.T, lifeLoan.T, amount.T, moneyRate.T, Label.T]).T

    with open('cache/feature.csv', 'wb') as f:
        csvwriter = csv.writer(f)
        for item in Feature:
            csvwriter.writerow(item)
        f.close()
    print(Label)

#to supplement the missing features, parameter ---> nearestNum
def supplementFeature(fullFeature, missFeature, nearestNum = 6):
    supplementedFeature = []
    fullFeature = np.array(fullFeature)

    for missFeatureTuple in missFeature:        #start to supplement missing features
        singleMissFeature = np.array(missFeatureTuple[0])

        print("=========================================================")
        valuedFeatureIndices = [i for i in range(singleMissFeature.shape[0]-1) if(i not in missFeatureTuple[1] and i != 25)]
        missedFeatureIndices = missFeatureTuple[1]
        '''
        print("valuedFeatureIndices:")
        print(valuedFeatureIndices)
        print("missedFeatureIndices:")
        print(missedFeatureIndices)
        print("----------------------------------------")
        '''

        euclideanDistance = []                  #calculate the distance between the missfeature and each full feature
        for singleFeature in fullFeature:
            euclideanDistance.append(np.linalg.norm(singleFeature[valuedFeatureIndices] - singleMissFeature[valuedFeatureIndices]))
        distanceTupleList = [(indice, distance) for indice, distance in enumerate(euclideanDistance)]
        distanceTupleList = sorted(distanceTupleList, key = lambda x: x[1])
        nearestIndices = [distanceTupleList[i][0] for i in range(nearestNum)]
        '''
        print("distanceTupleList:")
        print(distanceTupleList)
        print("nearestIndices:")
        print(nearestIndices)
        print("----------------------------------------")
        '''

        for i in missedFeatureIndices:
            singleMissFeature[i] = np.mean(fullFeature.T[i, nearestIndices])
            print("i->%d, means->%f") % (i, singleMissFeature[i])
        supplementedFeature.append(singleMissFeature)
    supplementedFeature = np.array(supplementedFeature)

    #Feature = SelectKBest(chi2, k=para_k).fit_transform(Feature, Label)
    #Feature = PCA(n_components=para_k).fit_transform(Feature, Label)
    return(fullFeature, supplementedFeature)



if(__name__ == "__main__"):
    featureCombination(10, 0.5, 10, 0.5, 10, 0.6, 3, 0.6)