from richness_feature import *
from tradingFreq_feature import *
from lifeLoan_feature import *
from amount_feature import *
from moneyRate_feature import *
from sklearn import preprocessing

def readFeature(tradingFreq_dim, tradingFreq_tolerance, lifeLoan_dim, lifeLoan_tolerance, amount_dim, amount_tolerance, moneyRate_dim, moneyRate_tolerance, supplement_nearestNum):
    print("********************************Combine All the Features***************************************")
    featureCombination(tradingFreq_dim, tradingFreq_tolerance, lifeLoan_dim, lifeLoan_tolerance, amount_dim, amount_tolerance, moneyRate_dim, moneyRate_tolerance)
    with open('cache/feature.csv', 'rb') as f:
        fullFeature = []
        missFeature = []
        csvreader = csv.reader(f)
        i = 1
        for rows in csvreader:
            featureTemp = [float(item) if item != 'nan' else item for item in rows]
            if('nan' in featureTemp):
                missIndices = [j for j, x in enumerate(featureTemp) if(x == 'nan')]
                featureTemp = [0 if(x == 'nan') else x for x in featureTemp]
                missFeature.append((featureTemp, missIndices))
                print(i)
                i += 1
            else:
                fullFeature.append(featureTemp)
                i += 1
        f.close()

    print("++++++++++++++++++begin supplementing missing features+++++++++++++++++")
    #a parameter of supplementFeature is the number of nearest neighbors of the missing samples
    newFullFeature, newMissFeature = supplementFeature(fullFeature, missFeature, supplement_nearestNum)
    trainSamples = []
    trainLabel = []
    testFeature = []
    testPlatform = []
    for item in newFullFeature:
        if(item[-1] < 21):
            trainLabel.append(1)
            trainSamples.append(item[:-1])
        elif(item[-1] < 51):
            trainLabel.append(0)
            trainSamples.append(item[:-1])
        else:
            testFeature.append(item[:-1])
            testPlatform.append(item[-1])

    for item in newMissFeature:
        if(item[-1] < 21):
            trainLabel.append(1)
            trainSamples.append(item[:-1])
        elif(item[-1] < 51):
            trainLabel.append(0)
            trainSamples.append(item[:-1])
        else:
            testFeature.append(item[:-1])
            testPlatform.append(item[-1])

    trainFeature = np.array(trainSamples)
    trainLabel = np.array(trainLabel)
    testFeature = np.array(testFeature)
    #normalize the feature
    featureMax = np.max(np.concatenate([trainFeature, testFeature]), axis=0)
    featureMin = np.min(np.concatenate([trainFeature, testFeature]), axis=0)

    trainFeature = (trainFeature - featureMin) / (featureMax - featureMin)
    testFeature = (testFeature - featureMin) / (featureMax - featureMin)

    '''
    trainNum = trainFeature.shape[0]
    scalar = preprocessing.StandardScaler()
    combinedFeature = np.concatenate([trainFeature, testFeature])
    normalizedFeature = scalar.fit_transform(combinedFeature)
    trainFeature = normalizedFeature[:trainNum, :]
    testFeature = normalizedFeature[trainNum:, :]
    '''
    print("++++++++++++++++++end supplementing missing features++++++++++++++++++++")
    return(trainFeature, trainLabel, testFeature, testPlatform)

def featureCombination(tradingFreq_dim, tradingFreq_tolerance, lifeLoan_dim, lifeLoan_tolerance, amount_dim, amount_tolerance, moneyRate_dim, moneyRate_tolerance):
    richNess = generateRichnessFeature()
    tradingFreq = generateFreqFeature(tradingFreq_dim, tradingFreq_tolerance)
    lifeLoan = generateLifeFeature(lifeLoan_dim, lifeLoan_tolerance)
    amount = generateAmountFeature(amount_dim, amount_tolerance)
    moneyRate = generateMoneyRateFeature(moneyRate_dim, moneyRate_tolerance)
    platform = np.array([i for i in xrange(1, 71)])
    Feature = np.concatenate([richNess.T, tradingFreq.T, lifeLoan.T, amount.T, moneyRate.T])
    featureLabel = list(Feature)
    featureLabel.append(platform)
    featureLabel = np.array(featureLabel).T
    with open('cache/feature.csv', 'wb') as f:
        csvwriter = csv.writer(f)
        for item in featureLabel:
            csvwriter.writerow(item)
            #csvwriter.writerow(Label[i])
        f.close()


#to supplement the missing features, parameter ---> nearestNum
def supplementFeature(fullFeature, missFeature, nearestNum):
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
    return(fullFeature, supplementedFeature)



if(__name__ == "__main__"):
    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 10, 0.5, 15, 0.5, 5, 0.6, 1)
    print(trainFeature.shape)
    print(trainLabel)
    print(testFeature.shape)
    print(testPlatform)