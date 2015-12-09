import csv
import numpy as np

def readFeature():
    with open('feature/feature.csv') as f:
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
    pass