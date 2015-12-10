import csv
import numpy as np

def generateRichnessFeature():
    print("********************************Generating Richness Feature***************************************")
    richNess = []
    for num in xrange(1, 51):       #calculate the richness of training samples
        sampleValue = []
        trainFilePath = 'noisy data/trainingData/' + str(num) + '.csv'
        with open(trainFilePath, 'rb') as f1:
            i = 0
            for row in csv.reader(f1):
                if(i == 0):
                    i += 1
                    continue
                sampleValue.append(row[1:26])
                i += 1
            f1.close()
        sampleValue_transform = np.array(sampleValue).T
        subRichness = [1 - (list(features).count('\\N')+list(features).count('')) / float(i-1) for features in sampleValue_transform]
        subRichness.append(i-1)
        richNess.append(subRichness)

    for num in xrange(51, 71):      #calculate the richness of testing samples
        sampleValue = []
        testFilePath = 'noisy data/testingData/' + str(num) + '.csv'
        with open(testFilePath, 'rb') as f2:
            i = 0
            for row in csv.reader(f2):
                if(i == 0):
                    i += 1
                    continue
                sampleValue.append(row[1:26])
                i += 1
            f2.close()
        sampleValue_transform = np.array(sampleValue).T
        subRichness = [1 - (list(features).count('\\N')+list(features).count('')) / float(i-1) for features in sampleValue_transform]
        subRichness.append(i-1)
        richNess.append(subRichness)
    with open('cache/richness-feature.csv', 'wb') as f3:
        csvwriter = csv.writer(f3)
        for item in richNess:
            csvwriter.writerow(item)
        f3.close()
    return(np.array(richNess))

if(__name__ == "__main__"):
    richNess_feature = generateRichnessDataset()
    print(richNess_feature)