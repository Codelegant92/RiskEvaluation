import numpy as np
import csv

def generateMoneyRateFeature(dim = 3, richTolerance = 0.6):      #the returned amount feature is a 70*dim ndarray of string
    print("********************************Generating Money RateFeature***************************************")
    moneyRate_feature = []
    writeFilePath = 'cache/moneyRate-'+str(dim)+'dim-feature.csv'
    with open(writeFilePath, 'wb') as f:
        csvwriter = csv.writer(f)
        for i in range(1, 51):
            readFilePath1 = 'clean data/trainingData/' + str(i) +'.csv'
            average_moneyRate_list_train = generateMoneyRate(readFilePath1, dim, richTolerance)
            moneyRate_feature.append(average_moneyRate_list_train)
            csvwriter.writerow(average_moneyRate_list_train)

        for i in range(51, 71):
            readFilePath2 = 'clean data/testingData/' + str(i) +'.csv'
            average_amount_list_test = generateMoneyRate(readFilePath2, dim, richTolerance)
            moneyRate_feature.append(average_amount_list_test)
            csvwriter.writerow(average_amount_list_test)
        f.close()
    return(np.array(moneyRate_feature))

def generateMoneyRate(filePath, dim, richTolerance):
    with open(filePath, 'rb') as f:
        i = 0
        j = 0
        moneyRateList = []
        for rows in csv.reader(f):
            if(i == 0):
                i += 1
            else:
                if(rows[3] == '' or rows[3] == '\\N'):
                    j += 1
                    continue
                else:
                    moneyRateList.append(float(rows[3])/100)
                i += 1
        f.close()
    interval = len(moneyRateList) / dim
    moneyRateArray = np.array(sorted(moneyRateList))
    if(interval >= 1 and float(j)/i < richTolerance):
        average_moneyRate_list = [np.mean(moneyRateArray[interval*i : interval*(i+1)]) for i in range(dim-1)]
        average_moneyRate_list.append(np.mean(moneyRateArray[interval*(i+1) : ]))
    else:
        average_moneyRate_list = ['nan' for i in range(dim)]
    return(average_moneyRate_list)

if __name__ == "__main__":
    feature = generateMoneyRateFeature()
    print(feature)