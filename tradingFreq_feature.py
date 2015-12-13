__author__ = 'robin'
import numpy as np
import csv

def generateFreqFeature(dim = 10, richTolerance = 0.5):      #the returned trading frequency feature is a 70*dim ndarray of string
    print("********************************Generating Trading Frequency Feature*******************************")
    tradingFreq_feature = []
    writeFilePath = 'cache/tradeFrequency-'+str(dim)+'dim-feature.csv'
    with open(writeFilePath, 'wb') as f:
        csvwriter = csv.writer(f)
        for i in range(1, 51):
            readFilePath1 = 'clean data/trainingData/' + str(i) + '.csv'
            average_timesPerDay_list_train = generateTradeFreq(readFilePath1, dim, richTolerance)
            tradingFreq_feature.append(average_timesPerDay_list_train)
            csvwriter.writerow(average_timesPerDay_list_train)

        for i in range(51,71):
            readFilePath2 = 'clean data/testingData/' + str(i) + '.csv'
            average_timesPerDay_list_test = generateTradeFreq(readFilePath2, dim, richTolerance)
            tradingFreq_feature.append(average_timesPerDay_list_test)
            csvwriter.writerow(average_timesPerDay_list_test)
        f.close()
    return(np.array(tradingFreq_feature))

def generateTradeFreq(filePath, dim, richTolerance):
    with open(filePath, 'rb') as f:
        i = 0
        j = 0
        timesPerDay = dict()
        for rows in csv.reader(f):
            if(i == 0):
                i += 1
            else:
                if(rows[0] == '' or rows[0] == '\\N'):
                    j += 1
                else:
                    if(rows[0] not in timesPerDay.keys()):
                        timesPerDay[rows[0]] = 1
                    else:
                        timesPerDay[rows[0]] += 1
                i += 1
        f.close()
    date_timesPerDay_list = [(keys, timesPerDay[keys]) for keys in timesPerDay]
    date_timesPerDay_list = sorted(date_timesPerDay_list, key = lambda x : x[0])
    timesPerDay_list = [items[1] for items in date_timesPerDay_list]
    interval = len(timesPerDay_list) / dim
    timesPerDay_list = np.array(timesPerDay_list)
    if(interval >= 1 and float(j)/i < richTolerance):
        average_timesPerDay_list = [np.mean(timesPerDay_list[interval*i : interval*(i+1)]) for i in range(dim-1)]
        average_timesPerDay_list.append(np.mean(timesPerDay_list[interval*(i+1) : ]))
        #print(len(date_timesPerDay_list), date_timesPerDay_list)
        #print(average_timesPerDay_list)
    else:
        average_timesPerDay_list = ['nan' for i in range(dim)]
    return(average_timesPerDay_list)

if __name__ == "__main__":
    feature = generateFreqFeature(10, 0.5)
    print(feature)