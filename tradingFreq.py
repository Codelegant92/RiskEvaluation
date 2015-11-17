__author__ = 'robin'

import numpy as np
import csv

def generateTradeFreq(filePath):
    with open(filePath, 'rb') as f:
        csvreader = csv.reader(f)
        i = 0
        timesPerDay = dict()
        for rows in csvreader:
            if(i == 0):
                i += 1
            else:
                if(rows[0] == ''):
                    continue
                else:
                    if(rows[0] not in timesPerDay.keys()):
                        timesPerDay[rows[0]] = 1
                    else:
                        timesPerDay[rows[0]] += 1
        f.close()
    date_timesPerDay_list = [(keys, timesPerDay[keys]) for keys in timesPerDay]
    date_timesPerDay_list = sorted(date_timesPerDay_list, key = lambda x : x[0])
    timesPerDay_list = [items[1] for items in date_timesPerDay_list]
    timesPerDay_list = np.array(timesPerDay_list)
    dim = 10
    interval = len(timesPerDay_list) / dim
    average_timesPerDay_list = [np.mean(timesPerDay_list[interval*i : interval*(i+1)-1]) for i in range(dim-1)]
    average_timesPerDay_list.append(np.mean(timesPerDay_list[interval*(i+1) : ]))
    print(len(date_timesPerDay_list), date_timesPerDay_list)
    print(average_timesPerDay_list)
    return(average_timesPerDay_list)

def generateFreqFeature():
    '''
    writeFilePath = 'tradeFreq.csv'
    with open(writeFilePath, 'wb') as f:
        csvwriter = csv.writer(f)
        for i in range(1,50):
            readFilePath = 'clean data/trainingData/' + str(i) + '.csv'
            average_timesPerDay_list = generateTradeFreq(readFilePath)
            if(len(average_timesPerDay_list) != 0):
                csvwriter.writerow(average_timesPerDay_list)
            else:
                csvwriter.writerow([0,0,0,0,0,0,0,0,0,0])
        f.close()
    '''
    for i in range(1,51):
        readFilePath = 'clean data/trainingData/' + str(i) + '.csv'
        average_timesPerDay_list = generateTradeFreq(readFilePath)

#generateTradeFreq('clean data/trainingData/18.csv')
generateFreqFeature()