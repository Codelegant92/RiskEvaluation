__author__ = 'robin'

import numpy as np
import csv

def generateTradeFreq(filePath):
    f = open(filePath, 'rb')
    csvreader = csv.reader(f)
    i = 0
    timesPerDay = dict()
    date_timesPerDay_list = []
    timesPerDay_list = []
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
    for keys in timesPerDay:
        date_timesPerDay_list.append((keys, timesPerDay[keys]))
    date_timesPerDay_list = sorted(date_timesPerDay_list, key = lambda x : x[0])
    for items in date_timesPerDay_list:
        timesPerDay_list.append(items[1])
    timesPerDay_list = np.array(timesPerDay_list)
    average_timesPerDay_list = []
    dim = 10
    interval = len(timesPerDay_list) / dim
    for i in range(dim - 1):
        average_timesPerDay_list.append(np.mean(timesPerDay_list[interval*i : interval*(i+1)-1]))
    average_timesPerDay_list.append(np.mean(timesPerDay_list[interval*(i+1) : ]))
    print(average_timesPerDay_list)


generateTradeFreq('clean data/trainingData/4.csv')