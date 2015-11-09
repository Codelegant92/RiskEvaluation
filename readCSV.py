__author__ = 'yangchen'

import csv
from commonFunction import *
import numpy as np

from decisionTree import decision_Tree, adboostDT, bagging_adboostDT, RandomForest_Classifer
from svm_classification import svmclassifier, baggingSVM
from regression import logistic_regression
from KNN import knn

def generateRichnessDataset():
    richNess = []
    sampleNum = np.zeros(50)
    featureNum = 26
    for num in xrange(1, 51):
        filePath = 'Data/trainingData/' + str(num) + '.csv'
        f1 = open(filePath, 'rb')
        nullNum = np.zeros(featureNum)
        for row in csv.reader(f1):
            j = 0
            sampleNum[num-1] += 1      #count the total number of samples of each platform
            for rows in row:
                if(j < featureNum):
                    if(rows == '\\N' or rows == ''):
                        nullNum[j] += 1
                    j += 1
        richNess.append((sampleNum[num - 1] - nullNum) * 1.0 / sampleNum[num - 1])
        f1.close()
    #print(sampleNum)
    testLabel = np.zeros(50)
    for index in range(20):
        testLabel[index] = 1
    return(np.array(richNess), np.array(testLabel))

def repayTime2deadLine_day(dateList, subtractedDays): #dateList format'2015-08-27', subtractedDays format7
    while(subtractedDays > 0):
        if(dateList[2] > 1):
            dateList[2] -= 1
        elif(dateList[2] == 1):
            if(dateList[1] in [5, 7, 8, 10, 12]):
                dateList[2] = 30
                dateList[1] -= 1
            elif(dateList[1] in [2, 4, 6, 9, 11]):
                dateList[2] = 31
                dateList[1] -= 1
            elif(dateList[1] == 3):
                if((dateList[0] % 100 == 0 and dateList[0] % 400 == 0) or (dateList[0] % 100 != 0 and dateList[0] % 4 == 0)):
                    dateList[2] = 29
                    dateList[1] = 2
                else:
                    dateList[2] = 28
                    dateList[1] = 2
            else:
                dateList[0] -= 1
                dateList[1] = 12
                dateList[2] = 31
        subtractedDays -= 1
    return(dateList)

#def repayTime2deadLine_mon(dateList, subtractedMon):

def generateDateFeature(filePath1, filePath2):
    f1 = open(filePath1, 'rb')
    readLines = csv.reader(f1)
    date = []
    i = 0
    for rows in readLines:
        if(i == 0):
            i += 1
            continue
        dateList = [int(rows[0][0:4]), int(rows[0][5:7]), int(rows[0][8:10])]
        date.append([dateList, int(rows[1].decode('utf-8-sig')[:-1])])
        #print([dateList, int(rows[1].decode('utf-8-sig')[:-1])])
    f1.close()
    newDateList = [['deadline', 'life loan']]
    newDateList.extend([repayTime2deadLine_day(item[0], item[1]), item[1]] for item in date)
    print(newDateList)
    f2 = open(filePath2, 'wb')
    writeLines = csv.writer(f2)
    writeLines.writerow(newDateList[0])
    for item in newDateList[1 : ]:
        year = str(item[0][0])
        month = '0'+str(item[0][1]) if(item[0][1] < 10) else str(item[0][1])
        day = '0'+str(item[0][2]) if(item[0][2] < 10) else str(item[0][2])
        item[0] = year + '-' + month + '-' + day
        item[1] = str(item[1])
        writeLines.writerow(item)
    f2.close()
    print(newDateList)
    return 0


def generateTradingTime_date(filePath1, filePath2):
    f = open(filePath1)
    dataPair = []
    i = 0
    for rows in f.readlines():
        if(i == 0):
            i += 1
            continue

        '''
        if((rows.split(' ')[0] == '\\N' or rows.split(' ')[0] == '') and (rows.split(' ')[1] == '\\N' or rows.split(' ')[1] == '')):
            dataPair.append(['', ''])
        elif((rows.split(' ')[0] == '\\N' or rows.split(' ')[0] == '') and (rows.split(' ')[1] != '\\N' and rows.split(' ')[1] != '')):
            if(rows.split(' ')[1][-4:] != '\xe5\xa4\xa9\n'):
                dataPair.append(['', str(int(rows.split(' ')[1].split(',')[:-4]) * 30)])
            else:
                dataPair.append(['', rows.split(' ')[1].split(',')[:-4]])
        elif((rows.split(' ')[0] != '\\N' and rows.split(' ')[0] != '') and (rows.split(' ')[1] == '\\N' or rows.split(' ')[1] == '')):
            dataPair.append([rows.split(' ')[0], ''])
        else:
            if(rows.split(' ')[1][-4:] != '\xe5\xa4\xa9\n'):
                dataPair.append([rows.split(' ')[0], str(int(rows.split(' ')[1].split(',')[:-4]) * 30)])
            else:
                dataPair.append([rows.split(' ')[0], rows.split(' ')[1].split(',')[:-4]])
        '''

        '''
        if(rows.split(' ')[2] != '\xe5\xa4\xa9\n'):
            dataPair.append([rows.split(' ')[0], str(int(rows.split(' ')[1].split(',')[1]) * 30)])
        else:
            dataPair.append([rows.split(' ')[0], rows.split(' ')[1].split(',')[1]])
        '''

        '''
        #2014-12-20 11:00:00, 24months, no other conatraints
        #dataPair.append([rows.split(' ')[0], str(int(rows.split(' ')[1].split(',')[1][:-5]) * 30)])
        #2014-06-21 12months/31days, to seperate these two
        if(rows.split(',')[1][-4:] == '\xe5\xa4\xa9\n'):
            dataPair.append([rows.split(',')[0], rows.split(',')[1][:-4]])
        else:
            dataPair.append([rows.split(',')[0], str(int(rows.split(',')[1][:-7]) * 30)])
        '''

        '''
        #complement '0' to single number
        if(rows.split(',')[0] == '\\N' and rows.split(',')[1] == '\\N\n'):
            dataPair.append(['', ''])
        else:
            date = rows.split(',')[0].split('-')
            print(date)

            if(len(date[1]) == 1):
                date[1] = '0'+date[1]
            if(len(date[2]) == 1):
                date[2] = '0'+date[2]
            date = date[0] + '-' + date[1] + '-' + date[2]
            dataPair.append([date, rows.split(',')[1][:-1]])
        #print(rows.split(',')[0].split('-'))
        '''

        #comlicated date representation //7.csv
        if(i % 2 != 0):
            dataPair.append([rows.split(',')[0][1:-1], ''])
        else:
            dataPair[-1][1] = rows.split(',')[1][:-4]
        i += 1

    f.close()

    #print(dataPair)

    f1 = open(filePath2, 'wb')
    csvwriter = csv.writer(f1)
    for item in dataPair:
        csvwriter.writerow(item)
    f1.close()

    return(0)

if(__name__ == "__main__"):
    '''
    dataFeature, dataLabel = generateRichnessDataset()
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, 5)
    '''
    #knn
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, knn, 1)
    #logistic regression
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
    #decision tree
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    #adboost decision tree
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, adboostDT, 50, 1.0)
    #bagging adboost decision tree
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, bagging_adboostDT, 50, 1.0)
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, RandomForest_Classifer)
    #svm
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, 2.0, 0.0625)
    #bagging svm
    #accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, baggingSVM, 2.0, 0.0625)
    '''
    print(accu1, accu2, (accu1+accu2)/2)
    '''
    #print(repayTime2deadLine([2015, 9, 19], 180))
    #generateDateFeature('3.csv')
    '''
    dataFeature, dataLabel = generateTest('Output/13237.txt', 'Output/13237-1.txt')
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, 5)
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, decision_Tree)
    print(accu1, accu2)
    '''
    #generateTradingTime_date('7.csv', '11.csv')
    f = open('11.csv', 'rb')
    csvreader = csv.reader(f)
    for rows in csvreader:
        print(rows)
    f.close()


'''
t = gainRatio(dataFeature, dataLabel)
ratioDict = []
for i in range(26):
    ratioDict.append((i, t[i]))
print(ratioDict)
ratioDict = sorted(ratioDict, key = lambda x : -x[1])
print(ratioDict)
'''
