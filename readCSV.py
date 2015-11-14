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
        filePath = 'noisy data/trainingData/' + str(num) + '.csv'
        f1 = open(filePath, 'rb')
        nullNum = np.zeros(featureNum)
        i = 0
        for row in csv.reader(f1):
            if(i == 0):
                i += 1
                continue
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
    print(sampleNum)
    '''
    f2 = open('TrainingRichness_1.csv', 'wb')
    csvwriter = csv.writer(f2)
    csvwriter.writerow(['releasetime', 'fulltime', 'repaytime', 'item_name', 'item_amount', 'item_status', 'item_introduction',
                        'life_loan', 'money_rate', 'loan_type', 'loan_use', 'bonding_company', 'deadline', 'repaytype',
                        'borrower_name', 'borrower_sex', 'borrower_age', 'borrower_education', 'borrower_marriage',
                        'borrower_industry', 'borrower_city', 'borrower_income', 'borrower_house', 'borrower_car',
                        'reward'])
    richNess = np.array(richNess)[:, 1:]
    for item in richNess:
        csvwriter.writerow(item)
    '''
    return(np.array(richNess)[:, 1:], np.array(testLabel))

def repayTime2deadLine_day(dateList, subtractedDays): #dateList format'2015-08-27', subtractedDays format7
    while(subtractedDays > 0):
        if(dateList[2] > 1):
            dateList[2] -= 1
        elif(dateList[2] == 1):
            if(dateList[1] in [5, 7, 10, 12]):
                dateList[2] = 30
                dateList[1] -= 1
            elif(dateList[1] in [2, 4, 6, 8, 9, 11]):
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

def repayTime2deadLine_mon(dateList, subtractedMons):
    while(subtractedMons > 0):
        if(dateList[1] > 1):
            dateList[1] -= 1
        elif(dateList[1] == 1):
            dateList[0] -= 1
            dateList[1] = 12
        subtractedMons -= 1
    return(dateList)

#as life_loan has two types of representations: days and months
def generateDateFeature(filePath1, filePath2):
    f1 = open(filePath1, 'rb')
    readLines = csv.reader(f1)
    i = 0
    newDateList = [['deadline', 'life loan']]
    for rows in readLines:
        if(i == 0):
            i += 1
            continue
        if(rows[0] == ''):
            if(rows[1] == ''):
                newDateList.append(['', ''])
            else:
                newDateList.append(['', rows[1].split(' ')[0]+rows[1].split(' ')[1]])
                #newDateList.append(['', rows[1])
        else:
            dateList = [int(rows[0][0:4]), int(rows[0][5:7]), int(rows[0][8:10])]
            if(rows[1] == ''):
                newDateList.append([dateList, ''])
            else:
                if(rows[1][-3:] == '\xe5\xa4\xa9'):
                    newDateList.append([repayTime2deadLine_day(dateList, int(rows[1].split(' ')[0])), rows[1].split(' ')[0]+rows[1].split(' ')[1]])
                    #newDateList.append([repayTime2deadLine_day(dateList, int(rows[1][:-3])), rows[1]])
                else:
                    print(len(rows[1]),rows[1])
                    newDateList.append([repayTime2deadLine_mon(dateList, int(rows[1].split(' ')[0])), rows[1].split(' ')[0]+rows[1].split(' ')[1]])
                    #newDateList.append([repayTime2deadLine_mon(dateList, int(rows[1][:-7])), rows[1]])
        #print([dateList, int(rows[1].decode('utf-8-sig')[:-1])])
    f1.close()
    print(newDateList)

    f2 = open(filePath2, 'wb')
    writeLines = csv.writer(f2)
    writeLines.writerow(newDateList[0])
    for item in newDateList[1 : ]:
        if(item[0] != ''):
            year = str(item[0][0])
            month = '0'+str(item[0][1]) if(item[0][1] < 10) else str(item[0][1])
            day = '0'+str(item[0][2]) if(item[0][2] < 10) else str(item[0][2])
            item[0] = year + '-' + month + '-' + day
        writeLines.writerow(item)
    f2.close()
    print(newDateList)
    return 0


def generateTradingTime_date(filePath1, filePath2):
    f = open(filePath1)
    csvreader = csv.reader(f)
    dataPair = [['release time', 'life loan']]
    i = 0
    for rows in csvreader:
        if(i == 0):
            i += 1
            continue

        #if some items are null and repayment time + months //9.csv
        if((rows[0] == '\\N' or rows[0] == '') and (rows[1] == '\\N' or rows[1] == '')):
            dataPair.append(['', ''])
        elif((rows[0] == '\\N' or rows[0] == '') and (rows[1] != '\\N' and rows[1] != '')):
            if(rows[1][-3:] == '\xe6\x9c\x88'):
                dataPair.append(['', str(int(rows[1][:-6])*30)])
                #dataPair.append([rows[0][:4]+'-'+rows[0][4:6]+'-'+rows[0][6:8], str(int(rows[1][:-7])*30)])
            elif(rows[1][-3:] == '\xe5\xa4\xa9'):
                dataPair.append(['', rows[1][:-3]])
            else:
                dataPair.append(['', rows[1]])
        elif((rows[0] != '\\N' and rows[0] != '') and (rows[1] == '\\N' or rows[1] == '')):
            dataPair.append([rows[0][:10], ''])
        else:
            if(rows[1].split(' ')[-1][-3:] == '\xe6\x9c\x88'):
                dataPair.append([rows[0].split(' ')[0], str(int(rows[1].split(' ')[0])*30)])
                #dataPair.append([rows[0][:4]+'-'+rows[0][4:6]+'-'+rows[0][6:8], str(int(rows[1][:-7])*30)])
            elif(rows[1].split(' ')[-1][-3:] == '\xe5\xa4\xa9'):
                dataPair.append([rows[0].split(' ')[0], rows[1].split(' ')[0]])
            else:
                dataPair.append([rows[0].split(' ')[0], rows[1]])


        '''
        if(rows.split(' ')[2] != '\xe5\xa4\xa9'):
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

        '''
        #comlicated date representation //7.csv
        if(i % 2 != 0):
            dataPair.append([rows.split(',')[0][1:-1], ''])
        else:
            dataPair[-1][1] = rows.split(',')[1][:-4]
        i += 1
        '''
        '''
        rows0 = rows.split(',')[0]
        rows1 = rows.split(',')[1]
        if(rows1 != '\n'):
            rows1 = str(int(rows1[:-1])*30)
        dataPair.append([rows0, rows1])
        '''

        '''
        if(i % 3 == 1):
            dataPair.append([rows.split('\t')[0][:11], ''])
        elif(i % 3 == 2):
            dataPair[-1][1] = rows.split('\t')[6][:-1]
        else:
            if(rows.split('\t')[6] != '\xe5\xa4\xa9"\n'):
                dataPair[-1][1] = str(int(dataPair[-1][1])*30)
            #print(rows.split('\t'))
        i += 1
        '''

        print(rows[0], rows[1].split(' '))
    print(dataPair)
    f.close()

    #print(dataPair)

    f1 = open(filePath2, 'wb')
    csvwriter = csv.writer(f1)
    for item in dataPair:
        csvwriter.writerow(item)
    f1.close()

    return(0)

if(__name__ == "__main__"):

    #dataFeature, dataLabel = generateRichnessDataset()
    #featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, 5)

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

    #print(accu1, accu2, (accu1+accu2)/2)

    #print(repayTime2deadLine([2015, 9, 19], 180))
    #generateDateFeature('3.csv')
    #generateDateFeature('18.csv', '18-2.csv')
    generateTradingTime_date('30.csv', '30-1.csv')


'''
t = gainRatio(dataFeature, dataLabel)
ratioDict = []
for i in range(26):
    ratioDict.append((i, t[i]))
print(ratioDict)
ratioDict = sorted(ratioDict, key = lambda x : -x[1])
print(ratioDict)
'''
