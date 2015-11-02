__author__ = 'yangchen'

import csv
import os
import numpy as np
'''
with open('Data/trainingData/6.csv', 'rb') as f:
    rd = csv.reader(f)
    i = 0
    for r in rd:
        if(i < 10):
            #print(type(r))
            #r1 = []
            #for item in r:
             #   r1.append(item.decode('utf-8-sig'))
            print(r)
            #print(','.join(r).decode('gb2312'))
            #print(','.join(r).decode('GB2312'))
            i += 1
'''

richNess = []
sampleNum = np.zeros(50)
featureNum = 26
for num in xrange(1, 51):
    filePath = 'Data/trainingData/' + str(num) + '.csv'
    f1 = open(filePath, 'rb')
    nullNum = np.zeros(featureNum)
    j = 0
    for row in csv.reader(f1):
        sampleNum[num] += 1      #count the total number of samples of each platform
        for rows in row:
            if(j < featureNum):
                if(rows == '\\N' or rows == ''):

    f1.close()
    print(smallRichness)
    richNess.append(smallRichness[:26])

f = open('trainingRichness.csv', 'wb')
w = csv.writer(f)
w.writerows(richNess)
f.close()

#richNessArray = np.array(richNess)
#print(richNessArray)
