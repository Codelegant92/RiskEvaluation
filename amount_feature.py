import numpy as np
import csv

def generateAmountFeature(dim = 10, richTolerance = 0.6):      #the returned amount feature is a 70*dim ndarray of string
    print("********************************Generating Amount Feature***************************************")
    amount_feature = []
    writeFilePath = 'cache/amount-'+str(dim)+'dim-feature.csv'
    with open(writeFilePath, 'wb') as f:
        csvwriter = csv.writer(f)
        for i in range(1, 51):
            readFilePath1 = 'clean data/trainingData/' + str(i) +'.csv'
            average_amount_list_train = generateAmount(readFilePath1, dim, richTolerance)
            amount_feature.append(average_amount_list_train)
            csvwriter.writerow(average_amount_list_train)

        for i in range(51, 71):
            readFilePath2 = 'clean data/testingData/' + str(i) +'.csv'
            average_amount_list_test = generateAmount(readFilePath2, dim, richTolerance)
            amount_feature.append(average_amount_list_test)
            csvwriter.writerow(average_amount_list_test)
        f.close()
    return(np.array(amount_feature))

def generateAmount(filePath, dim, richTolerance):
    with open(filePath, 'rb') as f:
        i = 0
        j = 0
        amountList = []
        for rows in csv.reader(f):
            if(i == 0):
                i += 1
            else:
                if(rows[2] == '' or rows[2] == '\\N'):
                    j += 1
                    continue
                else:
                    amountList.append(float(rows[2]))
                i += 1
        f.close()
    interval = len(amountList) / dim
    amountArray = np.array(sorted(amountList))
    if(interval >= 1 and float(j)/i < richTolerance):
        average_amount_list = [np.mean(amountArray[interval*i : interval*(i+1)])/10000 for i in range(dim-1)]
        average_amount_list.append(np.mean(amountArray[interval*(i+1) : ])/10000)
    else:
        average_amount_list = ['nan' for i in range(dim)]
    return(average_amount_list)

if __name__ == "__main__":
    feature = generateAmountFeature(10, 0.6)
    print(feature)