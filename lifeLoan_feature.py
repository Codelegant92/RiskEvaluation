import numpy as np
import csv

def generateLifeFeature(dim = 10, richTolerance = 0.5):      #the returned life loan feature is a 70*dim ndarray of string
    print("********************************Generating Lifeloan Feature***************************************")
    lifeLoan_feature = []
    writeFilePath = 'cache/lifeLoan-'+str(dim)+'dim-feature.csv'
    with open(writeFilePath, 'wb') as f:
        csvwriter = csv.writer(f)
        for i in range(1, 51):
            readFilePath1 = 'clean data/trainingData/' + str(i) +'.csv'
            average_lifeLoan_list_train = generateLifeLoan(readFilePath1, dim, richTolerance)
            lifeLoan_feature.append(average_lifeLoan_list_train)
            csvwriter.writerow(average_lifeLoan_list_train)

        for i in range(51, 71):
            readFilePath2 = 'clean data/testingData/' + str(i) +'.csv'
            average_lifeLoan_list_test = generateLifeLoan(readFilePath2, dim, richTolerance)
            lifeLoan_feature.append(average_lifeLoan_list_test)
            csvwriter.writerow(average_lifeLoan_list_test)
        f.close()
    return(np.array(lifeLoan_feature))

def generateLifeLoan(filePath, dim, richTolerance):
    with open(filePath, 'rb') as f:
        i = 0
        j = 0
        lifeLoanList = []
        for rows in csv.reader(f):
            if(i == 0):
                i += 1
            else:
                if(rows[1] == '' or rows[1] == 'dynamic' or rows[1] == '\\N'):
                    j += 1
                    continue
                else:
                    lifeLoanList.append(int(rows[1]))
                i += 1
        f.close()
    interval = len(lifeLoanList) / dim
    lifeLoanArray = np.array(sorted(lifeLoanList))
    if(interval >= 1 and float(j)/i < richTolerance):
        average_lifeLoan_list = [np.mean(lifeLoanArray[interval*i : interval*(i+1)])/30 for i in range(dim-1)]
        average_lifeLoan_list.append(np.mean(lifeLoanArray[interval*(i+1) : ])/30)
    else:
        average_lifeLoan_list = ['nan' for i in range(dim)]
    return(average_lifeLoan_list)

if __name__ == "__main__":
    feature = generateLifeFeature(10, 0.5)
    print(feature)