import numpy as np
import csv

def generateLifeLoan(filePath):
    with open(filePath, 'rb') as f:
        csvreader = csv.reader(f)
        i = 0
        lifeLoanList = []
        for rows in csvreader:
            if(i == 0):
                i += 1
            else:
                if(len(rows) < 2):
                    lifeLoanList.append(0)
                    continue
                if(rows[1] == '' or rows[1] == 'dynamic'):
                    continue
                else:
                    lifeLoanList.append(int(rows[1]))
        f.close()
    dim = 10
    interval = len(lifeLoanList) / dim
    lifeLoanArray = np.array(sorted(lifeLoanList))
    average_lifeLoan_list = [np.mean(lifeLoanArray[interval*i : interval*(i+1)]) for i in range(dim-1)]
    average_lifeLoan_list.append(np.mean(lifeLoanArray[interval*(i+1) : ]))
    print(len(lifeLoanList), lifeLoanArray)
    print(average_lifeLoan_list)
    return(average_lifeLoan_list)

def generateLifeFeature():
    writeFilePath = 'lifeLoan2.csv'
    with open(writeFilePath, 'wb') as f:
        csvwriter = csv.writer(f)
        for i in range(51, 71):
            print(i)
            readFilePath = 'clean data/testingData/' + str(i) +'.csv'
            average_lifeLOan_list = generateLifeLoan(readFilePath)
            if(len(average_lifeLOan_list) == 0):
                csvwriter.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                csvwriter.writerow(average_lifeLOan_list)
        f.close()

if __name__ == "__main__":
    generateLifeFeature()