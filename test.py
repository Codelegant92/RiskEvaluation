
from mainFunc import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2

if(__name__ == "__main__"):
    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 5, 1, 20, 1, 5, 0.6, 20)
    ctest = SelectKBest(chi2, k = 2)
    ctest.fit(trainFeature, trainLabel)
    x1 = ctest.transform(trainFeature)
    x2 = ctest.transform(testFeature)

    print(x1)
    print(x2)