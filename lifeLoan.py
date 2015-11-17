import numpy as np
import csv

def generateLifeLoan(filePath):
    f = open(filePath, 'rb')
    csvreader = csv.reader(f)