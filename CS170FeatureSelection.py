import csv
import math
import copy
import sys
import numpy as np
import time
import pandas as pd

def main():
    print("Type in the name of the file to test: ")
    filename = input()
    file = open(filename, 'r')
    s = csv.reader(file, delimiter=' ', skipinitialspace=True)
    num_features = len(next(s))

    print("Type the number of the algorithm you want to run:\n")
    print('1. Foward Selection\n')
    print('2. Backward Elimination\n\n')

    choice = int( input() )

    fil = pd.read_csv(filename)
    num_instances = len(fil)

    print('This dataset has ' + num_features + ' (not including the class attribute), with ' + num_instances + ' instances.')

    if choice == 1:
        forward(filename, num_features)
    elif choice == 2:
        backward(filename, num_features)

def forward(filename, num_features):
    added = set()
    curr = {}

    for i in range(1, num_features):
        bsf = 0
        featuretoadd = 0
        for j in range(1, num_features):
            if j not in added:
                #need to create deepcopy so its not changed later
                currset = copy.deepcopy(added)
                currset.add(j)
                #https://www.geeksforgeeks.org/how-to-read-text-files-with-pandas/
                #used read_fwf() because it said in the link above it works better with files of fixed column length, and i think that fits our file format better
                df = pd.read_fwf(filename, header=None)
                # need to make a copy of the dataframe too
                ##https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
                data = df.copy(deep=True)[:-1]
                accuracy = leave_one_out_cross_validation(data, currset, num_features)
                print('Using feature(s) ' + str(currset) + ' accuracy is ' + str(round(accuracy, 3)))
                if accuracy >= bsf:
                    bsf = accuracy
                    bsf_accuracy = accuracy
                    featuretoadd = j
        added.add(featuretoadd)
        addcopy = copy.deepcopy(added)
        curr[bsf_accuracy] = addcopy
        #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
        print('Feature set ' + str(added) + ' was best, accuracy is ' + str(round(bsf_accuracy, 3)) + '\n')

    #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
    print('Finished search!! The best feature subset is ' + str(curr[max(curr.keys())]) +
          ' which has an accuracy of ' + str(round(max(curr.keys()), 3)) + '\n')

main()
