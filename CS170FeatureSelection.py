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



main()