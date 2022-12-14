'''
NOTE
for pretty much the entire program, I referenced Dr. Keogh's Skeleton code 
that was provided on the slides on Project 2 breifing linked below
https://www.dropbox.com/sh/rltooq0t3khobuj/AAA3MYkZc8gb1RLa3tNSnsrga?dl=0&preview=Project_2_Briefing.pptx
'''

import csv
import math
import copy
import numpy as np
import pandas as pd
import math
from datetime import datetime

NUM_INSTANCES = 0

def main():
    global NUM_INSTANCES
    print("Type in the name of the file to test: ")
    filename = input()
    #Used https://docs.python.org/3/library/csv.html
    file = open(filename, 'r')
    s = csv.reader(file, delimiter=' ', skipinitialspace=True)
    num_features = len(next(s))
    print("Type the number of the algorithm you want to run:\n\t1) Foward Selection\n\t2) Backward Elimination\n")
    choice = int( input() )
    filecsv = pd.read_csv(filename)
    num_instances = len(filecsv) + 1
    NUM_INSTANCES = num_instances
    print('This dataset has ' + str(num_features - 1) + ' features (not including the class attribute), with ' + str(num_instances) + ' instances.')
    #https://www.geeksforgeeks.org/how-to-read-text-files-with-pandas/
    #used read_fwf() because it said in the link above it works better with files of fixed column length, and i think that fits our file format better
    if choice == 1:
        df = pd.read_fwf(filename, header=None)
        first = datetime.now()
        forward(df, num_features)
        second = datetime.now()
        time = second-first
        if time.total_seconds() < 60:
            print('Total time taken was approximately ' + str(round(time.total_seconds(), 1)) + ' seconds!')
        else:
            seconds = time.total_seconds()
            minutes = seconds / 60
            print('Total time taken was approximately ' + str(round(minutes, 0)) + ' minutes!')
    elif choice == 2:
        df = pd.read_fwf(filename, header=None)
        first = datetime.now()
        backward(df, num_features)
        second = datetime.now()
        time = second-first
        if time.total_seconds() < 60:
            print('Total time taken was approximately ' + str(round(time.total_seconds(), 1)) + ' seconds!')
        else:
            seconds = time.total_seconds()
            minutes = seconds / 60
            print('Total time taken was approximately ' + str(round(minutes, 0)) + ' minutes!')

def forward(df, num_features):
    global NUM_INSTANCES
    current_set_of_features = set()
    accuracy_list = {}
    #Used https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
    data = df.copy(deep=True)[:-1]
    for k in range(1, num_features):
        if k == 1:
            currset_copy = copy.deepcopy(current_set_of_features)
            accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
            nums = get_max_class(data)
            default_rate = nums / NUM_INSTANCES
            print('Feature set {} has accuracy ' + str(round(default_rate, 3)) + '\n')
        bsf = 0
        featuretoadd = 0
        for j in range(1, num_features):
            if j not in current_set_of_features:
                #need to create deepcopy so its not changed later
                currset_copy = copy.deepcopy(current_set_of_features)
                currset_copy.add(j)
                accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
                # if k == 1:
                #     currset_copy = copy.deepcopy(current_set_of_features)
                #     accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
                #     # print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
                #     print('Feature set {} has accuracy ' + str(round(accuracy, 3)) + '\n')
                #     k += 1
                # else:
                print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
                if accuracy >= bsf:
                    bsf = accuracy
                    featuretoadd = j
        current_set_of_features.add(featuretoadd)
        accuracy_list[bsf] = copy.deepcopy(current_set_of_features)
        #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(round(bsf, 3)) + '\n')
    #https://datagy.io/python-get-dictionary-key-with-max-value/#:~:text=The%20simplest%20way%20to%20get,maximum%20value%20of%20any%20iterable.
    bsf = max(accuracy_list.keys())
    #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
    print('Finished search!! The best feature subset is ' + str(accuracy_list[bsf]) +
          ' which has an accuracy of ' + str(round(bsf, 3)) + '\n')

#same code as forward, except instead of adding features, we are just removing the irrelevant ones
def backward(df, num_features):
    global NUM_INSTANCES
    data = df.copy(deep=True)[:-1]
    current_set_of_features = set()
    accuracy_list = {}

    for i in range(1, num_features):
        current_set_of_features.add(i)
    for i in range(1, num_features):
        bsf = 0
        featuretoadd = 0
        if i == 1:
            currset_copy = copy.deepcopy(current_set_of_features)
            accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
            # print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
            print('Feature set ' + str(currset_copy) + ' has accuracy ' + str(round(accuracy, 3)) + '\n')
        if i == num_features - 1:
            nums = get_max_class(data)
            default_rate = nums / NUM_INSTANCES
            empty_set = set()
            currset_copy = copy.deepcopy(empty_set)
            accuracy = leave_one_out_cross_validation(data, empty_set, num_features)
            # print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
            print('Feature set {} has accuracy ' + str(round(default_rate, 3)) + '\n')
            continue
        for j in range(1, num_features):
            if j in current_set_of_features:
                #need to create deepcopy so its not changed later
                currset_copy = copy.deepcopy(current_set_of_features)
                currset_copy.remove(j)
                accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
                # if i == 1:
                #     currset_copy = copy.deepcopy(current_set_of_features)
                #     accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
                #     # print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
                #     print('Feature set ' + str(currset_copy) + ' has accuracy ' + str(round(accuracy, 3)) + '\n')
                #     i+=1
                # else:
                print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
                if accuracy >= bsf:
                    bsf = accuracy
                    featuretoadd = j
        current_set_of_features.remove(featuretoadd)
        accuracy_list[bsf] = copy.deepcopy(current_set_of_features)
        #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(round(bsf, 3)) + '\n')
    #https://datagy.io/python-get-dictionary-key-with-max-value/#:~:text=The%20simplest%20way%20to%20get,maximum%20value%20of%20any%20iterable.
    bsf = max(accuracy_list.keys())
    #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
    print('Finished search!! The best feature subset is ' + str(accuracy_list[bsf]) +
          ' which has an accuracy of ' + str(round(bsf, 3)) + '\n')

def leave_one_out_cross_validation(data, currset, feature_to_add):
    number_correctly_classfied = 0
    df = data.copy(deep=True)
    # https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
    datadf = df.to_numpy(dtype='float', na_value=np.nan)
    # Replace irrelevant features with 0
    for i in range(1, feature_to_add):
        if i not in currset:
            datadf[:, i] = 0
    for i in range(int(len(data.index))):
        object_to_classify = datadf[i][1:]
        label_object_to_classify = datadf[i][0]
       #https://stackoverflow.com/questions/7781260/how-can-i-represent-an-infinite-number-in-python
        nearest_n_dist = math.inf
        nearest_n_loc = math.inf
        for k in range(int(len(data.index))):
            if k != i:
                #Used:
                #https://www.w3schools.com/python/ref_math_sqrt.asp#:~:text=The%20math.,than%20or%20equal%20to%200.
                #https://www.geeksforgeeks.org/sum-function-python/#:~:text=Python%20provides%20an%20inbuilt%20function,of%20numbers%20in%20the%20iterable.
                #distance = sqrt(sum((object_to_classify-data(k,2:end)).^2)); (from slides)
                distance = math.sqrt(sum((object_to_classify - datadf[k][1:]) ** 2))
                if distance < nearest_n_dist:
                    nearest_n_dist = distance
                    nearest_n_loc = k
                    nearest_n_label = datadf[nearest_n_loc][0]
        if label_object_to_classify == nearest_n_label:
            number_correctly_classfied += 1
    accuracy = number_correctly_classfied / int(len(data.index))
    return accuracy

#in office hours, professor said that default rate = count of most frequent class / total number of instances
#https://www.geeksforgeeks.org/how-to-get-first-column-of-pandas-dataframe/#:~:text=Method%201%3A%20Using%20iloc%5B%5D,the%20index%20for%20first%20column.&text=where%2C,with%200%20position%20as%20index
def get_max_class(data):
    one_count = 0
    two_count = 0
    cols = data.iloc[:, 0]

    for entry in cols:
        if int(entry) == 1:
            one_count += 1
        elif int(entry) == 2:
            two_count += 1
    
    # print( max(one_count, two_count) ) + 1
    return (max(one_count, two_count)) + 1


main()
