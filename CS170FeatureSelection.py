import csv
import math
import copy
import numpy as np
import pandas as pd
import math
from datetime import datetime

def main():
    print("Type in the name of the file to test: ")
    filename = input()
    #Used https://docs.python.org/3/library/csv.html
    file = open(filename, 'r')
    s = csv.reader(file, delimiter=' ', skipinitialspace=True)
    num_features = len(next(s))
    print("Type the number of the algorithm you want to run:\n\t1) Foward Selection\n\t2) Backward Elimination\n")
    choice = int( input() )
    filecsv = pd.read_csv(filename)
    num_instances = len(filecsv)
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
    current_set_of_features = set()
    bsf_list = {}
    #Used https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
    data = df.copy(deep=True)[:-1]
    for k in range(1, num_features):
        if k == 1:
            currset_copy = copy.deepcopy(current_set_of_features)
            accuracy = leave_one_out_cross_validation(data, currset_copy, num_features)
            print('Feature set {} has accuracy ' + str(round(accuracy, 3)) + '\n')
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
        curr_set = copy.deepcopy(current_set_of_features)
        bsf_list[bsf] = curr_set
        #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(round(bsf, 3)) + '\n')
    #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
    print('Finished search!! The best feature subset is ' + str(bsf_list[bsf]) +
          ' which has an accuracy of ' + str(round(bsf, 3)) + '\n')

#same code as forward, except instead of adding features, we are just removing the irrelevant ones
def backward(df, num_features):
    data = df.copy(deep=True)[:-1]
    current_set_of_features = set()
    bsf_list = {}
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
            empty_set = set()
            currset_copy = copy.deepcopy(empty_set)
            accuracy = leave_one_out_cross_validation(data, empty_set, num_features)
            # print('Using feature(s) ' + str(currset_copy) + ' accuracy is ' + str(round(accuracy, 3)))
            print('Feature set {} has accuracy ' + str(round(accuracy, 3)) + '\n')
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
        curr_set = copy.deepcopy(current_set_of_features)
        bsf_list[bsf] = curr_set
        #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(round(bsf, 3)) + '\n')
    #to avoid spurious precision: https://www.w3schools.com/python/ref_func_round.asp
    print('Finished search!! The best feature subset is ' + str(bsf_list[bsf]) +
          ' which has an accuracy of ' + str(round(bsf, 3)) + '\n')

def leave_one_out_cross_validation(data, currset, feature_to_add):
    number_correctly_classfied = 0
    df = data.copy(deep=True)
    # https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array for easier traversal
    datadf = df.to_numpy(dtype='float')
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
main()
