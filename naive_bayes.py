import csv
import random
import math

import numpy as np

#calculates the accuracy metrics for the made predictions
def accuracy_metric(actual, predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(actual)):
        if actual[i] == 1 and  predicted[i] == 1:
            tp += 1
        elif actual[i] == 0 and predicted[i] == 1:
            fp += 1
        elif actual[i] == 0 and predicted[i] == 0:
            tn += 1
        elif actual[i] == 1 and predicted[i] == 0:
            fn += 1
    accuracy = (tp + tn) / float(len(actual)) * 100.0
    if tp == 0:
        precision = 0
        recall = 0
        f1_measure = 0
    else:
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        f1_measure = ((2)*(precision * recall) / (precision + recall))   
    return accuracy, precision * 100.0, recall * 100.0, f1_measure*100.0

#checks whether the given value is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#loads the dataset from the file in dataframe format
def load_dataset(filename):
    data = []
    with open(filename) as f:
        for line in f:
            inner_list = []
            for val in line.split('\t'):
                if is_number(val):
                    inner_list.append(float(val))
                else:
                    inner_list.append(val)
            data.append(inner_list)
    return data

#separates the data by class i.e assign all rows belonging to one class
def separate_data_by_class(dataset):
    separated_data = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if (row[-1] not in separated_data):
            separated_data[row[-1]] = []
        separated_data[row[-1]].append(row)
    return separated_data

def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#counts the value of an attribute in the attribute column
def group_values_per_attribute(column_index, attribute):
    #check the type of feature in the feature types
    if ATTRIBUTE_TYPES[column_index] == "continuous":
        result = []
        result.append(mean(attribute))
        result.append(stdev(attribute))
        return result
    else:
        nominal_map = {}
        for value in attribute:
            if value not in nominal_map:
                nominal_map[value] = 1
            else:
                nominal_map[value] += 1
        return nominal_map
    
#groups attribute values by their count in the attribute column
def group_by_attribute_value(dataset):
    attribute_data = zip(*dataset)
    index = 0
    groups = []
    for column_index,attribute in enumerate(zip(*dataset)):
        groups.append(group_values_per_attribute(column_index, attribute))
    del groups[-1]
    return groups

#groups data by class value and attribute value by its count in the attribute column
def naive_bayes(dataset):
    separated_data = separate_data_by_class(dataset)
    total_count_by_class = {}
    class_prob = {}
    total_data_count = 0
    for key,val in separated_data.items():
        total_count_by_class[key] = len(val)
        total_data_count += len(val)
    for key,val in total_count_by_class.items():
        class_prob[key] = float(total_count_by_class[key]) / float(total_data_count)
    group_by_class_data = {}
    for class_value, rows in separated_data.items():
        group_by_class_data[class_value] = group_by_attribute_value(rows)
    return group_by_class_data, total_count_by_class, class_prob

#calculates the probability of the given value in the given attribute
def calculate_probability(test_val, mean, stdev):
    exponent = math.exp(-(math.pow(test_val-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries_by_class, test_row, total_count_by_class, class_prob):
    probabilities = {}
    #calculate the prior probability for this class
    #create a global map for whole dataset grouping by class
    #use the group all data to calculate the prior probablity of class
    total_prob = 0
    test_row_columns = len(test_row)
    for class_value, class_grouping in summaries_by_class.items():
        probabilities[class_value] = 1
        prob_test_val = 1.0
        for i in range(len(class_grouping)):
            #check if map or list
            class_item = class_grouping[i]
            test_val = test_row[i]
            if i == test_row_columns-1:
                break
            if isinstance(class_item, dict):
                test_val_count = 0
                if test_val in class_item:
                    test_val_count = class_item[test_val]
                    prob_test_val *= float(test_val_count) / float(total_count_by_class[class_value])
                else:
                    prob_test_val = 0
            else:
                mean,stdev = class_item
                prob_test_val *= calculate_probability(test_val, mean, stdev)     
        probabilities[class_value] = (float(class_prob[class_value])* float(prob_test_val))
        total_prob += probabilities[class_value]
    #return the normalized class probabilities
    for key,val in probabilities.items():
        probabilities[key] = float(val)/float(total_prob)
    return probabilities

#make prediction for a given test row
def predict(summaries_by_class, test_row, total_count_by_class, class_prob):
    probabilities = calculate_class_probabilities(summaries_by_class, test_row, total_count_by_class, class_prob)
    best_class, highest_prob = None, -1
    for class_value, prob in probabilities.items():
        if best_class is None or prob > highest_prob:
            highest_prob = prob
            best_class = class_value
    return best_class

#makes predictions using naive bayes posterior and prior probabilities
def get_predictions(summaries_by_class, test_set, total_count_by_class, class_prob):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries_by_class, test_set[i], total_count_by_class, class_prob)
        predictions.append(result)
    return predictions
 
#determines the type of each attribute
def determine_type_of_attribute(dataset):
    
    feature_types = []
    n_unique_values_threshold = 10
    n_columns = len(dataset[0])
    #exclude the label column
    dataset = np.asarray(dataset)
    for index in range(n_columns):
        unique_values = np.unique(dataset[:,index])
        example_value = unique_values[0]
        if (len(unique_values) <= n_unique_values_threshold):
                feature_types.append("nominal")
        else:
            feature_types.append("continuous")
    return feature_types

# splits a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

#evaluates naive bayes  algorithm for 10-fold cross validation by sampling data of different sizes
def evaluate_algorithm(dataset, algorithm, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    precision_list = list()
    recall_list = list()
    f1_list = list()
    start = 0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set,[])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
        #send test as list of lists
        summaries_by_class, total_count_by_class, class_prob = algorithm(train_set)
        predictions = get_predictions(summaries_by_class, test_set, total_count_by_class, class_prob)
        actual = [row[-1] for row in fold]
        accuracy, precision, recall, f1_measure = accuracy_metric(actual, predictions)
        start += 1
        print('Accuracy %.3f%%' % accuracy + ' Precision %.3f%%'% precision + ' Recall %.3f%%' % recall + ' F1 measure %.3f%%' % f1_measure)
        scores.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_measure)
    return scores, precision_list, recall_list, f1_list
random.seed(1)
dataset = load_dataset("filepath")
n_folds = 10
global ATTRIBUTE_TYPES
ATTRIBUTE_TYPES = determine_type_of_attribute(dataset)
NAIVE_BAYES_ALGO = naive_bayes
scores, precision_list, recall_list, f1_list = evaluate_algorithm(dataset, NAIVE_BAYES_ALGO, n_folds)
print("Scores: %s" % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print('Mean Precision: %.3f%%' % (sum(precision_list)/float(len(precision_list))))
print('Mean Recall: %.3f%%' % (sum(recall_list)/float(len(recall_list))))
print('Mean F1 measure: %.3f%%' % (sum(f1_list)/float(len(f1_list))))