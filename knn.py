import numpy as np

# calculates accuracy for the made predictions
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
        f1_measure = ((2) * (precision * recall) / (precision + recall))
    return accuracy, precision * 100.0, recall * 100.0, f1_measure * 100.0

# custom print method for all metrics
def print_metric(accuracy, precision, recall, f1_measure):
    print("accuracy -", accuracy)
    print("precision -", precision)
    print("recall -", recall)
    print("f1_measure -", f1_measure)

# returns true if s is a number value, else false
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# normalizes the data points between test and training set
def normalize(test_point, training_set):
    combined_set = np.vstack((test_point, training_set))
    normalize = np.atleast_1d(np.linalg.norm(combined_set, 2, 0))
    normalize[normalize==0] = 1
    normalize = combined_set / np.expand_dims(normalize, 0)
    normalized_test_point = normalize[0, :]
    normalized_training_set = normalize[1:, :]
    return normalized_test_point, normalized_training_set

# euclidean distance based on np
def euclidean_distance(point_one, point_two):
    distance = np.linalg.norm(point_one - point_two)
    return distance

# predicts the test point label based on kNN from training set
def predict_label(test_point, training_set, training_labels, K):
    distances = []
    for training_point in training_set:
        distances.append(euclidean_distance(test_point, training_point))
    distances = np.array(distances)
    K_closest_indices = np.argsort(distances)[:K] # closest indices not sorted
    K_closest_labels = training_labels[K_closest_indices]
    zero = 0
    one = 0
    for label in K_closest_labels: # vote counting by kNN
        if label == 1: one += 1
        else: zero += 1
    predicted_label = 0
    if one > zero: # set label to majority vote
        predicted_label = 1
    return predicted_label

# runs kNN algorithm with K inputand number of folds for cross validation
def kNN_cross_validation(data_file, class_labels, K, folds):
    data_partitions = np.array_split(data_file, folds)
    label_partitions = np.array_split(class_labels, folds)
    accuracy = precision = recall = f1_measure = 0 # tracks metric for all folds

    for fold in range(folds):
        test_set = data_partitions[fold] # test points
        actual_labels = label_partitions[fold] # expected labels for test points
        training_set = []
        training_labels = []
        for other_fold in range(folds): # remaining partitions as training set
            if other_fold != fold:
                training_set.extend(data_partitions[other_fold])
                training_labels.extend(label_partitions[other_fold])
        training_set = np.array(training_set)
        training_labels = np.array(training_labels)
        training_labels = training_labels.reshape(training_labels.shape[0], 1)

        predicted_labels = [] # predicted labels for test points
        for index in range(test_set.shape[0]):
            test_point = test_set[index]
            test_point = test_point.reshape(1, test_point.shape[0]) # fix shape
            n_test_point, n_training_set = normalize(test_point, training_set)
            predicted_labels.append(predict_label(n_test_point, n_training_set,
                                                            training_labels, K))

        predicted_labels = np.array(predicted_labels)
        a, p, r, f = accuracy_metric(actual_labels, predicted_labels)
        accuracy += a
        precision += p
        recall += r
        f1_measure += f
        print("\nFold:", fold + 1) # offset from starting index 0
        print_metric(a, p, r, f)
    return accuracy, precision, recall, f1_measure

# kNN start
print("KNN")
# name_of_data_file = "project3_dataset2.txt"
# K_value = 5
# folds = 10
name_of_data_file = input("Enter name of file: ")
K_value = int(input("Enter K value: "))
folds = int(input("Enter fold value for cross validation: "))
data_file = np.loadtxt(name_of_data_file, dtype="str")

class_labels = data_file[:, -1] # save last col separately
data_file = data_file[:, 0:-1] # remove last col from data

non_number_col = []
for col in range(data_file.shape[1]): # identify non number columns
    if not is_number(data_file[0][col]):
        non_number_col.append(col)
        print(col)

substitute_mapping = dict()
for col in non_number_col: # replace non number columns with 0 to 1 float value
    col_values = data_file[:, col].astype(str)
    unique_values = set()
    for value in col_values:
        unique_values.add(value)
    unique_values = list(unique_values)
    for substitute in range(len(unique_values)):
        substitute_mapping[unique_values[substitute]] = substitute
    for row in range(data_file.shape[0]):
        data_file[row][col] = substitute_mapping.get(data_file[row][col])

data_file = data_file.astype(float)
class_labels = class_labels.astype(float)

accuracy, precision, recall, f1_measure = kNN_cross_validation(data_file,
                                            class_labels, K_value, folds)
print("\nAverage:")
print_metric(accuracy/folds, precision/folds, recall/folds, f1_measure/folds)
# end
