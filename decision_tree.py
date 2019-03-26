import numpy as np
import pandas as pd

import random
from pprint import pprint

#calculates accuracy for the made predictions
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

#checks whether the given value is digit or float
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
    df = pd.DataFrame(data)
    column_names = list()
    for index in range(len(df.columns)-1):
        column_names.append(str(index))
    column_names.append("label")
    df.columns = column_names
    return df

#determines the type of feature present in the dataframe
def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_threshold = 10
    n_columns = len(df.columns)
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
                feature_types.append("nominal")
            else:
                feature_types.append("continuous")
    return feature_types

#splits the dataset into lower or upper values, or equal and not equal values based on the attribute type
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    #feature is continuous
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    # feature is nominal   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above

#calculates entropy of the splitted data
def calculate_entropy(data):
    
    label_column = data[:, -1]
    values, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

#calculates the overall entropy for a given split data
def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    total_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return total_entropy

#determines the best split for a given dataset and possible split points
#in case of nominal attributes, only un-used feature can be considered along the path of from the root to the node
def determine_best_split(data, possible_column_indices): 
    overall_entropy = 999
    for column_index in possible_column_indices:
        for value in possible_column_indices[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

#checks if the data belongs completely to one class
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    
#classfies the data into one class
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

#generates the potential split for the given data
def get_probable_splits(data, used_nominal_features):
    probable_splits = {}
    n_rows, n_columns = data.shape
    #consider all columns except label column which is the last column
    for column_index in range(n_columns - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        type_of_feature = FEATURE_TYPES[column_index]
        if type_of_feature == "continuous":
            probable_splits[column_index] = []
            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    probable_split = round(float(current_value) + float(previous_value) / 2.0)
                    probable_splits[column_index].append(probable_split)
        # feature is nominal         
        else:
            if column_index not in used_nominal_features:
                values_list = list()
                for val in unique_values:
                    if val not in used_nominal_features:
                        values_list.append(val)
                values_list_arr = np.asarray(values_list)
                probable_splits[column_index] = values_list_arr
    return probable_splits

#generates a decision tree with minimum number of samples at leaf node and depth less than the provided max depth
def decision_tree_algorithm(df, counter, min_samples, max_depth, used_nominal_features):
    # data initializations
    if counter == 0:
        global COLUMN_NAMES, FEATURE_TYPES
        COLUMN_NAMES = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    # base cases or len(used_nominal_features) == total_features-1
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    
    else:    
        counter += 1
        #find out the probable splits
        probable_splits = get_probable_splits(data, used_nominal_features)
        #if split column is nominal, do not use it again in building this path of the tree
        split_column, split_value = determine_best_split(data, probable_splits)
        left_data, right_data = split_data(data, split_column, split_value)
        
        #check the type of question based on attribute type
        feature_name = COLUMN_NAMES[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        feature_index = split_column
        if type_of_feature == "continuous":
            query = "{} <= {}".format(feature_name, split_value)
            
        # feature is nominal
        else:
            query = "{} = {}".format(feature_name, split_value)
            used_nominal_features.add(feature_index)
        
        # instantiate sub tree for the given root
        tree_node = {query: []}
        #check the size of left data
        if len(left_data) == 0 or len(right_data) == 0:
            if len(left_data) == 0:
                correct_data = right_data
            else:
                correct_data = left_data
            left_answer = right_answer = classify_data(correct_data)
            tree_node[query].append(left_answer)
            tree_node[query].append(right_answer)
            return tree_node
        
        #check the size of left data and right data
        if len(left_data) <= min_samples:
            left_answer = classify_data(left_data)
        else:
            left_answer = decision_tree_algorithm(left_data, counter, min_samples, max_depth, used_nominal_features)
        
        if len(right_data) <= min_samples:
            right_answer = classify_data(right_data)
        else:
            right_answer = decision_tree_algorithm(right_data, counter, min_samples, max_depth, used_nominal_features)
        
        #if answers are same, the leaf node is created
        if left_answer == right_answer:
            tree_node = left_answer
        else:
            tree_node[query].append(left_answer)
            tree_node[query].append(right_answer)
        if type_of_feature == "nominal" and len(used_nominal_features) != 0:
            used_nominal_features.remove(feature_index)
        return tree_node

#classifies the given test example into one of test classes using the given decision tree  
def classify_row(row, tree):
    query = list(tree.keys())[0]
    feature_name, operator, value = query.split(" ")
    feature_index = int(feature_name)
    #query for the data
    if operator == "<=":
        if row[feature_index] <= float(value):
            answer = tree[query][0]
        else:
            answer = tree[query][1]
    # feature is nominal
    else:
        if str(row[feature_index]) == value:
            answer = tree[query][0]
        else:
            answer = tree[query][1]
    # found answer
    if not isinstance(answer, dict):
        return answer
    # recurse to find the answer
    else:
        answer_tree = answer
        return classify_row(row, answer_tree)

# Split a dataframe into k folds
def cross_validation_split(dataframe, n_folds):
    dataframe_split = list()
    dataframe_copy = dataframe

    fold_size = int(len(dataframe) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            indices = random.randrange(len(dataframe_copy))
            fold.append(dataframe_copy.iloc[indices])
            dataframe_copy=dataframe_copy.drop(dataframe_copy.index[indices])
        dataframe_split.append(fold)
    return dataframe_split

#removes a numpy array from a list of numpy arrays
def remove_array(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')
        
#makes prediction for the random forest algorithm from the trees generated
def make_predictions(decision_tree, test):
    predictions = [classify_row(row, decision_tree) for row in test]
    return predictions

def evaluate_algorithm(dataset, algorithm, n_folds, min_samples,max_depth):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    precision_list = list()
    recall_list = list()
    f1_list = list()
    start = 0
    used_nominal_features = set()
    for fold in folds:
        start += 1
        train_set = list(folds)
        remove_array(train_set, fold)
        train_set = sum(train_set,[])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
        train_df = pd.DataFrame(train_set)
        column_names = []
        tree = algorithm(train_df, 0, min_samples, max_depth, used_nominal_features)
        predictions = make_predictions(tree, test_set)
        actual = [row[-1] for row in fold]
        accuracy, precision, recall, f1_measure = accuracy_metric(actual, predictions)
        print("Fold ", start)
        print('Accuracy %.3f%%' % accuracy + ' Precision %.3f%%'% precision + ' Recall %.3f%%' % recall + ' F1 measure %.3f%%' % f1_measure)
        scores.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_measure)
    return scores, precision_list, recall_list, f1_list

random.seed(0)
df = load_dataset("filepath")
DECISION_TREE_ALGO = decision_tree_algorithm
n_folds = 10
min_samples = 20
max_depth = 4
scores, precision_list, recall_list, f1_list = evaluate_algorithm(df, DECISION_TREE_ALGO, n_folds, min_samples, max_depth)
print("Scores: %s" % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print('Mean Precision: %.3f%%' % (sum(precision_list)/float(len(precision_list))))
print('Mean Recall: %.3f%%' % (sum(recall_list)/float(len(recall_list))))
print('Mean F1 measure: %.3f%%' % (sum(f1_list)/float(len(f1_list))))