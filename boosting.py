import numpy as np
import pandas as pd
import random
from pprint import pprint
import math

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
    n_unique_values_treshold = 10
    n_columns = len(df.columns)
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("nominal")
            else:
                feature_types.append("continuous")
    return feature_types

#splits the dataset into lower or upper values, or equal and not equal values based on the attribute type
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
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
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

#calculates the overall entropy for a given split data
def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy

#determines the best split for a given dataset and possible split points
#in case of nominal attributes, only un-used feature can be considered along the path of from the root to the node
def determine_best_split(data, potential_splits):
    
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

#checks the purity of data i.e whether the data belongs purely to one class
def check_purity(data):
    label_column = data[:,-1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    
#classifies the data into the class which has maximum number of class labels
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

#generates the probable split for the given data
def get_probable_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    features = list()
    #consider all the features except the last column which is the label column
    for column_index in range(n_columns - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        type_of_feature = FEATURE_TYPES[column_index]
        if type_of_feature == "continuous":
            potential_splits[column_index] = []
            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    potential_split = round(float(current_value) + float(previous_value) / 2.0)

                    potential_splits[column_index].append(potential_split)
        
        # feature is nominal         
        else:
            potential_splits[column_index] = unique_values
    
    return potential_splits

#generates a decision tree with minimum number of samples at leaf node and depth less than the provided max depth
def decision_tree_algorithm(df, counter, min_samples, max_depth):
    # data initializations
    if counter == 0:
        global COLUMN_NAMES, FEATURE_TYPES
        COLUMN_NAMES = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    # base cases
    if(len(data) == 0):
        return
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:    
        counter += 1
        #find out the probable splits
        probable_splits = get_probable_splits(data)
        split_column, split_value = determine_best_split(data, probable_splits)
        left_data, right_data = split_data(data, split_column, split_value)
        
        # determine question
        feature_name = COLUMN_NAMES[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            query = "{} <= {}".format(feature_name, split_value)
            
        # feature is nominal
        else:
            query = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub tree for the given root
        tree_node = {query: []}
        
        # find answers thorugh recursion
        left_answer = decision_tree_algorithm(left_data, counter, min_samples, max_depth)
        right_answer = decision_tree_algorithm(right_data, counter, min_samples, max_depth)
        
        #if answers are same, the leaf node is created
        if left_answer == right_answer:
            tree_node = left_answer
        else:
            tree_node[query].append(left_answer)
            tree_node[query].append(right_answer)
        
        return tree_node

    
#classifies the given test example into one of test classes using the given decision tree  
def classify_row(row, tree):
    query = list(tree.keys())[0]
    feature_name, operator, value = query.split(" ")

    #query for the data
    feature_index = int(feature_name)
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
def random_forest_predict(decision_trees, row):
    predictions = [classify_row(row, tree) for tree in decision_trees]
    return max(set(predictions), key = predictions.count)
        
        
#creates a random subsample from the dataset with replacement
def create_sub_sample(df, fraction):
    sample = list()
    total_samples = len(df)
    n_samples = round(total_samples * fraction)
    while len(sample) < n_samples:
        index = random.randrange(total_samples)
        sample.append(df.iloc[index])
    return sample

#boosting algorithm
def boosting(train, test, sample_size, n_trees, min_samples, max_depth):
    decision_trees_list = list()
    
    for index in range(n_trees):
        sample = create_sub_sample(train, sample_size)
        sample = pd.DataFrame(sample)
        decision_tree = decision_tree_algorithm(sample, 0, min_samples, max_depth)
        decision_trees_list.append(decision_tree)
    predictions = [random_forest_predict(decision_trees_list, row) for row in test]
    return predictions

# Classification and Regression Tree Algorithm
def make_prediction(tree, test):
    predictions = list()
    for row in test:
        prediction = classify_row(row, tree)
        predictions.append(prediction)
    return(predictions)

#Boosting changes to calculate error
def calculate_error(class_list, train_data, weight_column):
    sum = 0
    original_class_list = train_data[:, len(train_data[0])-1]
    for i in range(len(class_list)):
        if class_list[i] != original_class_list[i]:
            sum += weight_column[i]
    return sum

#classify test data based on weights
def classify_test_data(test_data, forest, alpha_list):
    predictions = list()
    for k in range(len(test_data)):
        weight0 = 0
        weight1 = 0
        for i in range(len(forest)):
            classifier_weight = alpha_list[i]
            row = test_data[k]
            prediction = classify_row(row, forest[i])
            if prediction == 0:
                weight0 += classifier_weight
            else:
                weight1 += classifier_weight
        if weight1 > weight0:
            predictions.append(1)
        else:
            predictions.append(0)
    return(predictions)

#calculates accuracy metrics
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
        
#evaluates random forest algorithm for 10-fold cross validation by sampling data of different sizes
def evaluate_algorithm(dataset, algorithm, n_folds, sample_size, n_trees, min_samples, max_depth):
    folds = cross_validation_split(dataset, n_folds)
    accuracy_list = list()
    precision_list = list()
    recall_list = list()
    f1_measure_list = list()
    fold_length = int(len(dataset) / n_folds)
    for i, fold in enumerate(folds):
        print ("Fold::",i+1)
        train_set = list(folds)
        remove_array(train_set, fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        train_data = np.array(train_set)
        test_data = np.array(test_set)
        start = i*fold_length
        end = start + fold_length
        test_set_ids = set(range(start, end))
        train_set_ids = set(range(int(len(dataset)))).difference(test_set_ids)
        test_set_ids = list(test_set_ids)
        train_set_ids = list(train_set_ids)
        weight_column = [1 / len(train_set_ids) for x in range(len(train_set_ids))]
        forest = []
        alpha_list = []
        for j in range (n_trees):
            error = 999
            tree = None
            while error>0.5:
                sample_train_ids = np.random.choice(train_set_ids, len(train_set_ids), replace=True, p=weight_column)
                datasetArray = np.array(dataset)
                sample_train_set = datasetArray[sample_train_ids]
                sample = pd.DataFrame(sample_train_set)
                tree = decision_tree_algorithm(sample, 0, min_samples, max_depth)
                predicted = make_prediction(tree, train_set)
                error = calculate_error(predicted, np.array(train_data), weight_column)
            alpha = (1/2)*math.log((1-error)/error)
            alpha_list.append(alpha)
            forest.append(tree)
            for k in range(len(predicted)):
                true_label = train_data[k][len(train_data[0])-1]
                predicted_label = predicted[k]
                if true_label != predicted_label:
                    weight_column[k] *= math.exp(-1 * alpha * - 1)
                else:
                    weight_column[k] *= math.exp(-1 * alpha * + 1)
            new_sum = np.sum(weight_column)
            for k in range(len(weight_column)):
                weight_column[k] /= new_sum
        actual = [row[-1] for row in fold]
        predicted = classify_test_data(test_set, forest, alpha_list)
        accuracy, precision, recall, f1_measure = accuracy_metric(actual, predicted)
        print('Accuracy %.3f%%' % accuracy + ' Precision %.3f%%'% precision + ' Recall %.3f%%' % recall + ' F1 measure %.3f%%' % f1_measure)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_measure_list.append(f1_measure)
        
    return accuracy_list, precision_list, recall_list, f1_measure_list

random.seed(0)
df = load_dataset("file-path")
DECISION_TREE_ALGO = decision_tree_algorithm
n_folds = 10
n_columns = len(df.columns)
sample_size = 0.75
min_samples = 20
max_depth = 5 
BOOSTING_ALGO = boosting
n_trees = 5
accuracy_list, precision_list, recall_list, f1_measure_list  = evaluate_algorithm(df, BOOSTING_ALGO, n_folds, sample_size, n_trees, min_samples, max_depth)
print('Mean Accuracy: %.3f%%' % (sum(accuracy_list)/float(len(accuracy_list))))
print('Mean Precision: %.3f%%' % (sum(precision_list)/float(len(precision_list))))
print('Mean recall: %.3f%%' % (sum(recall_list)/float(len(recall_list))))
print('Mean f1_measure: %.3f%%' % (sum(f1_measure_list)/float(len(f1_measure_list))))