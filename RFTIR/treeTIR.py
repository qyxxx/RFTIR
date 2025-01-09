import numpy as np
import pandas as pd
import random
import math
import sys
from lifelines import CoxTimeVaryingFitter

# for tree visualization
from graphviz import Digraph

# for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit


# calculate the value in range
def value_in_range(x, glucoserange = [-sys.maxsize, sys.maxsize]):
  if glucoserange[0] <= x and x <= glucoserange[1]:
    return 1
  else:
    return 0


# split data set basead on a feature and a value
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet.loc[dataSet[feature] <= value]
    mat1 = dataSet.loc[dataSet[feature] > value]
    return mat0, mat1

# estimators of TIR
def naive_est(x, value_in_range = 'value_in_range', patient_id = 'patient_id', time = 'time'):
    TIR = (x.groupby(patient_id)[value_in_range].mean().reset_index())[value_in_range].mean()
    return TIR

def proposed_est(x, value_in_range = 'value_in_range', patient_id = 'patient_id', time = 'time'):
    TIR = (x.groupby(time)[value_in_range].mean().reset_index())[value_in_range].mean()
    return TIR

def proposed_est_weight(dat, value_in_range = 'value_in_range', time = 'time', patient_id = 'patient_id'):
    TIR = (dat.groupby(time))[[value_in_range, 'weight']].apply(lambda x: np.average(x[value_in_range], weights = x['weight'])).mean()
    return TIR


# choose the best split
def chooseBestSplit(dataSet, candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', value_in_range = 'value_in_range', leafType = proposed_est, min_samples_split = 30, min_diff_tir = 0.001):
    # extract wide format data
    data_wide = dataSet.groupby(id).first().reset_index()
    #candidate_vars.insert(0, id)
    data_wide_cov = data_wide[candidate_vars]
    # # quit if all values are the same
    if data_wide_cov.drop_duplicates().shape[0] == 1:
        return None, None, leafType(dataSet, patient_id = id, time = time, value_in_range = value_in_range)
    # number of candidate variables
    n = len(candidate_vars)
    best_diff_tir = 0; bestIndex = 0; bestValue = 0
    for featIndex in range(n):
    #for featIndex in random.sample(range(n), math.floor(n/3)):
        for splitVal in data_wide_cov[candidate_vars[featIndex]].unique():
            mat0, mat1 = binSplitDataSet(dataSet, candidate_vars[featIndex], splitVal)
            if (len(mat0[id].unique()) < min_samples_split) or (len(mat1[id].unique()) < min_samples_split): continue
            new_diff_tir = abs(leafType(mat0, patient_id = id, time = time, value_in_range = value_in_range) - leafType(mat1, patient_id = id, time = time, value_in_range = value_in_range))
            if new_diff_tir > best_diff_tir:
                bestIndex = featIndex
                bestValue = splitVal
                best_diff_tir = new_diff_tir

    if best_diff_tir < min_diff_tir:
        return None, None, leafType(dataSet, patient_id = id, time = time, value_in_range = value_in_range)
    mat0, mat1 = binSplitDataSet(dataSet, candidate_vars[bestIndex], bestValue)
    if (len(mat0[id].unique()) < min_samples_split) or (len(mat1[id].unique()) < min_samples_split):
        return None, None, leafType(dataSet, patient_id = id, time = time, value_in_range = value_in_range)

    return best_diff_tir, bestIndex, bestValue

# choose the best split with random feature selection
def chooseBestSplitRand(dataSet, candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', value_in_range = 'value_in_range', leafType = proposed_est, min_samples_split = 30, min_diff_tir = 0.001, max_features = 0.3):
    # extract wide format data
    data_wide = dataSet.groupby(id).first().reset_index()
    #candidate_vars.insert(0, id)
    data_wide_cov = data_wide[candidate_vars]
    # # quit if all values are the same
    if data_wide_cov.drop_duplicates().shape[0] == 1:
        return None, None, leafType(dataSet, patient_id = id, time = time, value_in_range = value_in_range)
    # number of candidate variables
    n = len(candidate_vars)
    best_diff_tir = 0; bestIndex = 0; bestValue = 0
    for featIndex in random.sample(range(n), max(1, int(max_features * n))):
        for splitVal in data_wide_cov[candidate_vars[featIndex]].unique():
            mat0, mat1 = binSplitDataSet(dataSet, candidate_vars[featIndex], splitVal)
            if (len(mat0[id].unique()) < min_samples_split) or (len(mat1[id].unique()) < min_samples_split): continue
            new_diff_tir = abs(leafType(mat0, patient_id = id, time = time, value_in_range = value_in_range) - leafType(mat1, patient_id = id, time = time, value_in_range = value_in_range))
            if new_diff_tir > best_diff_tir:
                bestIndex = featIndex
                bestValue = splitVal
                best_diff_tir = new_diff_tir

    if best_diff_tir < min_diff_tir:
        return None, None, leafType(dataSet, patient_id = id, time = time, value_in_range = value_in_range)
    mat0, mat1 = binSplitDataSet(dataSet, candidate_vars[bestIndex], bestValue)
    if (len(mat0[id].unique()) < min_samples_split) or (len(mat1[id].unique()) < min_samples_split):
        return None, None, leafType(dataSet, patient_id = id, time = time, value_in_range = value_in_range)

    return best_diff_tir, bestIndex, bestValue


# create a tree
def createTree(dataSet, candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', value_in_range = 'value_in_range', leafType = proposed_est, min_samples_split = 30, min_diff_tir = 0.001, depth = 0, max_depth = 5):
    depth += 1
    best_diff_tir, featIndex, splitVal = chooseBestSplit(dataSet, candidate_vars = candidate_vars, id = id, time = time, value_in_range = value_in_range, leafType = leafType, min_samples_split=min_samples_split, min_diff_tir=min_diff_tir)
    if featIndex == None: 
        return splitVal
    retTree = {}
    retTree['spInd'] = featIndex
    retTree['spVar'] = candidate_vars[featIndex]
    retTree['spVal'] = splitVal
    retTree['depth'] = depth
    retTree['diff_tir'] = best_diff_tir
    lSet, rSet = binSplitDataSet(dataSet, candidate_vars[featIndex], splitVal)
    if depth < max_depth:
        retTree['left'] = createTree(lSet, candidate_vars = candidate_vars, id = id, time = time, value_in_range = value_in_range, leafType = leafType, min_samples_split=min_samples_split, min_diff_tir=min_diff_tir, depth = depth, max_depth = max_depth)
        retTree['right'] = createTree(rSet, candidate_vars = candidate_vars, id = id, time = time, value_in_range = value_in_range, leafType = leafType, min_samples_split=min_samples_split, min_diff_tir=min_diff_tir, depth = depth, max_depth = max_depth)
        return retTree
    retTree['left'] = leafType(lSet, patient_id = id, time = time, value_in_range = value_in_range)
    retTree['right'] = leafType(rSet, patient_id = id, time = time, value_in_range = value_in_range)
    return retTree


# create a tree with random feature selection
def createTreeRand(dataSet, candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', value_in_range = 'value_in_range', leafType = proposed_est, min_samples_split = 30, min_diff_tir = 0.001, depth = 0, max_depth = 5, max_features = 0.3):
    depth += 1
    best_diff_tir, featIndex, splitVal = chooseBestSplitRand(dataSet, candidate_vars = candidate_vars, id = id, time = time, value_in_range = value_in_range, leafType = leafType, min_samples_split= min_samples_split, min_diff_tir = min_diff_tir, max_features = max_features)
    if featIndex == None: 
        return splitVal
    retTree = {}
    retTree['spInd'] = featIndex
    retTree['spVar'] = candidate_vars[featIndex]
    retTree['spVal'] = splitVal
    retTree['depth'] = depth
    retTree['diff_tir'] = best_diff_tir
    lSet, rSet = binSplitDataSet(dataSet, candidate_vars[featIndex], splitVal)
    if depth < max_depth:
        retTree['left'] = createTreeRand(lSet, candidate_vars = candidate_vars, id = id, time = time, value_in_range = value_in_range, leafType = leafType, min_samples_split= min_samples_split, min_diff_tir = min_diff_tir, depth = depth, max_depth=max_depth, max_features = max_features)
        retTree['right'] = createTreeRand(rSet, candidate_vars = candidate_vars, id = id, time = time, value_in_range = value_in_range, leafType = leafType, min_samples_split= min_samples_split, min_diff_tir = min_diff_tir, depth = depth, max_depth=max_depth, max_features = max_features)
        return retTree
    retTree['left'] = leafType(lSet, patient_id = id, time = time, value_in_range = value_in_range)
    retTree['right'] = leafType(rSet, patient_id = id, time = time, value_in_range = value_in_range)
    return retTree


# bootstrap sampling
def bootstrap_sample(data_long, id = 'patient_id'):
    patient_id = data_long[id].unique()
    n = len(patient_id)
    bootstrap_patient_id = np.random.choice(patient_id, size = n, replace = True)
    bootstrap_sample = data_long.loc[data_long[id].isin(bootstrap_patient_id)]
    return bootstrap_sample


# create a forest
def createForest(n_trees = 10, data = np.zeros(5), candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', glucose = 'glucose', glucoserange = [70, 180], leafType = proposed_est, min_samples_split = 5, min_diff_tir = 0.001, depth = 0, max_depth = 5, max_features = 0.3):
    forest = []
    data['TIR_RF_value_in_range'] = data[glucose].apply(lambda x: value_in_range(x, glucoserange = glucoserange))
    for _ in range(n_trees):
        boot_sample = bootstrap_sample(data, id = id)
        tree = createTreeRand(boot_sample, candidate_vars = candidate_vars, id = id, time = time, value_in_range = 'TIR_RF_value_in_range', leafType = leafType, min_samples_split= min_samples_split, min_diff_tir = min_diff_tir, depth = depth, max_depth = max_depth, max_features = max_features)
        forest.append((tree))
    return forest

def createForest_cox(n_trees = 10, data = np.zeros(5), candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', glucose = 'glucose', glucoserange = [70, 180],
                leafType = proposed_est_weight, period = 5, formula = 'var1',
                min_samples_split = 5, min_diff_tir = 0.001, depth = 0, max_depth = 5, max_features = 0.3):
    forest = []
    data['TIR_RF_value_in_range'] = data[glucose].apply(lambda x: value_in_range(x, glucoserange))
    # prepare data for cox model
    data['event'] = False
    data.loc[data.groupby(id)[time].idxmax(), 'event'] = True
    data['time2'] = data[time] + period
    #fit cox model with time dep covariate
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(data, id_col = id, event_col= 'event', start_col=time, stop_col='time2', formula=formula)
    cumulative_hazard = ctv.baseline_cumulative_hazard_
    cumulative_hazard[time] = cumulative_hazard.index
    cumulative_hazard['hazard diff'] = cumulative_hazard['baseline hazard'].diff().fillna(cumulative_hazard['baseline hazard'])
    data = data.reset_index(drop=True)
    data['predict partial hazard'] = ctv.predict_partial_hazard(data).reset_index(drop=True)
    data = data.merge(cumulative_hazard, on = time, how = 'left').fillna(value = 0)
    data['lambda_exp_diff'] = data['hazard diff'] * data['predict partial hazard']
    data['cum_lambda_exp_diff'] = data.groupby([id])['lambda_exp_diff'].cumsum()
    data['weight'] = 1/(np.exp(- data['cum_lambda_exp_diff']))

    for _ in range(n_trees):
        boot_sample = bootstrap_sample(data, id = id)
        tree = createTreeRand(boot_sample, candidate_vars = candidate_vars, id = id, time = time, value_in_range = 'TIR_RF_value_in_range', leafType = leafType, min_samples_split= min_samples_split, min_diff_tir = min_diff_tir, depth = depth, max_depth = max_depth, max_features = max_features)
        forest.append((tree))
    return forest


# test if the input is a tree
def isTree(obj):
    return (type(obj).__name__ == 'dict')

def regTreeEval(model, inDat):
    return float(model)


# tree prediction - single data point
def treeForeCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spVar']] <= tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

# tree prediction  - multiple data points
def TreePredict(tree, testData, modelEval = regTreeEval):
    if isinstance(testData, pd.Series): testData = testData.to_frame().T
    m = len(testData)
    TIR = np.zeros(m)
    for i in range(m):
        TIR[i] = treeForeCast(tree, testData.iloc[i], modelEval)
    return TIR


# train test split
def train_test_split(data, test_size = 0.2, id = 'patient_id'):
    patient_id = data[id].unique()
    n = len(patient_id)
    test_patient_id = np.random.choice(patient_id, size = math.floor(n*test_size), replace = False)
    data_train = data.loc[~data[id].isin(test_patient_id)]
    data_test = data.loc[data[id].isin(test_patient_id)]
    return data_train, data_test


# tree prediction error
def predict_error(tree, testdata, l = 50, nl = 50, glucose = 'glucose', glucoserange = [70, 180], candidate_vars = ['rand_group', 'sex', 'age', 'obese'], modelEval = regTreeEval, id = 'patient_id', time = 'time', esttype = proposed_est):
    patient_id = testdata[id].unique()
    n = len(patient_id)
    predict_error = np.zeros(l)
    for i in range(l):
        test_patient_id = np.random.choice(patient_id, size = nl, replace = False)
        subsample = testdata.loc[testdata[id].isin(test_patient_id)]
        meanTIR_subsample = esttype(subsample, patient_id = id, time = time, glucose = glucose, glucoserange = glucoserange)
        subsample_wide = subsample.groupby(id).first().reset_index()
        subsample_cov = subsample_wide[candidate_vars]
        subsample_meanpred = TreePredict(tree, subsample_cov, modelEval).mean()
        predict_error[i] = abs(meanTIR_subsample - subsample_meanpred)
    return predict_error.mean()


# plot tree
def visualize_tree(node, parent=None, edge_label='', dot=None):
    if dot is None:
        dot = Digraph()
        dot.attr('node', shape='box')
    
    # Generate a unique identifier for each node based on its memory id
    node_id = str(id(node))

    if isinstance(node, dict):
        # Intermediate node
        label = f"{node['spVar']} <= {node['spVal']}"
        dot.node(node_id, label)
        if parent is not None:
            dot.edge(str(id(parent)), node_id, label=edge_label)
        
        # Recursively visualize the left and right child nodes
        visualize_tree(node['left'], node, 'True', dot)
        visualize_tree(node['right'], node, 'False', dot)
    else:
        # Leaf node
        label = f"mean TIR: {round(node * 100,2)}%"
        dot.node(node_id, label)
        if parent is not None:
            dot.edge(str(id(parent)), node_id, label=edge_label)

    return dot

# calculate variable importance (depth)
def calculate_variable_importance(node, variables, current_depth=1, scores=None):
    if scores is None:
        scores = {var: 0 for var in variables}
    
    if isinstance(node, dict) and 'spVar' in node:
        # Update the score if the current variable is one of the specified variables
        if node['spVar'] in variables:
            scores[node['spVar']] += (2 ** (-current_depth)) * node['diff_tir']
        
        # Recursively process the left and right children
        if 'left' in node:
            calculate_variable_importance(node['left'], variables, current_depth + 1, scores)
        if 'right' in node:
            calculate_variable_importance(node['right'], variables, current_depth + 1, scores)
    
    return scores


# forest prediction
def ForestPredict(forest, testData, modelEval = regTreeEval):
    if isinstance(testData, pd.Series): testData = testData.to_frame().T
    m = len(testData)
    preds = np.zeros(m)
    for tree in forest:
        preds += TreePredict(tree, testData, modelEval)

    return preds / len(forest)


# forest prediction error
def predict_error_forest(forest, testdata, l = 50, nl = 50, glucose = 'glucose', glucoserange = [70, 180], candidate_vars = ['rand_group', 'sex', 'age', 'obese'], modelEval = regTreeEval, id = 'patient_id', time = 'time', esttype = proposed_est):
    patient_id = testdata[id].unique()
    n = len(patient_id)
    predict_error = np.zeros(l)
    for i in range(l):
        test_patient_id = np.random.choice(patient_id, size = nl, replace = False)
        subsample = testdata.loc[testdata[id].isin(test_patient_id)]
        meanTIR_subsample = esttype(subsample, patient_id = id, time = time, glucose = glucose, glucoserange = glucoserange)
        subsample_wide = subsample.groupby(id).first().reset_index()
        subsample_cov = subsample_wide[candidate_vars]
        subsample_meanpred = ForestPredict(forest, subsample_cov, modelEval).mean()
        predict_error[i] = abs(meanTIR_subsample - subsample_meanpred)
    return predict_error.mean()

# def calculate_variable_importance for forest
# the variable importance is the average of the variable importance of all trees in the forest
def calculate_variable_importance_forest(forest, variables):
    # Initialize a dictionary to store the total importance scores
    total_scores = {var: 0 for var in variables}
    
    # Calculate the total importance scores for each variable
    for tree in forest:
        scores = calculate_variable_importance(tree, variables)
        for var in variables:
            total_scores[var] += scores[var]
    
    # Calculate the average importance scores
    n_trees = len(forest)
    avg_scores = {var: total_scores[var] / n_trees for var in variables}
    
    return avg_scores

def predict_error_lm(model, testdata, l = 50, nl = 50, glucose = 'glucose', glucoserange = [70, 180], candidate_vars = ['rand_group', 'sex', 'age', 'obese'], id = 'patient_id', time = 'time', esttype = naive_est):
    # note Mar 26 change esttype to proposed est
    patient_id = testdata[id].unique()
    n = len(patient_id)
    predict_error = np.zeros(l)
    for i in range(l):
        test_patient_id = np.random.choice(patient_id, size = nl, replace = False)
        subsample = testdata.loc[testdata[id].isin(test_patient_id)]
        meanTIR_subsample = esttype(subsample, patient_id = id, time = time, glucose = glucose, glucoserange = glucoserange)
        subsample_wide = subsample.groupby(id).first().reset_index()
        subsample_meanpred = model.predict(subsample_wide[candidate_vars]).mean()
        predict_error[i] = abs(meanTIR_subsample - subsample_meanpred)
    return predict_error.mean()



# cross validation
def patient_level_shufflesplit_validation(data, id='patient_id', n_splits = 10, test_size=0.3, random_state=0):
    """
    :param data: pandas DataFrame containing the dataset with patient IDs.
    :param id: string, the column name in `data` containing patient IDs.
    :param n_splits: int, number of splits for the cross-validation.
    :param test_size: float, proportion of the dataset to include in the test split.
    :return: A generator that yields train-test splits.
    """
    # Ensure the patient_id column is present
    if id not in data.columns:
        raise ValueError(f"Column {id} not found in data")

    # Extract unique patient IDs
    unique_patients = data[id].unique()
    #kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    # Generate indices for patient-level splits
    for train_indices, test_indices in ss.split(unique_patients):
        train_patients = unique_patients[train_indices]
        test_patients = unique_patients[test_indices]

        # Select data corresponding to train/test patients
        train_data = data[data[id].isin(train_patients)]
        test_data = data[data[id].isin(test_patients)]

        yield train_data, test_data


def patient_level_cross_validation(data, id='patient_id', n_splits=5):
    """
    Perform patient-level cross-validation.

    :param data: pandas DataFrame containing the dataset with patient IDs.
    :param id: string, the column name in `data` containing patient IDs.
    :param n_splits: int, number of folds for the cross-validation.
    :return: A generator that yields train-test splits.
    """
    # Ensure the patient_id column is present
    if id not in data.columns:
        raise ValueError(f"Column {id} not found in data")

    # Extract unique patient IDs
    unique_patients = data[id].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

    # Generate indices for patient-level splits
    for train_indices, test_indices in kf.split(unique_patients):
        train_patients = unique_patients[train_indices]
        test_patients = unique_patients[test_indices]

        # Select data corresponding to train/test patients
        train_data = data[data[id].isin(train_patients)]
        test_data = data[data[id].isin(test_patients)]

        yield train_data, test_data

