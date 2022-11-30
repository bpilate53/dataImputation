import argparse
import matplotlib.pyplot as plot
import miceforest as mice
import pandas as pd
import numpy as np
import scipy as sp
import sklearn as skl
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import tensorflow as tf
from datetime import datetime 

import sklearn.neighbors._base 
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

def knnImputationMethod (full_data):
    print("Running KNN Algorithm")
    dataset = full_data.copy()
    imputer = KNNImputer(n_neighbors=5)
   
    start_time = datetime.now()
    knn_imputed_data = imputer.fit_transform(dataset)
    end_time = datetime.now()
    
    print("kNN Time: " , end_time-start_time)
    return knn_imputed_data

def miceImputationMethod(full_data):
    print("Running Mice Algorithm")
    dataset = full_data.copy()
    mice_imputed_data = mice.ampute_data(dataset,perc=0.20)
    start_time = datetime.now()
    # mf.ImputationKernel
    # Create kernel. 
    kds = mice.ImputationKernel(
        mice_imputed_data,
  save_all_iterations=True,)
    # Run the MICE algorithm for 3 iterations
    kds.mice(3)
    end_time = datetime.now()
    
    print("Mice algorithm Time: " , end_time-start_time)
    
    # Return the completed kernel data
    mice_complete_data = kds.complete_data()
    return mice_complete_data

def missForestMethod(full_data):
    print("Running MissForest Algorithm")
    dataset = full_data.copy()
    imputer = MissForest()
    
    start_time = datetime.now()
    x_MissForest_imputed_data = imputer.fit_transform(X=full_data)
    end_time = datetime.now()
    
    print("MissForest Algorithm Time: " , end_time-start_time)
    
    return x_MissForest_imputed_data

def iterativeImputerMethod(full_data):
    print("Running Bayesian Ridge Algorithm")
    dataset = full_data.copy()
    start_time = datetime.now()
    iterativeImputer_imputed_data = IterativeImputer().fit_transform(dataset)
    end_time = datetime.now()
    
    print("Bayesian Ridge Algorithm Time: " , end_time-start_time)
    return iterativeImputer_imputed_data

def extraTreesRegressorMethod(full_data):
    print("Running ExtraTreesRegressor Algorithm")
    dataset = full_data.copy()
    estimator_rf = ExtraTreesRegressor(n_estimators=10, random_state=0)
    
    start_time = datetime.now()
    x_rf_imputed_data = IterativeImputer(estimator=estimator_rf, random_state=0, max_iter=50).fit_transform(dataset)
    end_time = datetime.now()
    
    print("Extra Trees Regressor Time: " , end_time-start_time) 
    return x_rf_imputed_data

def adaBoostRegressorMethod(full_data):
    print("Running AdaBoost Algorithm")
    dataset = full_data.copy()
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    
    start_time = datetime.now()
    regr_imputed_data = IterativeImputer(estimator=regr, random_state=0).fit_transform(dataset)
    end_time = datetime.now()
    
    print("AdaBoost Algorithm Time: " , end_time-start_time) 

    return regr_imputed_data

# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("algorithm", choices=['knn', 'mice', 'missForest', 'iterativeImputer', 'ada', 'extratrees'])
parser.add_argument("column", choices=[5, 10, 15], type=int)
parser.add_argument("percentage", choices=[10, 20], type=int)
# Read arguments from command line
print (parser.parse_args())
arg = parser.parse_args()

full_data = pd.read_csv('Real-Time_Road_Conditions.csv')
null_data = pd.DataFrame()

if (arg.column ==5 and arg.percentage ==10):
    null_data = pd.read_csv('data_selected_indexes_in_5_columns_0.2_rows_0.1_0.6959182615372935')
elif (arg.column ==5 and arg.percentage ==20):
    null_data = pd.read_csv('data_selected_indexes_in_5_columns_0.2_rows_0.2_0.26174381293823246')

percentage_of_rows_to_select = 0.1
total_rows_to_select = int(np.floor(full_data.count()[0]*percentage_of_rows_to_select))
full_data = full_data[:total_rows_to_select]

null_data.drop(columns=['Unnamed: 0'], inplace=True)
null_data.head()

full_columns = full_data.columns
null_columns = null_data.columns


# full data with null datum in selected columns
for column in full_columns:
    if column in null_columns:
        full_data[column] = null_data[column]

## removing id, sensor_id, location 
full_data.drop(columns=['id', 'sensor_id', 'Location', 'timestamp'], inplace=True)

columns_to_encode = ['Location Name', 'condition_text_displayed', 'condition_text_measured', 'grip_text']
full_data = pd.get_dummies(full_data, columns=['Location Name', 'condition_text_displayed', 'condition_text_measured', 'grip_text'])

if(arg.algorithm =='knn'):
    knnImputationMethod(full_data)
if(arg.algorithm == 'mice'):
    miceImputationMethod(full_data)
if(arg.algorithm == 'missForest'):
    missForestMethod(full_data)
if(arg.algorithm == 'iterativeImputer'):
    iterativeImputerMethod(full_data)
if(arg.algorithm == 'extratrees'):
    extraTreesRegressorMethod(full_data)
if(arg.algorithm == 'ada'):
    adaBoostRegressorMethod(full_data)
