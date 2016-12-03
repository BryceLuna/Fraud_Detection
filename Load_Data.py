import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

'''
Notes:
-Consider transforming varibles to be on same scale (depends on algo)
-Consider binning non-continuous variables
-Split in the same way you did for NLP
-Think about undersampling instead of over-sampling or giving more weight to minority class
-Try rounding if over-sampling minority class
-Consider using LogisticRegressionCV for searching over Cs
-Consider using class_weight and an intercept to avoid scaling and
 resampling for Logistic Regression
'''

def split_data(df):
    '''
    returns training and testing datasets
    '''
    y = df['acct_type']
    X = df.drop(['acct_type'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test

def resample_data(X, y, categorical_lst):
    '''
    up-samples minority class
    '''
    sm = SMOTE(kind='regular')
    X_train_re, y_train_re = sm.fit_sample(X,y)
    #rounding categorical variables
    X_train_re[:,categorical_lst] = np.round(X_train_re[:,categorical_lst])
    return X_train_re, y_train_re

def standardize_variables(X_train, X_test, numerical_lst):
    '''
    normalize/standardize numerical variables
    '''
    train_mat = np.copy(X_train)
    test_mat = np.copy(X_test)
    scaler = StandardScaler()
    train_mat[:,numerical_lst] = scaler.fit_transform(train_mat[:,numerical_lst])
    test_mat[:,numerical_lst] = scaler.transform(test_mat[:,numerical_lst])
    return train_mat, test_mat
