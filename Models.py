import pandas as pd
import numpy as np
import cPickle as pickle
from scipy.stats import randint as sp_randint
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

'''
Notes:
-Consider transforming varibles to be on same scale (depends on algo)
-Consider binning non-continuous variables
-Split in the same way you did for NLP
-Think about undersampling instead of over-sampling or giving more weight to minority class
-Try rounding if over-sampling minority class
-Consider using LogisticRegressionCV for searching over Cs
-Perhaps could have used class_weight and an intercept to avoid scaling and
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

def parameter_search(model, X, y, params, metric, n=10):
    '''
    returns the best parameters of the classification model
    '''
    random_search = RandomizedSearchCV(model, param_distributions=params, \
    scoring = metric, n_jobs=3, n_iter=n)
    random_search.fit(X, y)
    return random_search

def main():
    
    numerical_lst = [0,4,6,11,12] #starts at zero - dropped acct_type
    categorical_lst = [1,2,3,5,7,8,9,10,13,14,15,16,17,18,19,20] #prob (21) left out
    df = pd.read_pickle('data/df_clean.pkl')
    with open('data/y_prob.pkl','r') as f:
        y_prob = pickle.load(f)
    df['fraud_prob'] = y_prob
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train, categorical_lst)
    X_train_re_std, X_test_re_std = standardize_variables(X_train_resampled, X_test, numerical_lst)

    #Logistic Regression
    logistic_params = {"C":[1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4]}
    logistic = LogisticRegression()
    logistic_searched = parameter_search(\
    logistic, X_train_re_std, y_train_resampled, logistic_params, 'f1', 5)
    with open('models/logistic_searched.pkl','w') as l:
        pickle.dump(logistic_searched, l)

    #Random Forest
    forest = RandomForestClassifier(n_estimators=200, n_jobs=3)
    forest_params = {"max_depth": [3, 4, None],
              "max_features": sp_randint(1, 15),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 20),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    forest_searched = parameter_search(\
    forest, X_train_resampled, y_train_resampled, forest_params, 'f1')
    with open('models/randomForest_searched.pkl','w') as f:
        pickle.dump(forest_searched, f)

    #Gradient Boosting
    boosting = GradientBoostingClassifier(n_estimators=200)
    gradient_params = {"max_depth": [1, 2, 3],
              "max_features": sp_randint(1, 15),
              "learning_rate": [.1, .2, .5],
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 20)}
    boosting_searched = parameter_search(\
    boosting, X_train_resampled, y_train_resampled, gradient_params, 'f1')
    with open('models/boosting_searched.pkl','w') as b:
        pickle.dump(boosting_searched, b)

    
if __name__ == '__main__':
    main()
