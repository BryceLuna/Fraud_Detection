import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

'''
Notes:
-Consider transforming varibles to be on same scale (depends on algo)
-It's important to train on balanced classes
-Don't know if it's necessary to resample the testing set
-Consider binning non-continuous variables
-You need to split in the same way you did for NLP
-Think about undersampling instead of over-sampling
-Perhaps try and using rounding if using over-sampling minority class
'''

def split_data(df):
    '''
    returns training and testing datasets
    '''
    y = df['acct_type']
    X = df.drop(['acct_type'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #X_train_resampled, y_train_resampled = sm.fit_sample(X_test,y_test)
    return X_train, X_test, y_train, y_test #X_train_resampled, y_train_resampled, X_train_resampled, y_train_resampled

def resample_data(X, y, categorical_lst):
    sm = SMOTE(kind='regular')
    X_train_resampled, y_train_resampled = sm.fit_sample(X,y)
    #rounding categorical variables
    X_train_resampled[:,categorical_lst] = np.round(X_train_resampled[:,categorical_lst])
    return X_train_resampled, y_train_resampled

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

def grid_search(model, X, y, params, scoring):
    '''
    returns the best parameters of the classification model
    '''
    clf = GridSearchCV(model, params, scoring=scoring)
    clf.fit(X,y)
    pass

def build_classifier(model,params={}):
    model.fit(**params)
    pass

def main():
    # with open('models/vectorizer.pkl') as f:
    #     vectorizer = pickle.load(f)
    # with open('models/mnb_model.pkl') as m:
    #     mnb_model = pickle.load(m)
    #with open('data/df_clean.pkl','r') as d:
    df = pd.read_pickle('data/df_clean.pkl')
    with open('data/y_prob.pkl','r') as f:
        y_prob = pickle.load(f)
    df['fraud_prob'] = y_prob
    X_train, X_test, y_train, y_test = split_data(df)
    transform_body_length(X_train, X_test)

    build_Classifier(X_train, y_train, params)


if __name__ == '__main__':
    #main()
    numerical_lst = [0,4,6,11,12] #starts at zero - dropped acct_type
    categorical_lst = [1,2,3,5,7,8,9,10,13,14,15,16,17,18,19,20,21] #prob (22) left out
    df = pd.read_pickle('data/df_clean.pkl')
    with open('data/y_prob.pkl','r') as f:
        y_prob = pickle.load(f)
    df['fraud_prob'] = y_prob
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train, categorical_lst)
    X_train_std, X_test_std = standardize_variables(X_train_resampled, X_test, numerical_lst)
    #Logistic Regression
    logistic_params = {}
    #Random Forest
    param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    #logistc_model = grid_search(LogisticRegression, X_train_std, y_train_resampled,{}, )
