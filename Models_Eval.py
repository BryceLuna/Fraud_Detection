import pandas as pd
import numpy as np
import cPickle as pickle
from Load_Data import split_data, resample_data, standardize_variables
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def evaluate_model(model, X, y):
    '''
    returns the classification report
    '''
    y_pred =  model.predict(X)
    report = classification_report(y, y_pred)
    return report


def main():
    pass


if __name__ == '__main__':
    #main()
    numerical_lst = [0,4,6,11,12] #starts at zero - dropped acct_type
    categorical_lst = [1,2,3,5,7,8,9,10,13,14,15,16,17,18,19,20] #prob (21) left out
    df = pd.read_pickle('data/df_clean.pkl')
    with open('data/y_prob.pkl','r') as f:
        y_prob = pickle.load(f)
    df['fraud_prob'] = y_prob
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train, categorical_lst)
    X_train_re_std, X_test_re_std = standardize_variables(X_train_resampled, X_test, numerical_lst)

    with open('models/logistic_searched.pkl','r') as l:
        logistic_params = pickle.load(l)

    logistic_model = LogisticRegression(**logistic_params.best_params_)
    logistic_model.fit(X_train_re_std, y_train_resampled)
    logistic_report = evaluate_model(logistic_model, X_test, y_test)


    with open('models/randomForest_searched.pkl','r') as b:
        rf_params = pickle.load(b)

    rf_model = RandomForestClassifier(n_estimators=200, n_jobs=3, **rf_params.best_params_)
    rf_model.fit(X_train_resampled, y_train_resampled)
    rf_report = evaluate_model(rf_model, X_test, y_test)

    with open('models/boosting_searched.pkl','r') as b:
        boosting_params = pickle.load(b)

    boosting_model = GradientBoostingClassifier(n_estimators = 200, **boosting_params.best_params_)
    boosting_model.fit(X_train_resampled, y_train_resampled)
    boosting_report = evaluate_model(boosting_model, X_test, y_test)
