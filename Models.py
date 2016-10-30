import pandas as pd
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

'''
Notes:
need to transform body_length varible and any other continuous numeric variables
Don't forget cross-validation
'''

def split_data(df):
    '''
    returns training and testing datasets
    '''
    sm = SMOTE(kind='regular')
    Y = df['acct_type']
    X = df['description']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=123)
    X_train_resampled, y_train_resampled = sm.fit_sample(X_train,y_train)
    X_train_resampled, y_train_resampled = sm.fit_sample(X_test,y_test)
    return X_train_resampled, y_train_resampled, X_train_resampled, y_train_resampled

def transform_body_length(X_train, X_test, feature):
    '''
    normalizes the body_length variable
    '''
    scaler = StandardScaler()
    scaler.fit_transform()
    scaler.transform()
    
    pass

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
    with open('data/df_clean.pkl') as d:
        df = pd.read_pickle(d)
    with open('data/y_prob.pkl','r') as f:
        y_prob = pickle.load(f)
    df['fraud_prob'] = y_prob
    X_train, X_test, y_train, y_test = split_data(df)
    transform_body_length(X_train, X_test)

    
    build_Classifier(X_train, y_train, params)
    

if __name__ == '__main__':
    main()
