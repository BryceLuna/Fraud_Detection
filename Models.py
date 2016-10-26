import pandas as pd
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE

'''
Notes:
consider using SMOTE to balance the classes
might be data-leakage if you don't split the same way as you did for NLP
need to transform body_length varible in same way as training 
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

def build_Classifier(model,params={}):
    

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
    
    build_Classifier()
    

if __name__ == '__main__':
    main()
