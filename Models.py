import pandas as pd
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split


'''
Notes:
consider using SMOTE to balance the classes
might be data-leakage if you don't split the same way
need to transform body_length varible in same way as training 
'''

def split_data(df):
    '''
    returns training and testing datasets
    '''

    Y = df['acct_type']
    X = df['description']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=123)

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
    split_data(df)

if __name__ == '__main__':
    main()
