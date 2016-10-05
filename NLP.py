import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

def build_model(df):
    X = df['description']
    Y = df['acct_type']
    X_train, X_test, y_train, y_test = train_test_split(x,Y, test_size=0.3, random_state=123)
    model()
    fit_transform()
    return tf_mat, model


def main():
    df = pd.read_pickle('data/df_text')
    tf_mat, model = build_model(df)
    with open('','w') as v:
        pickle.dumps(,v)
    with open('','w') as m:
        pickle.dumps(,m)

    
if __name__ == '__main__':
    main()
