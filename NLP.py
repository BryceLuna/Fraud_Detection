import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

def build_model(df):
    X = df['description']
    Y = df['acct_type']
    X_train, X_test, y_train, y_test = train_test_split(x,Y, test_size=0.2, random_state=123)
    #might want to use CountVectorizer if you want to take into account doc length
    vectorizer = TfidfVectorizer(stop_words='english')
    tf_mat = vectorizer.fit_transform(X_train)
    #should experiment with different alpha's
    model = MultinomialNB(alpha=.01)
    model.fit(tf_mat,y_train)
    return vectorizer, model


def main():
    df = pd.read_pickle('data/df_text')
    vectorizer, mnb_model = build_model(df)
    with open('models/vectorizer.pkl','w') as v:
        pickle.dumps(,v)
    with open('models/mnb_model.pkl','w') as m:
        pickle.dumps(mnb_model,m)


if __name__ == '__main__':
    main()
