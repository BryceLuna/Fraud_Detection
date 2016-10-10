import pandas as pd
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

'''
Notes:
consider using CountVectorizer instead of TF-IDF to account for description length
consider stemming and lemmatizing
should experiment with different alpha's
'''

def build_model(df):
    X = df['description']
    Y = df['acct_type']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=123)
    vectorizer = TfidfVectorizer(stop_words='english')
    tf_mat = vectorizer.fit_transform(X_train)
    model = MultinomialNB(alpha=.01)
    model.fit(tf_mat,y_train)
    return vectorizer, model

def generate_nlp_prob(vectorizer, model, description):
    mat = vectorizer.fit_transform(description)
    y_prob = model.predict_proba(mat)
    return y_prob


def main():
    df = pd.read_pickle('data/df_text.pkl')
    vectorizer, mnb_model = build_model(df)
    with open('models/vectorizer.pkl','w') as v:
        pickle.dump(vectorizer,v)
    with open('models/mnb_model.pkl','w') as m:
        pickle.dump(mnb_model,m)


if __name__ == '__main__':
    main()
