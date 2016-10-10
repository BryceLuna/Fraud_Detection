import pandas as pd
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





def main():
    with open('models/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('models/mnb_model.pkl') as m:
        mnb_model = pickle.load(m)
    with open('data/df_clean.pkl') as d:
        df = pd.read_pickle(d)
        

if __name__ == '__main__':
    main()
