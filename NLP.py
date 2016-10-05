from bs4 import BeautifulSoup
import string
from unidecode import unidecode
import pandas as pd

def clean_description(txt):
    soup = BeautifulSoup(txt, 'html.parser')
    return unidecode(soup.text).translate(None, string.punctuation)
    


def main():
    df = pd.read_json('data/train_new.json')
    df['text'] = df.description.map(lambda x: clean_description)
    
if __name__ == '__main__':
    main()
