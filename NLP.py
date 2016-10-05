from bs4 import BeautifulSoup
import string
from unidecode import unidecode


def clean_description(txt):
    soup = BeautifulSoup(txt, 'html.parser')
    return unidecode(soup.txt).translate(None, string.punctuation)
    


def main():
    df_original = pd.read_json('data/train_new.json')
    df = df_original[['acct_type','description']]
    df['text'] = df.description.map(lambda x: clean_desciption)
    
if __name__ == '__main__':
    main()
