from bs4 import BeautifulSoup


def clean_description(txt):
    pass


def main():
    df_original = pd.read_json('data/train_new.json')
    df = df_original[['acct_type','description']]
    df['text'] = df.description.map(lambda x: clean_desciption)
    
if __name__ == '__main__':
    main()
