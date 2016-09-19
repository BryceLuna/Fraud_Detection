import pandas as pd

#Data Cleaning
def clean_data(data_frame,columns_lst):
    #Dropping features that aren't obviously useful
    df = data_frame.copy()
    df.drop(columns_lst,axis=1,inplace=1)

    #Flag if the country variable doesn't match the venue_country
    #Note:will not flag if both are null
    df['diff_country'] = (df['venue_country'] != df['country']).apply(lambda x: 0 if x == False else 1)
    df.drop('venue_country',inplace=1,axis=1)
    df.country.fillna(value='unknown',inplace=True)
    df['country'] = df.country.map(lambda x: 'unknown' if x == '' else str(x))
    df['country_unknown'] = df.country.map(lambda x: 1 if x =='unknown' else 0)

    #dropping obscure countries, get the top 4 most common countries
    countries_lst = df.country.value_counts()[:4].index.values
    df['country'] = df.country.map(lambda x: 'other' if x not in country_lst else x)

    #flag specific email domains
    df['hotmail'] = df['email_domain'].str.contains('hotmail').apply(lambda x: 0 if x == False else 1)
    df['yahoo'] = df['email_domain'].str.contains('yahoo').apply(lambda x: 0 if x == False else 1)
    df['live'] = df['email_domain'].str.contains('live').apply(lambda x: 0 if x == False else 1)
    df.drop('email_domain', axis=1,inplace=1)

    df.org_facebook.fillna(value=-1,inplace=1)

    df['org_nameQ'] = df.org_name.map(lambda x: 0 if x == '' else 1)
    df.drop('org_name',axis=1,inplace=1)

    df['payout_type'] = df.payout_type.map(lambda x: 'unknown' if x =='' else x)

    #flag if there isn't a venue name
    df.venue_name.fillna(value='',inplace=1)
    df['venue_nameQ'] = df.venue_name.map(lambda x: 0 if x =='' else 1)
    df.drop('venue_name',axis=1,)

#clean all the varibles then change their datatypes then do dummies

    return df




def main():
    original_df = pd.read_json('data/train_new.json')
    columns = ['approx_payout_date','channels','gts','has_header',\
    'listed','name','name_length','org_desc','ticket_types','venue_latitude',\
    'venue_longitude','object_id','org_twitter','payee_name','num_order',\
    'previous_payouts','show_map','sale_duration2','user_type','venue_address',\
    'venue_state','description','venue_name']
    df = clean_data(original_df,columns)


if __name__ == '__main__':
    main()
