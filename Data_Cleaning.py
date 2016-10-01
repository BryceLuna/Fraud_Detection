import pandas as pd

'''
Notes:
1. Handle minority class
2. Split data
3. Scale data as needed
4. Do NLP on description

'''
#Data Cleaning
def clean_data(data_frame,columns_lst):
    #clean all the varibles then change their datatypes then do dummies

    #Dropping features that aren't obviously useful
    df = data_frame.copy()
    df.drop(columns_lst,axis=1,inplace=1)

    #Flag if the country variable doesn't match the venue_country
    #Note:will not flag if both are null
    df['diff_country'] = (df['venue_country'] != df['country']).apply(lambda x: 0 if x == False else 1)
    df.drop('venue_country',inplace=1,axis=1)

    #df.country.fillna(value='unknown',inplace=True)
    #Flag if country wasn't available, check if this variable is predictive
    #df['country_nan'] = df.country.isnull().astype(int)
    #This might not be necessary because empty string countries would likely get filted below
    #df['country'] = df.country.map(lambda x: 'empty_string' if x == '' else str(x))
    #df['country_unknown'] = df.country.map(lambda x: 1 if x =='unknown' else 0)

    #dropping obscure countries, get the top 4 most common countries
    countries_lst = df.country.value_counts()[:4].index.values
    df['country'] = df.country.map(lambda x: 'minor' if x not in countries_lst else 'major')

    df.delivery_method.fillna(value=-1,inplace=1)
    df['delivery_method'] = df.delivery_method.astype(int)

    #flag specific email domains
    df['hotmail'] = df['email_domain'].str.contains('hotmail').apply(lambda x: 0 if x == False else 1)
    df['yahoo'] = df['email_domain'].str.contains('yahoo').apply(lambda x: 0 if x == False else 1)
    df['live'] = df['email_domain'].str.contains('live').apply(lambda x: 0 if x == False else 1)
    df.drop('email_domain', axis=1,inplace=1)

    #Transforming time variables - from Unix Time
    df['user_created'] = pd.to_datetime(df['user_created'], unit='s')
    df['event_created'] = pd.to_datetime(df['event_created'], unit='s')
    df['event_start'] = pd.to_datetime(df['event_start'], unit='s')

    df['user_create_to_start'] = df['event_start'] - df['user_created']
    df['create_to_start'] = df['event_start'] - df['event_created']
    df.drop(['event_created','event_start','user_created'], axis=1,inplace=1)

    df.org_facebook.fillna(value=-1,inplace=1)
    df['org_facebook'] = df.org_facebook.map(lambda x: 1 if x == 0 else 0).astype(int)

    #Flag if there is no org name
    df['org_nameQ'] = df.org_name.map(lambda x: 0 if x == '' else 1)
    df.drop('org_name',axis=1,inplace=1)

    df['payout_type'] = df.payout_type.map(lambda x: 'unknown' if x =='' else x)

    #flag if sale_duration doesn't exist
    df['sale_duration_nan'] = df.sale_duration.isnull().astype(int)
    df.drop('sale_duration',axis=1,inplace=1)

    #flag if there isn't a venue name
    df.venue_name.fillna(value='',inplace=1)
    df['venue_nameQ'] = df.venue_name.map(lambda x: 0 if x =='' else 1)
    df.drop('venue_name',axis=1,inplace=1)

    df = pd.get_dummies(df,\
    columns=['country','delivery_method','payout_type'],drop_first=1)

    #delivery method 1 is negatively correlated with method 0
    df.drop('delivery_method_1',axis=1,inplace=1)
    return df
    

def main():
    original_df = pd.read_json('data/train_new.json')
    columns = ['approx_payout_date','channels','gts','has_header',\
    'listed','name','name_length','org_desc','ticket_types','venue_latitude',\
    'venue_longitude','object_id','org_twitter','payee_name','num_order',\
    'previous_payouts','show_map','sale_duration2','user_type','venue_address',\
    'venue_state','description','event_end','event_published','currency']
    df = clean_data(original_df,columns)
    df.to_csv("C:/Users/Anon/Desktop/clean_df.csv",index=False)


if __name__ == '__main__':
    main()
