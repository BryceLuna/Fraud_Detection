# Fraud Detection
Detecting fraudulent events posted on Eventbrite
## Table of Contents
1. [Background](#background)
2. [The Data](#the-data)
3. [Files and Code]
  * [Exploratory Data Analysis]
  * [Data_Cleaning.py]
  * [NLP.py]
  * [Load_Data.py]
  * [Search_Models_Params.py]
  * [Models_Eval.py]
4. [Results]

## Background 
Eventbrite is an internet company that facilitates the creation and organization of events.  Unfortunately, there are a few unscrupulous individuals that use Eventbrite's platform to try and scam others by posting fraudulent events.  The goal of this project is to build a model that could detect these events.

## The Data
The data, provided by Eventbrite, is a json file consisting of over 14,000 events.  There are 44 data points collected on each event. 

## Files and Code
What follows is a short description of each project file as well as some considerations/design choices made along the way.

### Exploratory Data Analysis
The start of any data science project of this type begins with an exploration of the data (EDA).  The purpose is to simply "look at the data," address any anomalies, and ultimately make a determination of which of the 43 features are predictive of the events that were flagged as fraud.

Considerations:
1.  The target variable, acct_type, consisted of various labels such as spammer, premium, fraudster etc.  To simplify matters, every label with the phrase "fraud" in it was converted into a "1" otherwise the label was set as a "0."

2.  Any outliers and varying scales of the numerical features were taken into consideration.  These issues can cause problems when training a machine learning model, particularly for Logistic Regression.

3.  Of the 43 features, only around 20 of them were determined to be predictive.  A linear correlation heatmap was constructed to identify if any of the selected variables were highly correlated.   

A full accounting of the data exploration can be found in the Ipython notebook file: EDA.ipynb.

### Data_Cleaning.py
The data did not come pre-cleaned.  Here a function was written to dummify categorical variables, drop variables that were not predictive of the target variable, fill in missing values, set correct data types, and construct new variables.  

A html text description of each user's event was included in the data as one of the features.  A function was written to parse the html and return plain text.  These descriptions were later used to generate a new predictive variable.

### NLP.py
The goal of NLP.py was to generate a new feature variable, probability of fraud, from the event description.  Each event text was converted to a TF-IDF vector.  The matrix resulting from the transformation was then used to train a Multinomial Naive Bayes model (MNB).  Finally, each plain text description was passed to the MNB model and a probability of fraud was output.    

Considerations:
1.  Because TF-IDF was used to generate the word count frequencies, each event's description length was not taken into account.  However, it is plausible that fraudulent events have on average shorter description lengths.  To account for this possibility the body length variable, a measure of the text length, was retained in Data_Cleaning.py.

2.  Stop words were removed when constructing the TF-IDF matrix.  However, stemming and lemmatizing were not done.  It is possible that there might be some performance gain by doing these operations.

3.  The MNB hyper-parameter, alpha, was not chosen in a systematic way.  Further tuning might be possible by searching for an optimal alpha value.

### Load_Data.py
Load_Data.py contains three functions that pre-process the data for the machine learning models.  These functions, split the data into training and test sets, resample the data, and standardize numeric features.

Considerations:
1.    The same random seed used in the NLP file was used in split the data.  The concern was that if the data was not split in the same way the engineered variable constructed in NLP.py could perfectly predict fraud on some events depending on if they were in the training set of the MNB model.  

2.  Fraudulent events made up ~10% of the total.  Therefore, the minority class was over-sampled to balance the classes.  The sampling algorithm turned categorical columns into numeric, therefore, these columns were rounded.  Potentially, the imbalanced class issue could have been addressed by setting parameters of the respective learning models.

3.  

### Search_Models_Params.py
TBD

### Models_Eval.py
TBD

## Results
TBD
