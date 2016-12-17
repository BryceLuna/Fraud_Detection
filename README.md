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

A full accounting of the data exploration can be found in the Ipython notebook file: EDA.ipynb.

### Data_Cleaning.py
### NLP.py
### Load_Data.py
### Search_Models_Params.py
### Models_Eval.py

## Results
