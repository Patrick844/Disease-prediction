#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 


# ### Read data

def missing_values(df):
    df = df.fillna("no symptoms")
    return df


def weight_symptoms(df_severity,df):
    for symp in df_severity.index.tolist():
        weight = df_severity.loc[symp][0]
        df = df.replace(symp,weight)
    return df


def split(df):
    X = df.iloc[:,1:]
    Y = df.iloc[:,0]
    
    x_train, x_test,y_train, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
    return (x_train,
            x_test,
            y_train,
            ytest)
    

def pipeline(df,df_severity):
    if "Disease" in df.columns:
        df =df.iloc[:,:-5]
        df = missing_values(df)
        df = weight_symptoms(df_severity,df)
        (x_train,
         x_test,
         y_train,
         y_test) = split(df)
        return x_train, x_test, y_train, y_test
    else:
        df = missing_values(df)
        df = weight_symptoms(df_severity,df)
        return df





