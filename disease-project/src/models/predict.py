#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")
import os
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd
import src.features.Feature_engineering as ft
import json


# In[67]:


df_precautions = pd.read_csv("../data/raw/symptom_precaution.csv")
df_description = pd.read_csv("../data/raw/symptom_Description.csv")
df_severity = pd.read_csv("../data/processed/Symptom-severity.csv",index_col="Symptom")


# In[91]:


def predict(df):
    for i in range(len(df)):
        df_t = ft.pipeline(df,df_severity)
        x = df_t.iloc[i,:].to_frame().T.to_json(orient="split")
        x = json.dumps(x)
        res = os.popen('curl -X POST http://127.0.0.1:1234/invocations -H "Content-Type:application/json; format=pandas-split" --data '+ x).read()
        print("------- Patient ",i," -------")
        print("Disease :",res[2:-2])
        disease_prec = df_precautions[df_precautions["Disease"]==res[2:-2]]
        print(" ")
        print("------ Precautions -------")
        for col in disease_prec.columns[1:]:
            
            print(col," \t ",disease_prec[col].tolist()[0])
        print(" ")
        disease_desc = df_description[df_description["Disease"]==res[2:-2]]
        print("------ Description -------")
        print(disease_desc["Description"].tolist()[0])
        print(" ")
        
        
        

