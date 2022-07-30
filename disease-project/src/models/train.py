#!/usr/bin/env python

import sys
sys.path.append("..")
from sklearn.metrics import recall_score,f1_score,precision_score
from src.features import Feature_engineering
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.svm import SVC
import glob
import shap
import os


# In[7]:


import mlflow
import mlflow.sklearn


# In[2]:


df_severity = pd.read_csv("../data/processed/Symptom-severity.csv",index_col="Symptom")


# In[9]:


def model_training(df):
    with mlflow.start_run(run_name="SVC"):
        (x_train,
         x_test,
         y_train,
         y_test) = Feature_engineering.pipeline(df,df_severity)
        clf = SVC(gamma='auto', C=2,probability=True)
        clf.fit(x_train.to_numpy(),y_train.to_numpy())
        recall, f1, precision,score,expl_kernel,shap_values_k,x_t = evaluation(x_test,y_test,clf)
        
        mlflow.log_param("gamma","auto")
        mlflow.log_param("C",2)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1",f1)
        mlflow.log_metric("precision",precision)
        mlflow.log_metric("score",score)
        mlflow.sklearn.log_model(clf,"SVC")
        
    return recall, f1, precision,score,expl_kernel,shap_values_k,x_t


# In[10]:


def model_training_1(df):
    with mlflow.start_run(run_name="logistic"):
        (x_train,
         x_test,
         y_train,
         y_test) = Feature_engineering.pipeline(df,df_severity)
        clf = LogisticRegression(penalty='l2', C=5)
        clf.fit(x_train,y_train)
        recall, f1, precision,score = evaluation(x_test,y_test,clf)
        
        mlflow.log_param("penalty","l2")
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1",f1)
        mlflow.log_metric("precision",precision)
        mlflow.log_metric("score",score)
        mlflow.sklearn.log_model(clf,"Logistic")
    return recall, f1, precision, score


# In[11]:


def evaluation(x_test,y_test,clf):
    y_pred = clf.predict(x_test)
    recall = recall_score(y_test,y_pred,average="micro")
    f1 = f1_score(y_test,y_pred,average="micro")
    precision = precision_score(y_test,y_pred,average="micro")
    #expl = shap.Explainer(clf)
    expl_kernel = shap.KernelExplainer(clf.predict_proba,x_test.iloc[:5,:].to_numpy())

    shap_values_k = expl_kernel.shap_values(x_test)
    #shap_values = expl.shap_values(x_test)
    score = clf.score(x_test,y_test)
    return recall, f1, precision,score,expl_kernel,shap_values_k,x_test.iloc[:5,:]

