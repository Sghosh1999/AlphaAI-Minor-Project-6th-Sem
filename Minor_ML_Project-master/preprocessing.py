#Importing Necessary Libraries
#Author: Sayantan Ghosh
#Date: 13/02/2019

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.datasets import make_classification
import plotly.express as px
import streamlit as st





#Step1: Handling Missing Values
#Step2 :Handling Categorivcal features
#Step3 :PCA Dimensionality reduction
#Step4 :Outlier Removing
#Step5 :Feature Scaling(MinMax Scaling & Standard Scaling)

def feature_reduction(dup_dataset,target_column,independent_column):
    #Applying Feature Reduction using Ensemble Trees
    #target_column : The Feature to predict
    #independent-column : The predictors
    X = independent_column
    tempX=X
    y = target_column
    no_of_features = X.shape[1]
    no_of_classes = y.nunique()
    X,y = make_classification(n_samples=1000,
                              n_features=no_of_features,
                              n_classes=no_of_classes,
                              random_state=0)
    #Build the Forest and Compute the Feature Importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    
    #Fitting The Random Forest
    forest.fit(X,y)

    #Storing Feature Importances
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    f_col=[]
    f_col_imp=[]
    for i in range(tempX.shape[1]):
        f_col.append(tempX.columns[indices[i]])
    for j in range(tempX.shape[1]):
        f_col_imp.append(importances[indices[j]])

    data = {'Features' :f_col, 'Importances' : f_col_imp}
    data = pd.DataFrame(data)
    fig = px.bar(data, x='Features', y='Importances')
    st.plotly_chart(fig)




def handling_missing_values(dup_dataset):
    #Removing the NAN varibales by Mean Strategy
    for cols in list(dup_dataset.columns):
        if dup_dataset[cols].isnull().sum!=0:
            mean_val = dup_dataset[cols].mean()
            dup_dataset[cols]=dup_dataset[cols].fillna(mean_val)

     



