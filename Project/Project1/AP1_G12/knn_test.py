#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
from fancyimpute import  KNN
import math


# In[3]:


#Reading naidx file

file_name = '/home/anusha/Documents/sem/sem3/ML/CSC591_ML/Project1/AP1/test_data/test_data/naidx.csv'
df_naidx=pd.read_csv(file_name)
len_naidx=len(df_naidx)


# In[4]:


#Reading files for imputation

file_name = '/home/anusha/Documents/sem/sem3/ML/CSC591_ML/Project1/AP1/test_data/test_data/test_with_missing/{}.csv'
df_list = []

for i in range(1, 2268):
    df=pd.read_csv(file_name.format(i))
    
    #Applying KNN per patient using k=5
    train_cols = list(df)
    df_new=pd.DataFrame(KNN(k=5).fit_transform(df))
    df_new.columns = train_cols
    
    #Writing to files
    filename='./test_imputed_G12/{}.csv'
    df_new.to_csv(filename.format(i), sep=',', index=False)


# In[6]:


#Check presence of Nan
print (df_new.isnull().values.any())

