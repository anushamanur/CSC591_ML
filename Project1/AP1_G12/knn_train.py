#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing
from fancyimpute import  KNN #Imputation
import math


# In[6]:


#Reading naidx file

file_name = './AP1/train_data/train_data/naidx.csv'
df_naidx=pd.read_csv(file_name)
len_naidx=len(df_naidx)


# In[7]:


#Reading groundtruth
file_name = './AP1/train_data/train_data/train_groundtruth/{}.csv'
df_list = []
for i in range(1, 6001):
    pt=pd.read_csv(file_name.format(i))
    pt['pt_no'] = i
    pt['index'] = range(1,len(pt)+1)
    df_list.append(pt)
df_truth = pd.concat(df_list)


# In[11]:


#Reading files with missing values for imputation using train data
file_name = './AP1/train_data/train_data/train_with_missing/{}.csv'
df_list = []
for i in range(1, 6001):
    
    df=pd.read_csv(file_name.format(i))
    df['pt_no'] = i
    df['index'] = range(1,len(df)+1)

    #Applying KNN with k=5
    train_cols = list(df)
    df_new=pd.DataFrame(KNN(k=5).fit_transform(df))
    df_new.columns = train_cols
    
    df_list.append(df_new)
df_train = pd.concat(df_list)


# In[12]:


#RMSD per column

def rmsd(imputed_val):
    d=[-1]*13
    diff=0
    for i in range(1,14):
        print ("i = ", i)
        df_naidx_sub=df_naidx.loc[df_naidx['test'] == ('X'+str(i))]
        len_naidx_per_col=len(df_naidx_sub)
        diff=0
        for index, row in df_naidx_sub.iterrows():
            x= imputed_val.loc[(imputed_val['index'] == row['i']) & (imputed_val['pt_no'] == row['pt.num']) ,row['test']].iloc[0]
            y= df_truth.loc[(df_truth['index'] == row['i']) & (df_truth['pt_no'] == row['pt.num']),row['test']].iloc[0]
            df_truth_sub=df_truth.loc[(df_truth['pt_no'] == row['pt.num'])]
            rnge=df_truth_sub[row['test']].max()-df_truth_sub[row['test']].min()
            diff+=((x-y)/rnge)**2
        d[i-1]=math.sqrt(diff/len_naidx_per_col)

    print (d) 


# In[13]:


#RMSD per column
res=rmsd(df_train)


# In[ ]:




