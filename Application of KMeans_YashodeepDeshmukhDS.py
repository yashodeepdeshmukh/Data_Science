#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


# In[2]:


#Importing data set from library


# In[3]:


df = pd.read_excel('titanic.xls')
df


# In[4]:


#data cleaning


# In[5]:


df[['boat','home','cabin','fare']]=df[['boat','home','cabin','fare']].fillna(0)
df.dropna(subset=['age'], inplace=True)
clean=df.drop(['name','body','ticket','home.dest'],1,inplace=False)


# In[6]:


clean


# In[7]:


#data_conversion


# In[8]:


def handle_non_numerical_data(clean):
    columns = clean.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if clean[column].dtype != np.int64 and clean[column].dtype != np.float64:
            column_contents = clean[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            clean[column] = list(map(convert_to_int, clean[column]))

    return clean
clean = handle_non_numerical_data(clean)


# In[9]:


#Kmeans


# In[10]:


X = np.array(clean.drop(['survived'], 1).astype(float))
y = np.array(clean['survived'])


# In[11]:


clf = KMeans(n_clusters=2)


# In[12]:


clf.fit(X)


# In[13]:


#check probability


# In[14]:Check accuracy


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))




