#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as num
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[7]:


data= pd.read_csv("advertising.csv")


# In[8]:


data


# In[9]:


data.head(5)


# In[11]:


data.shape


# In[12]:


data.info()


# In[13]:


# checking for missing values
data.isnull().sum()


# No missing values in our dataset

# In[14]:


data.describe()


# In[15]:


sns.set()


# In[19]:


plt.figure(figsize=(6,6))
sns.distplot(data['TV'])
plt.show()


# In[20]:


plt.figure(figsize=(6,6))
sns.distplot(data['Radio'])
plt.show()


# In[21]:


plt.figure(figsize=(6,6))
sns.distplot(data['Newspaper'])
plt.show()


# In[22]:


plt.figure(figsize=(6,6))
sns.distplot(data['Sales'])
plt.show()


# In[32]:


# spliting Features and target
X= data.drop(columns='Sales', axis=1)
Y= data['Sales']


# In[33]:


print(X)


# In[34]:


print(Y)


# In[35]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=2)


# In[36]:


print(X.shape, X_train.shape, X_test.shape)


# In[37]:


regressor= XGBRegressor()


# In[38]:


regressor.fit(X_train, Y_train)


# Evaluation of our model

# In[39]:


training_data= regressor.predict(X_train)


# In[41]:


r2_train= metrics.r2_score(Y_train, training_data)


# In[42]:


print('R square value: ',r2_train)


# In[44]:


test_data= regressor.predict(X_test)


# In[45]:


r2_test= metrics.r2_score(Y_test, training_data)


# In[46]:


print('R square value: ',r2_test)

