#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as num
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


data= pd.read_csv("creditcard.csv")


# In[6]:


data


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.info()


# In[11]:


# gives sum of missing values in each columns
data.isnull().sum()


# In[12]:


# distribution of legit and fraud transactions
data['Class'].value_counts()


# 0 is normal transaction
# 1 is fraud transactions

# In[13]:


legit= data[data.Class==0]
fraud= data[data.Class==1]


# In[14]:


print(legit.shape)
print(fraud.shape)


# In[15]:


legit.Amount.describe()


# In[16]:


fraud.Amount.describe()


# In[17]:


data.groupby("Class").mean()


# Build a sample dataset containing similar distribution of normal transaction and Fraudluent transaction

# Number of Fraud Transaction= 492

# In[18]:


legit_sample= legit.sample(n=492)


# In[20]:


new_data= pd.concat([legit_sample, fraud], axis=0)


# In[21]:


new_data.head()


# In[22]:


new_data['Class'].value_counts()


# In[23]:


new_data.groupby('Class').mean()


# In[24]:


X= new_data.drop(columns='Class', axis=1)
Y= new_data['Class']


# In[25]:


print(X)


# In[26]:


print(Y)


# Spliting data into Training and Testing data

# In[30]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[31]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training

# In[35]:


model = LogisticRegression()


# In[37]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=1000)


# In[39]:


model.fit(X_train, Y_train)


# # Accuracy Scores

# In[42]:


X_train_prediction= model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)


# In[45]:


print('Accuracy on training data: ', training_data_accuracy)


# In[47]:


X_test_prediction= model.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)


# In[48]:


print("Accuracy score of test data: ",test_data_accuracy)

