#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report, accuracy_score


# In[43]:


data= pd.read_csv("IRIS.csv")


# In[44]:


data.head()


# In[45]:


type(data)


# In[46]:


data.info()


# In[47]:


data.describe()


# In[48]:


data.iloc[50:100]


# In[49]:


sns.pairplot(data, hue="species")
plt.show()




# In[50]:


x = data.drop("species", axis=1)
x


# In[51]:


y = data["species"]
y


# In[70]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[71]:


x_train


# In[73]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)


# In[75]:


y_pred = knn.predict(x_test)


# In[76]:


print("Accuracy: ", accuracy_score(y_test, y_pred))


# In[77]:


print(classification_report(y_test, y_pred))


# In[78]:


x_test.head()


# In[ ]:




