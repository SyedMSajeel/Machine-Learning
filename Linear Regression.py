#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# In[4]:


data = pandas.read_csv('cost_revenue_clean.csv')


# In[5]:


data.describe()


# In[6]:


X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])


# In[7]:


plt.figure(figsize=(10,6))
plt.scatter(X,Y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Producton Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0,3000000000)
plt.xlim(0,450000000)
plt.show()


# In[8]:


regression = LinearRegression()
regression.fit(X, Y)


# In[9]:


regression.coef_ #theta_1


# In[11]:


regression.intercept_ #intercept


# In[15]:


plt.figure(figsize=(10,6))
plt.scatter(X,Y, alpha=0.3)
plt.plot(X, regression.predict(X), color='red', linewidth=4)

plt.title('Film Cost vs Global Revenue')
plt.xlabel('Producton Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0,3000000000)
plt.xlim(0,450000000)
plt.show()


# In[16]:


regression.score(X, Y)


# In[ ]:




