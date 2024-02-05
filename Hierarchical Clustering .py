#!/usr/bin/env python
# coding: utf-8

# In[37]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[38]:


#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"]='1'




# In[39]:


#loading our datasets
online_Shoppers = pd.read_csv('online_shoppers_intention.csv')


# In[40]:


#Data information

online_Shoppers.shape


# In[5]:


#viewing our dataset
online_Shoppers


# In[41]:


online_Shoppers.shape


# In[42]:


#inspecting our dataset

#showing the five first row of our datasets

online_Shoppers.head()


# In[43]:


#last five rows
online_Shoppers.tail()


# In[9]:


#checking for information about our dataset
online_Shoppers.info()


# In[44]:


#checking for null values
online_Shoppers.isnull().sum()


# In[45]:


#checking for duplicates

online_Shoppers.duplicated().value_counts()  


# In[46]:


#dropping duplicate

online_Shoppers=online_Shoppers.drop_duplicates()


# In[47]:


online_Shoppers.describe()


# In[48]:


online_Shoppers['VisitorType'].value_counts()


# In[49]:


online_Shoppers['Weekend'].value_counts()


# In[50]:


online_Shoppers['Revenue'].value_counts()


# In[51]:


online_Shoppers['Month'].value_counts()


# In[52]:


online_Shoppers


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.barplot(x='Revenue', y='SpecialDay', data=online_Shoppers)
plt.xlabel('Revenue')
plt.ylabel('Specialday')
plt.title('Bar Plot of Revenue vs. Special day')
plt.figure(figsize=(6, 3)) 
plt.show()


# In[53]:


sns.countplot(data=online_Shoppers,x='VisitorType',hue='Revenue')
plt.show()


# In[54]:


sns.countplot(data=online_Shoppers,x='Weekend',hue='Revenue')


# In[55]:


sns.scatterplot(x='PageValues',y='BounceRates',data=online_Shoppers,hue='VisitorType',palette='Dark2')


# In[23]:


sns.scatterplot(x='PageValues',y='ExitRates',data=online_Shoppers,hue='VisitorType',palette='Dark2')


# In[56]:


import seaborn as sns
import matplotlib.pyplot as plt


selected_columns =selected_columns = [  'BounceRates', 'ExitRates', 'PageValues',    'Revenue']

# Create a new DataFrame with the selected columns
subset_data = online_Shoppers[selected_columns]

# Create the pair plot for the selected columns
sns.pairplot(subset_data)

# Show the plot
plt.show()


# In[57]:


online_Shoppers.shape


# In[26]:


online_Shoppers


# In[58]:


from sklearn.preprocessing import LabelEncoder

# Create an instance of the LabelEncoder
encoder = LabelEncoder()


categorical_columns = ['Month', 'VisitorType', 'Weekend', 'Month', 'Revenue']


for column in categorical_columns:
    online_Shoppers[column] = encoder.fit_transform(online_Shoppers[column])


# In[59]:


#Standardization

from sklearn import preprocessing

X = online_Shoppers.iloc[:,0:18].values


# In[60]:


scaler = preprocessing.StandardScaler().fit(X)


# In[61]:


scaler.mean_


# In[62]:


scaler.scale_


# In[63]:


X_Scaled =scaler.transform(X)


# In[64]:


X_Scaled


# In[65]:


# Using the dendrogram to find the optimal of clusters
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))
dendrogram =sch.dendrogram(sch.linkage(X, method ='ward'))
plt.title('Dendrogram')  
plt.xlabel('Shoppers')
plt.ylabel('Euclidean distances') 
plt.show()  


# In[35]:


#fitting Hierarchical Clustering TO THE dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters =2, affinity = 'euclidean',linkage ='ward')
y_hc = hc.fit_predict(X)


# In[36]:


# Scatter plot for each cluster
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='blue', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='red', label='Cluster 2')


# In[ ]:





# In[ ]:




