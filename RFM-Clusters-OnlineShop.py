#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
import seaborn as sns


# In[2]:


data = pd.read_excel('Online Retail.xlsx')
data.head()


# In[3]:


data.info()


# In[4]:


data = data[data['CustomerID'].notnull()]


# In[5]:


data.info()


# In[9]:


data['InvoiceDay'] = data['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, x.day))


# In[10]:


data.head()


# In[11]:


dt.timedelta(1)


# In[12]:


pin_date = max(data['InvoiceDay']) + dt.timedelta(1)
pin_date


# In[13]:


data['TotalSum'] = data['Quantity'] * data['UnitPrice']
data.head()


# In[14]:


rfm = data.groupby('CustomerID').agg({
    'InvoiceDay': lambda x: (pin_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'
})
rfm


# In[15]:


data[data['CustomerID'] == 12346.0]


# In[17]:


rfm.rename(columns={
    'InvoiceDay': 'R',
    'InvoiceNo' : 'F',
    'TotalSum' :'M'
}, inplace=True)
rfm


# In[19]:


R_l=range(4,0,-1)
R_G= pd.qcut(rfm['R'],q=4 ,labels=R_l)
F_l=range(1,5)
F_G=pd.qcut(rfm['F'], q=4,labels=F_l)
M_l= range(1,5)
M_G=pd.qcut(rfm['M'],q=4, labels=M_l)


# In[21]:


rfm['Recency']=R_G.values
rfm['Frequency']=F_G.values
rfm['Monetary']=M_G.values


# In[22]:


rfm


# In[27]:


x=rfm[['Recency','Frequency','Monetary'] ]
kmeans=KMeans(n_clusters=5, init ='k-means++',max_iter=300) 
kmeans.fit(x)


# In[28]:


kmeans.labels_


# In[39]:


rfm['kmeans_cluster']=kmeans.labels_


# In[41]:


rfm[rfm['kmeans_cluster'] == 2].mean()

rfm[rfm['kmeans_cluster'] == 2].astype(float).mean()

# In[42]:


rfm[rfm['kmeans_cluster'] == 2].astype(float).mean()


# In[43]:


rfm.astype(float).mean()


# In[44]:


kmeans.inertia_


# In[49]:


wcss = {}
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init='k-means++',max_iter=300)
    kmeans.fit(x)
    wcss[k]=kmeans.inertia_
    sns.pointplot(x=list(wcss.keys()),y=list(wcss.values()))


# In[ ]:




