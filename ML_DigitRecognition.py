#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[32]:


plt.gray() 
for i in range(9):
    plt.matshow(digits.images[i])


# In[12]:


dir(digits)


# In[13]:


digits.data[0]


# In[14]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)


# In[31]:


model.fit(X_train, y_train)


# In[18]:


model.score(X_test, y_test)


# In[19]:


model.predict(digits.data[0:5])


# In[20]:


y_predicted = model.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[22]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




