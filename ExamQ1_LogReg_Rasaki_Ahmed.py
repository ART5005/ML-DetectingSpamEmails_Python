#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[37]:


email= pd.read_csv("C:\\Users\\tayea\\Downloads\\Spam.csv")


# In[38]:


email.tail(3)


# In[39]:


email.shape


# In[40]:


# Define x and y
X = email.drop(["spam"], axis = 1).values
y = email["spam"].values


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345, stratify = y)
lr = LogisticRegression(max_iter = 1000, random_state = 12345)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred


# In[49]:


print(X_train.shape)
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape)


# In[43]:


round(accuracy_score(y_test, y_pred)*100, 2) 


# In[44]:


print("\nAccuracy Score: ", accuracy_score(y_test, y_pred)*100, "\n")
print("\nClassRep: \n", classification_report(y_test, y_pred), end = "")
print("\nConf_Matrix: \n" ,confusion_matrix(y_test, y_pred))


# In[ ]:


# The confusion matrix shows that the model predicted 803 true positive and 469 true negative correctly.
# And has 34 false posive and 75 false negative .
# The classification report shows that the model predicts an important mail with accuracy of 85% and 74%  for spam
# Recall shows the percentage of positive predictions relative to actual positives
# The f1-scores are closer to 1 which shows that the model is good.
# The f1-score shows that the model can identify 90% of spam emails and 94% of good emails i.e, for every 100 spam
# emails, the model will identify 90, and for every 100 good emails, the model will identify 94.


# In[45]:


# predicting the first four rows of the dataset
lr.predict(X[:4, :])


# In[46]:


# predicting the last four rows of the dataset
lr.predict(X[-4:, :])


# In[ ]:




