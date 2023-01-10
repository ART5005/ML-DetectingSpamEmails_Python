#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np


# In[40]:


from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[41]:


email= pd.read_csv("C:\\Users\\tayea\\Downloads\\Spam.csv")


# In[42]:


email.head(5)


# In[43]:


email.shape


# In[44]:


email.isnull().any().any()


# In[45]:


# Define x and y
X = email.drop(["spam"], axis = 1).values
y = email["spam"].values


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345, stratify = y )
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[59]:


print(X_train.shape)
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape)


# In[47]:


print("\nAccuracy Score: ", accuracy_score(y_test, y_pred)*100, "\n")
print("Class_Report:\n", classification_report(y_test, y_pred), end = "")
print("\nConf_Matrix: \n",confusion_matrix(y_test, y_pred))


# In[48]:


# The confusion matrix shows that the model predicted 686 true positive and 427 true negative correctly.
# And has 151 false posive and 117 false negative .
# The classification report shows that the model predicts an important mail with accuracy of 85% and 74%  for spam
# Recall shows the percentage of positive predictions relative to actual positives
# The f1-scores are closer to 1 which shows that the model is good.
# The f1-score shows that the model is able to identify 76% of spam emails and 84% of good emails i.e, for every 100 spam
# emails, the model will identify 74, and for every 100 good emails, the model will identify 84.


# In[49]:


from sklearn.model_selection import GridSearchCV
knn2 = neighbors.KNeighborsClassifier()
# create a dictionary of all values to test for different n_neighbors
param_grid = {"n_neighbors":np.arange(1, 50)}

knn_grid = GridSearchCV(knn, param_grid, cv = 5)
# fit model
knn_grid.fit(X, y)


# In[50]:


# best k
knn_grid.best_params_


# In[51]:


# Accuracy scores at different k values
i = 0
for i in range(1, 31):
    knn = neighbors.KNeighborsClassifier(n_neighbors = i)       # The best value of k 1
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    z = knn.score(X_test, y_test)
    i += 0
    print("when k =", i, "Accuracy_score =", round(z*100, 1), end = " ,  ")


# In[52]:


# In this particular dataset, the best value for K is 1 because it has the best scores. 
# for confusion matrix and accuracy score, and has the a larger number for true positive and true negative . 
# For other values of k, the accuracy scores are lower so also the number of true positive and true negative.


# In[53]:


# Predicting the first four rows of the dataset
knn.predict(X[:4, :])


# In[54]:


# Predicting the last four rows of the dataset
knn.predict(X[-4:, :]) 


# In[ ]:




