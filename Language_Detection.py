#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Language Identification Program. Everyone knows that we can translate a language 
# using Google Translate. However, what if we don't knnow what langugae they are speaking?
# How will we deal with that situation then?


# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[3]:


# Load in the dataset and check to see whether or not there are many 
# duplicates or missing values in the dataset before we proceed 


# In[4]:


lang = pd.read_csv("/Users/dawsontam/Downloads/Language_Detection.csv")
lang.head()


# In[5]:


#get general info on dataset

lang.info()


# In[6]:


#check for any missing values in the dataset 

lang.isna().sum()


# In[7]:


#plot distriubtion of languages to see if we are going to be training our porgram with a certian bias towards certain languages

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (15, 10))

sns.countplot(data = lang, x= "Language", order = lang["Language"]\
             .value_counts().index)

plt.xticks(rotation = 90)
plt.show()


# In[8]:


X = lang["Text"]
Y = lang["Language"]

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_scaled = cv.fit_transform(X)


# In[ ]:





# In[9]:


#split into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 123)


# In[10]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb_predictions = mnb.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score

result = accuracy_score(y_test, mnb_predictions)
result


# In[12]:


# I will use a confusion matrix to meausure the success of the algorithmn

from sklearn.metrics import confusion_matrix, classification_report

sns.heatmap(confusion_matrix(y_test, mnb_predictions), annot = True)
plt.ylabel("Actual")
plt.xlabel("Prediction")

print(classification_report(y_test, mnb_predictions))


# In[13]:


# As we can see from the confusion matrix, we have high number of true
# positives as well as high numbers of true negatives in the bottom right
# Thus, we can conclude that the Multnomial Naive Bayes Regression is 
# appropriate for the data. 

