
# coding: utf-8

# In[1]:

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')


# In[2]:

mnist = fetch_mldata('MNIST original', data_home='./')


# In[3]:

train_data = mnist['data'][0:60000]
train_label = mnist['target'][0:60000]

test_data = mnist['data'][60000:70001]
test_label = mnist['target'][60000:70001]


# In[4]:

rf_clf = RandomForestClassifier(n_estimators=10)
rf_clf.fit(train_data, train_label)

rf_label = rf_clf.predict(test_data)


# In[5]:

sum(rf_label == test_label) / test_data.shape[0]

