
# coding: utf-8

# In[18]:

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')


# In[9]:

mnist = fetch_mldata('MNIST original', data_home='./')
print(mnist)


# In[10]:

idx = list(range(70000))
random.shuffle(idx)


# In[16]:

kf = KFold(n_splits=10, shuffle=True)
accuracy = []
for train_index, test_index in kf.split(mnist['data']):
    train_data = mnist['data'][train_idx]
    train_label = mnist['target'][train_idx]

    test_data = mnist['data'][test_idx]
    test_label = mnist['target'][test_idx]
    
    rf_clf = RandomForestClassifier(n_estimators=10)
    rf_clf.fit(train_data, train_label)
    
    rf_label = rf_clf.predict(test_data)
    current_accuracy = sum(rf_label == test_label) / test_data.shape[0]
    accuracy.append(current_accuracy)
    
    print("Current accuracy : ", current_accuracy)


# In[19]:

np.mean(accuracy)

