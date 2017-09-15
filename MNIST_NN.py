
# coding: utf-8

# In[55]:

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')
#딥러닝


# In[56]:

mnist = fetch_mldata('MNIST original', data_home='./')


# In[57]:

mnist.data.shape


# In[58]:

mnist.target.shape


# In[59]:

idx = list(range(70000))
random.shuffle(idx)


# In[60]:

idx = list(range(70000))
random.shuffle(idx)
train_data = mnist.data[idx[0:50000]]
train_label = mnist.target[idx[0:50000]]

test_data = mnist.data[idx[50000:70000]]
test_label = mnist.target[idx[50000:70000]]


# In[61]:

#shape함수 통해 0번째부터 9번째까지 zeros함수이용해 0으로 채워주고
#필요한 숫자만 1로 해놓으면 그 숫자 들어갈때 그 숫자로 변함
one_hot_train_label = np.zeros([train_label.shape[0], 10])
for i in range(train_label.shape[0]):
    one_hot_train_label[i][int(train_label[i])] = 1


# In[62]:

one_hot_train_label


# In[63]:

one_hot_test_label = np.zeros([train_label.shape[0],10])
#shape함수 통해 0번째부터 9번째까지 zero(0)으로 채워주고
#필요한 숫자만 1로 해놓으면 그 숫자 들어갈때 그 숫자로 변함
#test에서는 12345순이아니라 test순대로.
for i in range(test_label.shape[0]):
    one_hot_test_label[i][int(test_label[i])] = 1


# In[64]:

one_hot_test_label


# In[65]:

nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(50, ), random_state=1)
#alpha=최고효율학습점에 도달하기 위해서 조금씩 이동하기위해 필요.
#neural network,mlp(multi layer perceptron)
nn_clf.fit(train_data, one_hot_train_label)


# In[66]:

nn_label = nn_clf.predict(test_data)


# In[67]:

nn_label


# In[68]:

#argmax : 몇번째 입력값이 가장큰지 nn_label [1,0,0,0...]인것은 0번째가 가장크니 0으로 값반환
nn_predict = np.argmax(nn_label, axis=1)
test_answer = np.argmax(one_hot_test_label, axis=1)


# In[88]:

sum (nn_predict==test_answer) / one_hot_test_label.shape[0]

