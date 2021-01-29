#!/usr/bin/env python
# coding: utf-8

# """
# 
# Edyta Bartoś i Marcin Grelewicz, 
# 
# usage of red and white data from: 
# https://machinelearningmastery.com/standard-machine-learning-datasets/, 
# for building the classifier SVM-> Support Vector Machine, which is commonly 
# used for supervised learning to analyze data and recognize patterns, by dividing
# data on hyperplane with maximum margin possible. 
# 
# Two datasets have 12 atribute informations, icluding quality, which is our reference. 
# More info about atributes of wine here: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# 
# """

# In[65]:


# Import Required Libraries 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 


# In[66]:


# Read in white wine data 
data_red = pd.read_csv('winequality-red.csv', sep= ";") 
  
# Read in red wine data 
data_white = pd.read_csv('winequality-white.csv', sep= ";") 


# In[67]:


# show sample of data
data_red.sample(5) 


# In[68]:


# Add `type` column to `red` with price one 
data_red['type'] = 1

# Add `type` column to `white` with price zero 
data_white['type'] = 0

# Append `white` to `red` 
all_data = data_red.append(data_white, ignore_index = True) 


# In[69]:


all_data.tail(5)


# In[70]:


# Import `train_test_split` from `sklearn.model_selection` 
from sklearn.model_selection import train_test_split 
#splitting data to X ve y
#X = all_data.drop(['quality', 'type'], axis = 1)
y = np.ravel(all_data.type)
X = all_data.iloc[:, 0:11] 
#y = np.ravel(all_data.type) 

# Splitting the data set for training and validating 
X_train, X_test, y_train, y_test = train_test_split( 
		X, y, test_size = 0.25, random_state = 50) 


# In[71]:


X_test.head()


# In[72]:


# Feature Scaling to X_train and X_test to classify better.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[73]:


print(X_test)


# In[74]:


# Import `Sequential` from `keras.models` 
from keras.models import Sequential 

# Import `Dense` from `keras.layers` 
from keras.layers import Dense 

# Initialize the constructor 
model = Sequential() 

# Add an input layer 
model.add(Dense(12, activation ='relu', input_shape =(11, ))) 

# Add one hidden layer 
model.add(Dense(9, activation ='relu')) 

# Add an output layer 
#model.add(Dense(1, activation ='sigmoid')) 

# Model output shape 
model.output_shape

# Model summary 
model.summary()

# Model config 
model.get_config() 

# List all weight tensors 
model.get_weights() 
model.compile(loss ='binary_crossentropy', 
optimizer ='adam', metrics =['accuracy']) 


# In[75]:


from datetime import datetime
import tensorflow as tf

from tensorflow import keras

from matplotlib import pyplot as plt
history = model.fit(X_train, y_train, validation_split = 0.5, epochs=6, batch_size=1)

# Predicting the Value 
y_pred = model.predict(X_test) 
print(y_pred) 


# Here we are going to see 1 model without activator "sigmoid". We are using only RELU with 21 layers.

# In[76]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[77]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[78]:


# Import `Sequential` from `keras.models` 
from keras.models import Sequential 

# Import `Dense` from `keras.layers` 
from keras.layers import Dense 

# Initialize the constructor 
model = Sequential() 

# Add an input layer 
model.add(Dense(12, activation ='relu', input_shape =(11, ))) 

# Add one hidden layer 
#model.add(Dense(9, activation ='relu')) 

# Add an output layer 
model.add(Dense(1, activation ='sigmoid')) 

# Model output shape 
model.output_shape

# Model summary 
model.summary()

# Model config 
model.get_config() 

# List all weight tensors 
model.get_weights() 
model.compile(loss ='binary_crossentropy', 
optimizer ='adam', metrics =['accuracy']) 

from datetime import datetime
import tensorflow as tf

from tensorflow import keras

from matplotlib import pyplot as plt
history = model.fit(X_train, y_train, validation_split = 0.5, epochs=6, batch_size=1)

# Predicting the Value 
y_pred = model.predict(X_test) 
print(y_pred) 


# In[ ]:


Here we are going to see 1 model without 1 activator "RELU", but we are adding "sigmoid" instead. Model has 13 layers.


# In[79]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[80]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

