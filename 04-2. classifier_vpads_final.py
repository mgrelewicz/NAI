# -*- coding: utf-8 -*-

"""
# Edyta Bartoś, Marcin Grelewicz
#
# Dataset: Vicon Physical Action Data Set from: 
#    http://sites.google.com/site/ttheod/
# for building the classifier SVM-> Support Vector Machine, which is commonly 
# used for supervised learning to analyze data and recognize patterns, 
# by dividing data on hyperplane with maximum margin possible. 
# 
# Attribute Information: Each file in the dataset contains 28 columns 
# The dataset consists of 10 normal, and 10 aggressive physical actions.
# Normal: Bowing, Clapping, Handshaking, Hugging, Jumping, Running, Seating, 
# Standing, Walking, Waving
# Aggressive: Elbowing, Frontkicking, Hamering, Headering, Kneeing, Pulling, 
# Punching, Pushing, Sidekicking, Slapping
# Each of them consist of 27 atributes. 
#
# We've chosen 2 normal and 2 aggressive physical actions for our 
# classifier purposes. 
"""

# In[1]:


# load datasets
import pandas as pd

aggressive1 = pd.read_csv("Slapping.txt",sep='\s+', header=None)
normal1 = pd.read_csv("Clapping.txt",sep='\s+', header=None)
aggressive2 = pd.read_csv("Punching.txt",sep='\s+', header=None)
normal2 = pd.read_csv("Running.txt",sep='\s+', header=None)

normal1 = normal1.dropna()
aggressive1 = aggressive1.dropna()
normal2 = normal2.dropna()
aggressive2 = aggressive2.dropna()


# In[2]:


#print 1 dataset as example:
print(aggressive2)


# In[3]:


#print dataset normal
print(normal1)


# In[4]:


#we create 2 classes "agressive" and "normal" with binary description

n1_len = len(normal1)
n2_len = len(normal2)
a1_len = len(aggressive1)
a2_len = len(aggressive2)

normal1_class = [0 for i in range(n1_len)]  
aggressive1_class = [1 for i in range(a1_len)]
normal2_class = [0 for i in range(n2_len)]  
aggressive2_class = [1 for i in range(a2_len)]

dat1_0 = pd.DataFrame(normal1_class, columns=None)
dat1_1 = pd.DataFrame(aggressive1_class, columns=None)
dat2_0 = pd.DataFrame(normal2_class, columns=None)
dat2_1 = pd.DataFrame(aggressive2_class, columns=None)

aggressive_tmp1 = pd.concat([aggressive1, dat1_1], axis=1)
normal_tmp1 = pd.concat([normal1, dat1_0], axis=1)
aggressive_tmp2 = pd.concat([aggressive2, dat2_1], axis=1)
normal_tmp2 = pd.concat([normal2, dat2_0], axis=1)

aggressive = pd.concat([aggressive_tmp1, aggressive_tmp2], axis=0)
normal = pd.concat([normal_tmp1, normal_tmp2], axis=0)


# In[5]:


#after adding last column
print(normal)


# In[6]:


# Merge agressive and normal dataset 
data = aggressive.iloc[:, 1:].merge(normal.iloc[:, 1:], how='outer')
#print('Data shape: ', data.shape)


# In[7]:


#all merged datasets, after adding last column
data


# In[8]:


# split data for X- atributes, Y-quality (reference)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[9]:


## Pearson pairwise correlation to show which features are most correlated:
correlations = X.corrwith(y)
correlations.sort_values(inplace=True)

fields = correlations.map(abs).sort_values().iloc[-5:].index
## print top absolute correlations:
print('top correlations: ', fields)

## Plot correlation bars:
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='Pearson correlation')


# In[10]:


# Aggressive- 1, normal- 0;
import seaborn as sns
sns.countplot(y)


# In[11]:


from sklearn.preprocessing import StandardScaler

## Scale values:
scaler = StandardScaler()
X = scaler.fit_transform(X[fields])
X = pd.DataFrame(X)
y = pd.DataFrame(y)


# In[12]:


# Return contigous flattened array (1 dimensional array)
import numpy as np
y = np.ravel(y)


# In[13]:


from sklearn.model_selection import train_test_split

## Split data to train and test portions:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=50, 
                                                    shuffle=True)

## Check if train and test have the same size:
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[14]:


## Prediction without tunning hyperparameters
from sklearn import svm
from sklearn.metrics import classification_report

svc = svm.SVC(kernel='linear', gamma=0.1, C=10).fit(X_train, y_train)
y_pred = svc.predict(X_test)

print(classification_report(y_test, y_pred))


# In[15]:


from sklearn.metrics import confusion_matrix

## Creation of confusion matrix
metrics = list()
cm = dict()

## Confusion matrix:
cm = confusion_matrix(y_test, y_pred)


# In[16]:


from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score
import matplotlib.pyplot as plt

## Precision, recall, f-score from the multi-class support function
precision, recall, fscore, _ = score(y_test, y_pred, average='weighted')

## The usual way to calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                          'fscore':fscore, 'accuracy':accuracy}, 
                         name='Model'))

metrics = pd.concat(metrics, axis=1)

print(metrics)

plot = sns.heatmap(cm, annot=True, fmt='d');
plt.tight_layout


# In[17]:


## (one-time run) GridSearchCV to tune hyperparameters for the SVM:

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[.001, .005, .01, .1, .5], 
              'gamma':[1, 2, 4, 5, 7, 10]}

svc = svm.SVC(gamma='scale')
gscv = GridSearchCV(svc, param_grid=parameters, cv=None)
gscv.fit(X_train, y_train)

## printing the best parameters:
print(gscv.best_estimator_)
print(gscv.best_params_)


# In[18]:


## Tunning
svc = svm.SVC(kernel='rbf', gamma=5, C=0.1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)


# In[19]:


## Creation of confusion matrix after fine-tuning 
metrics = list()
cm = dict()

## confusion matrix:
cm = confusion_matrix(y_test, y_pred)


# In[20]:


# Precision, recall, f-score from the multi-class support function
precision, recall, fscore, _ = score(y_test, y_pred, average='weighted')

# The usual way to calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                          'fscore':fscore, 'accuracy':accuracy}, name='Model'))

metrics = pd.concat(metrics, axis=1)

print(metrics)

plot = sns.heatmap(cm, annot=True, fmt='d');
plt.tight_layout()



