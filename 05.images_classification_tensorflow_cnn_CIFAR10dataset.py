#!/usr/bin/env python
# coding: utf-8

# """
# Edyta Bartoś, Marcin Grelewicz,
# The use of Convolutional Neural Network (CNN) for images classification
# 
# based on Tensorflow tutorial:
# https://www.tensorflow.org/tutorials/images/cnn
# """

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Dataset: CIFAR10 
# The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each class. 
# The dataset is divided into 50,000 training images and 10,000 testing images. 
# The classes are mutually exclusive and there is no overlap between them.
# Animals: frog, dog, cat, horse, and bird correspond to indexes 6, 5, 3, 7, and 2.

# In[2]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# In[3]:


test_labels


# In[4]:


# Concatenate train and test images
X = np.concatenate((train_images, test_images))
y = np.concatenate((train_labels,test_labels))

# Check shape
print(X.shape) # (60000, 32, 32, 3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=1234)

# Check shape
print(X_train.shape) # (50000, 32, 32, 3)

# View first image
plt.imshow(X_train[0])
plt.show()


# In[5]:


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[10]:


## Ploting the first 25 images from the training set and display the class name below each image
class_names = ['n/a', 'n/a', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'n/a', 'n/a']

plt.figure(figsize=(12,12))
for i in range(20):
    plt.subplot(5,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # hence the extra index below
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[11]:


## defining the convolutional base using a common pattern: 
## a stack of Conv2D and MaxPooling2D layers.
## As input, a CNN takes tensors of shape (image_height, image_width, color_channels), 
## ignoring the batch size

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[12]:


#display the architecture of the model so far
model.summary()


# In[13]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# The final architecture of the model
model.summary()


# The (4, 4, 64) outputs were flattened into vectors of shape (1024) before going through two Dense layers.

# In[14]:


#Compile and train the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# In[15]:


#Evaluate the model

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# In[16]:


print(test_acc)


# In[17]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# In[ ]:


model.save('D:/PJATK/EDUX/_7semestr/NAI/models/cifar/', include_optimizer=True)


# In[18]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


# In[19]:


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# In[20]:


predictions = probability_model.predict(test_images)


# In[21]:


predictions[3]


# In[22]:


np.argmax(predictions[3])


# In[23]:


test_labels[0]


# In[24]:


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(100*np.max(predictions_array),), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[25]:


i = 3
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
plt.xlabel(class_names[train_labels[i][0]])

plt.show()


# In[ ]:





# In[ ]:




