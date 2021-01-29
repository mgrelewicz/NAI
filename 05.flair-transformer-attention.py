#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

# data = pd.read_csv(('all_scraped_articles_clean.csv'), sep=',', header=None, 
#                    encoding='utf-8')
text = pd.read_csv(('all_scraped_articles.csv'), sep=',', header=None, encoding='utf-8').iloc[1:-1,1] #[row,col]
titles = pd.read_csv(('all_scraped_articles.csv'), sep=',', header=None, encoding='utf-8').iloc[1:-1,0]

# print(data.iloc[0,0])
#print(titles[:2])
#print(text)


# In[5]:


len_text = text.size
print("\nDataset size: ", len_text)
train_size = int(len_text*0.8)
print("Training size: ", train_size)
test_size = int(len_text - train_size)
print("Test size: ", test_size, "\n")

train_texts = []
train_labels = []
test_texts = []
test_labels = []

for sentence in text[:train_size]:
    train_texts.append(sentence)
for sentence in titles[:train_size]:   
    train_labels.append(sentence)
for sentence in text[:test_size]:
    test_texts.append(sentence)
for sentence in titles[:test_size]:
    test_labels.append(sentence)

#print(train_texts[9797])
#print(train_labels[9797])
#print(test_texts[2449])
#print(test_labels[2449])


# In[6]:


## Do wektoryzacji:
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
# wektory glove o niższej liczbie wymiarów mogą dać lepsze wyniki:
embeddings = WordEmbeddings('glove')

train_sentences = []
test_sentences = []
for text in train_texts:
  train_sentences.append(Sentence(text))
for text in test_texts:
  test_sentences.append(Sentence(text))

#print(test_sentences)

embeddings.embed(train_sentences)
embeddings.embed(test_sentences)


# In[7]:


import numpy as np

vectorized_texts_train = np.array([np.array([np.array(token.embedding) for  token in sentence]) for sentence in train_sentences ])
vectorized_texts_test = np.array([np.array([np.array(token.embedding) for  token in sentence]) for sentence in test_sentences ])

from keras.preprocessing import sequence
# przycinanie (padding)
vectorized_texts_train = sequence.pad_sequences(vectorized_texts_train,20, dtype='float32', padding='post', truncating='post')
vectorized_texts_test = sequence.pad_sequences(vectorized_texts_test,20, dtype='float32', padding='post', truncating='post')

print(vectorized_texts_test.shape)

classes = list(set(train_labels))
print(classes)

nb_classes = len(set(train_labels))

train_labels = list(map(lambda x: classes.index(x),train_labels ))
test_labels = list(map(lambda x: classes.index(x),test_labels ))

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(test_labels)


# In[8]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[9]:


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        #2
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        #3
        scaled_score = score / tf.math.sqrt(dim_key)
        #4
        weights = tf.nn.softmax(scaled_score, axis=-1)
        #5
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


# In[10]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[11]:


embed_dim = 100  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
maxlen = 20

inputs = layers.Input(shape=(maxlen,embed_dim))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(nb_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()


# In[12]:


model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    vectorized_texts_train, train_labels, batch_size=32, epochs=50, validation_data=(vectorized_texts_test, test_labels)
)


# In[15]:


import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[16]:


model.save('1st_model')


# In[ ]:





# In[ ]:




