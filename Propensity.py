#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re
import nltk
import spacy
import string
import requests

import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[2]:


pip install spacy


# In[2]:


#Raw_data = pd.read_csv("twcs.csv")
#Text_data = Raw_data.rename(columns={'title': 'text'})

#Text_data["text"] = Text_data["text"].astype(str)
#Raw_data.head()


# In[10]:


Raw_data = pd.read_csv(r"C:\Users\rpras\Downloads\Dataset\twcs\twcs.csv", nrows=5000)
Text_data = Raw_data[["text"]]

Text_data["text"] = Text_data["text"].astype(str)
Raw_data.shape


# In[11]:


Text_data['text'] = Text_data['text'].apply(lambda x: x.replace(u'\xa0',u' '))# to replace non-breaking space with regular space
Text_data['text'] = Text_data['text'].apply(lambda x: x.replace('\u200a',' '))# to replace hair space with regular space


# ## To make it lower all letters

# In[12]:


Text_data["text"] = Text_data["text"].str.lower() # We do not need function for this part '.lower' will be enough.
Text_data.head()


# ## To remove curse words

# In[14]:


pip install better-profanity


# In[15]:


from better_profanity import profanity
censored = profanity.censor
Text_data["text"] = Text_data["text"].apply(lambda text: censored(text))
Text_data.head()


# ## To remove punctiuation

# In[16]:


punctuation_remove = string.punctuation
def punctuation_remover(text): # This is the Function that we use to remove punctuation
    return text.translate(str.maketrans('', '', punctuation_remove))

Text_data["text"] = Text_data["text"].apply(lambda text: punctuation_remover(text))
Text_data.head()


# ### Those are the whole stop words.

# In[17]:


from nltk.corpus import stopwords
", ".join(stopwords.words('english')) # We will remove the following words.


# ## To remove stop words

# In[18]:


Stop_Words = set(stopwords.words('english'))
def Stopword_remover(text): # This is the Function that we use to remove stopwords.
    return " ".join([x for x in str(text).split() if x not in Stop_Words])

Text_data["text"] = Text_data["text"].apply(lambda text: Stopword_remover(text))
Text_data


# ## Lemmatization

# In[19]:


from nltk.stem import WordNetLemmatizer # We are not sure to use Lemmatization. We will decide after we see the results.
#
Lem = WordNetLemmatizer()
def words_lemmatizer(text):
    return " ".join([Lem.lemmatize(word) for word in text.split()])

Text_data["text"] = Text_data["text"].apply(lambda text: words_lemmatizer(text))
Text_data.head()


# ## html_remover

# In[20]:


def html_remover(headline_text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',headline_text)

Text_data["text"] = Text_data["text"].apply(html_remover)


# ## URL_remover

# In[21]:


def URL_remover(headline_text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', headline_text)
Text_data["text"] = Text_data["text"].apply(URL_remover)


# ## Emoji_remover

# In[22]:


def emoji_remover(data):
    emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoji, '', data)
Text_data["text"] = Text_data["text"].apply(emoji_remover)
print(Text_data.to_string())


# ## Tokenzation

# In[23]:


Tok = Tokenizer(oov_token='<oov>')  #OOV means "out of Vocabulary"
Tok.fit_on_texts(Text_data['text'])
Numbers_of_total_words = len(Tok.word_index) + 1 # To find total number of words.


# In[24]:


print("Total number of different words in the datasets: ", Numbers_of_total_words)
print('We test tokenization in few words.')
print("<oov>: ", Tok.word_index['<oov>'])
print("hope: ", Tok.word_index['hope']) 
print("name: ", Tok.word_index['name'])


# In[25]:


input_seq = []
for line in Text_data['text']:
    List_of_tokens = Tok.texts_to_sequences([line])[0]
    #print(List_of_tokens)
    
    for i in range(1, len(List_of_tokens)):
        n_gram = List_of_tokens[:i+1]
        input_seq.append(n_gram)

print(input_seq)
print("Total number of input sequences: ", len(input_seq))


# In[26]:


# pad sequences 
maxiumum_sequence_length = max([len(x) for x in input_seq])
input_seq = np.array(pad_sequences(input_seq, maxlen=maxiumum_sequence_length, padding='pre'))
input_seq[1]


# In[27]:


# Here we are creating features and labels.
a, label = input_seq[:,:-1],input_seq[:,-1]
b = tf.keras.utils.to_categorical(label, num_classes=Numbers_of_total_words)


# In[28]:


print(a[2])
print(label[2])
print(b[2][1])


# ## Bi- LSTM Neural Network Model training

# In[31]:


model = Sequential()
model.add(Embedding(Numbers_of_total_words, 100, input_length=maxiumum_sequence_length-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(Numbers_of_total_words, activation='softmax'))
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(a, b, epochs=10, verbose=1)
#print model.summary()
print(model)


# In[32]:


input_text = input().strip().lower()
number_of_words = 2 #To decide the number of words to predict.
  
for _ in range(number_of_words):
    List_of_tokens = Tok.texts_to_sequences([input_text])[0]
    List_of_tokens = pad_sequences([List_of_tokens], maxlen=maxiumum_sequence_length-1, padding='pre')
    predicted_word = model.predict(List_of_tokens, verbose=0)
    predicted_word = np.argmax(predicted_word,axis=1) # like the input we have to change output to sequence.
    output_word = "" 
    for word, index in Tok.word_index.items():
        if index == predicted_word:
            output_word = word
            break
    input_text += " " + output_word
print(input_text)


# In[33]:


model.save('saved_model/my_model')


# In[34]:


new_model = tf.keras.models.load_model('saved_model/my_model')


# In[36]:


input_text = input().strip().lower()
number_of_words = 4 #To decide the number of words to predict.
  
for _ in range(number_of_words):
    List_of_tokens = Tok.texts_to_sequences([input_text])[0]
    List_of_tokens = pad_sequences([List_of_tokens], maxlen=maxiumum_sequence_length-1, padding='pre')
    predicted_word = new_model.predict(List_of_tokens, verbose=0)
    predicted_word = np.argmax(predicted_word,axis=1) # like the input we have to change output to sequence.
    output_word = "" 
    for word, index in Tok.word_index.items():
        if index == predicted_word:
            output_word = word
            break
    input_text += " " + output_word
print(input_text)


# In[ ]:




