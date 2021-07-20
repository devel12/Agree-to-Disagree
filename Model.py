#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install transformers')
# get_ipython().system('pip install sentencepiece')


# In[2]:


from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import pipeline
import os


# In[3]:


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')


# In[4]:


neigh = joblib.load('filename.pkl')


# In[5]:


from bs4 import BeautifulSoup
# function to remove HTML tags
def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()


import unicodedata
# function to remove accented characters
def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text
# call function


# In[6]:


def clean(incoming):
  var = []
  for sen in incoming:
    sen = remove_html_tags(sen)
    sen = remove_accented_chars(sen)
    var.append(sen)

  return var


# In[7]:


## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
#summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")


# In[8]:


def listify(lis):
  txt = []

  for x in lis:
    if len(x.split()) > 400:
      txt.append(summarizer(x, max_length=100, min_length=5, do_sample=False)[0]['summary_text'])
    
    elif len(x.split()) > 200:
      txt.append(summarizer(x, max_length=70, min_length=5, do_sample=False)[0]['summary_text'])
    
    elif len(x.split()) > 100:
      txt.append(summarizer(x, max_length=60, min_length=5, do_sample=False)[0]['summary_text'])

    elif len(x.split()) > 75:
      txt.append(summarizer(x, max_length=50, min_length=5, do_sample=False)[0]['summary_text'])

    else :
      txt.append(summarizer(x, max_length=40, min_length=5, do_sample=False)[0]['summary_text'])
  
  return txt


# In[9]:


def getPreds(txt):
  encoded_input = tokenizer(txt, padding=True, truncation=True, return_tensors='pt')

  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)
  bert_embeddings = model_output[0]
  cls = []

  for x in bert_embeddings:
    cls.append(x[0])
  cls = torch.stack(cls)

  preds = neigh.predict(cls)
  return preds


# In[10]:


def getGradeAndScore(preds):
  good = 0
  bad = 0
  neutral = 0
  blocker = 0

  for c in preds :
    if c == 0:
      good +=1
  
    elif c == 1:
      neutral +=1
  
    elif c == 2:
      bad +=1
  
    elif c == 3:
      blocker +=1

  score = good - bad - 3*blocker

  grade = ""
  if score < -10:
    grade = "E"

  elif score > -10 and blocker > 0 :
    grade = "D"

  elif score > -10 and score < -4 and blocker ==0 :
    grade = "C"

  elif score > -4 and blocker == 0 and bad > 0 :
    grade = "B"

  else :
    grade = "A"

  return (score,grade)


# In[11]:


class Model:
  def __init__(self, incoming = None):
    self.incoming = None
  def predict(self,incoming):
    self.incoming= incoming    
    self.incoming = clean(self.incoming)    
    test_list = listify(self.incoming)
    preds = getPreds(test_list)    
    return getGradeAndScore(preds)



# In[12]:


# model_1 =  Model()
# model_1.predict(['<p>we sell you personal data without your permission','we can remove your account without your approval'])


# In[13]:


# import pickle
# mod = pickle.load(open('modelnew.pkl', 'rb'))


# In[ ]:




