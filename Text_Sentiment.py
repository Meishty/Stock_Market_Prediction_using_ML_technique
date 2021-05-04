#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import functools
import operator
import re
import textblob
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment


# In[29]:


df=pd.read_excel("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/NEWS/MakeMyTrip_Updated_articles_final_Jan.xlsx")
df=df.dropna()
df['Date']=list(df['Date'].apply(lambda x:x.strip()))
df.columns


# In[30]:


df=df.astype(str)
df['Title_Refined']=list(df['Title'].apply(lambda x:str.lower(x)))
df['Title_Refined']


# In[31]:


def define_sentiment(x):
#        sid_obj = SentimentIntensityAnalyzer()
#        sentiment_dict = sid_obj.polarity_scores(x)
#        return sentiment_dict['compound']
         senti=textblob.TextBlob(x)
         return senti.sentiment[0]


# In[32]:


compound=[]
for i in df['Title_Refined'].tolist():
    compound.append(define_sentiment(i))


# In[33]:


df['Sentiment Score']=compound
df


# In[34]:


#df[df['Sentiment Score']>0].groupby('Date').count()
def label_setup(x):
    if x==0:
        return 0
    elif x>0:
        return 1
    else:
        return -1


# In[35]:


df['Class Label']=list(df['Sentiment Score'].apply(lambda x:label_setup(x)))
df


# In[36]:


df.to_excel("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/CALCULATIONS/MMYT_Jan.xlsx",index=0)


# In[ ]:





# In[ ]:




