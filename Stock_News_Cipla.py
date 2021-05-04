#!/usr/bin/env python
# coding: utf-8

# In[3]:


from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd


# In[4]:


googlenews=GoogleNews(start='01/01/2021',end='02/01/2021')
googlenews.search('Cipla')
df_f=pd.DataFrame()
for i in range(1,31):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
    df_f=pd.concat([df,df_f])
    print(i)
    
googlenews.clear()


# In[5]:


googlenews=GoogleNews(start='01/01/2021',end='02/01/2021')#Bharti Airtel
googlenews.search('Cipla Limited')
df_f1=pd.DataFrame()
for i in range(1,31):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
    df_f1=pd.concat([df,df_f1])
    print(i)
    
googlenews.clear()


# In[6]:


df=pd.concat([df_f,df_f1],axis=0)
df=df.drop_duplicates().sort_values(by='datetime',ascending=True)
df_final=df.copy()
df_final


# In[7]:


df_final1=df_final[df_final['date'].str.contains("Jan")].copy()


# In[8]:


l=[]
df_final1=df_final1.fillna('')
for ind in range(df_final1.shape[0]):
    d={}
    try:
        article = Article(df_final1['link'].tolist()[ind])
        article.download()
        article.parse()
        article.nlp()
        d['Date']=df_final1['date'].tolist()[ind]
        d['Media']=df_final1['media'].tolist()[ind]
        d['Title']=article.title
        l.append(d)
        print("Done",ind)
        
    except:
        print(ind)
    #print(ind)
news_df=pd.DataFrame(l)
news_df.to_excel("Cipla_articles_final_Jan.xlsx")


# In[ ]:




