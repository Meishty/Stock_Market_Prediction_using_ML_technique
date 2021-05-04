#!/usr/bin/env python
# coding: utf-8

# In[11]:


from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd


# In[12]:


googlenews=GoogleNews(start='01/01/2021',end='02/01/2021')
googlenews.search('MakeMyTrip')
df_f1=pd.DataFrame()
for i in range(1,31):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
    df_f1=pd.concat([df,df_f1])
    print(i)
googlenews.clear()


# In[14]:


googlenews=GoogleNews(start='01/01/2021',end='02/01/2021')
googlenews.search('intitle:MakeMyTrip Limited')
df_f2=pd.DataFrame()
for i in range(1,31):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
    df_f2=pd.concat([df,df_f2])
    print(i)
googlenews.clear()


# In[ ]:


googlenews=GoogleNews(start='01/01/2021',end='02/01/2021')
googlenews.search('intitle: MAKEMYTRIP LIMITED')
df_f3=pd.DataFrame()
for i in range(1,31):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
    df_f3=pd.concat([df,df_f3])
    print(i)
    
googlenews.clear()


# In[15]:


df=pd.concat([df_f1],axis=0)#,df_f2,df_f3
df=df.drop_duplicates().sort_values(by='datetime',ascending=True)
df_final=df.copy()
df_final


# In[16]:


df_final1=df_final[df_final['date'].str.contains("Jan")].copy()


# In[17]:


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
        d['Summary']=article.summary
        l.append(d)
        print("Done",ind)
        
    except:
        print(ind)
    #print(ind)
news_df=pd.DataFrame(l)
news_df.to_excel("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/NEWS/MakeMyTrip_Updated_articles_final_Jan.xlsx")


# In[ ]:




