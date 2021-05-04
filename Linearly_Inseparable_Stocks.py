#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


# In[2]:


style.use('ggplot')


# In[3]:


df=pd.read_csv("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/STOCKS/TCS_Oct.csv",usecols=['Date','Close'])


# In[4]:


def set_close_label(cl_list):
    lab=[]
    lab.append(0)
    for i in range(1,len(cl_list)):
        if cl_list[i]>cl_list[i-1]:
            lab.append(1)
        elif cl_list[i]<=cl_list[i-1]:
            lab.append(0)
    return lab


# In[5]:


df['Label']=set_close_label(df['Close'].tolist())
df


# In[13]:


import seaborn as sns


# In[18]:


plot1_img=sns.scatterplot(x="Date", y="Close", data=df, hue="Label")
figure = plot1_img.get_figure()    
figure.savefig('svm_conf.png', dpi=400)


# In[ ]:





# In[ ]:




