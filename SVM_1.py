#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import datetime as dt
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np


# In[9]:


tcs_nov=pd.read_csv("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/STOCKS/AIRTEL_All.csv",usecols=['Date','Close'])
tcs_oct=pd.read_csv("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/STOCKS/AIRTEL.csv",usecols=['Date','Close'])
df_read=pd.concat([tcs_nov,tcs_oct],axis=0)
df_read


# In[10]:


df_read=df_read.reset_index(drop=True)
df_read.dropna(inplace=True)
df_read1=df_read.copy()


# In[11]:


fore=df_read1[df_read1['Date']>='2021-01-01'].shape[0]
df_read1['Prediction'] = df_read1[['Close']].shift(-fore)
df_read1


# In[12]:


def date_float(x):
    time_tuple = time.strptime(x, "%Y-%m-%d")
    timestamp = time.mktime(time_tuple)
    return timestamp


# In[13]:


X1=df_read1[['Close','Prediction']]
X = np.array(X1.drop(['Prediction'],1))
X = X[:-fore]
print(X)


# In[14]:


y = np.array(df_read1['Prediction'])
y = y[:-fore]
print(y)


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.15,random_state=42)


# In[16]:


svr = SVR(kernel='rbf',gamma=0.005, C=100)
svr.fit(x_train, y_train)


# In[17]:


svm_confidence = svr.score(x_test, y_test)
print("svm confidence: ", round(svm_confidence,2))


# In[18]:


X2=df_read1[['Close','Prediction']]
forecast = np.array(X2.drop(['Prediction'],1))[-fore:]
print(forecast)


# In[19]:


svm_prediction = svr.predict(forecast)
print(svm_prediction)


# In[20]:


a1=df_read1[df_read1['Date']>='2021-01-01']
a1


# In[21]:


def p_chan(x,y):
    return ((y-x)/x)*100


# In[22]:


c=pd.DataFrame()
c['Date']=list(a1['Date'])
c['Predicted']=list(svm_prediction)
c['Actual Close']=list(a1['Close'])


# In[23]:


diff=[]
for i in range(len(list(c['Predicted']))):
    diff.append(p_chan(list(c['Actual Close'])[i],list(c['Predicted'])[i]))
c['Difference']=diff
c


# In[24]:


def accuracy(x):
    if x<0:
        return 100+x
    else:
        return 100-x


# In[25]:


accu=[]
for i in range(len(list(c['Difference']))):
    accu.append(accuracy(list(c['Difference'])[i]))
c['Accuracy']=accu
c


# In[26]:


print("Accuracy is",c['Accuracy'].mean())


# In[27]:


#svm_confidence,c['Accuracy'].mean()
import seaborn as sns

c1=c[['Date','Predicted','Actual Close']]
c1['Date'] = pd.DatetimeIndex(c1.Date)
c1 = c1.set_index('Date')

fig, ax = plt.subplots(figsize=(9,4))
c1.plot(ax=ax)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 1.5, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




