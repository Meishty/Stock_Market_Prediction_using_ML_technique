#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt


# In[93]:


df=pd.read_excel("/home/pogo/Desktop/Ba/Desktop/Books/Capstone/Phase 2/Parallel/Data/CALCULATIONS/MMYT_Jan.xlsx")
df=df.dropna()
pos=df[df['Date'].str.contains("Jan")]
pos


# In[94]:


pos1=pos[['Date','Class Label']]
#pos1['Date']=pd.to_datetime(pos1['Date'])
#chart=pos['Class Label'].value_counts().reset_index().rename(columns={"Class Label":"Count","index":"Class Label"})


# In[95]:


ex=pos1.fillna('').groupby('Date').agg(lambda x:sum(list(x))).reset_index()
final=[]
for i in ex['Class Label'].tolist():
    if i>0:
        final.append(1)
    elif i==0:
        final.append(0)
    else:
        final.append(-1)
ex['Final Class Label']=final
ex


# In[96]:


import seaborn as sns

c1=ex[['Date','Final Class Label']]
#pos1['Date'] = pd.DatetimeIndex(pos1.Date)
c1 = c1.set_index('Date')

fig, ax = plt.subplots(figsize=(14,4))
c1.plot(ax=ax)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 2, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




