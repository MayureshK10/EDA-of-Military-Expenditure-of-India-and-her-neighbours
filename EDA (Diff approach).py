#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary libraries
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
import matplotlib.pyplot as plt #used for potting graphs 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #data visualization library


# In[2]:


df = pd.read_csv("C:/Users/Mayuresh/OneDrive/Documents/MSc_VIT/SEM 2/EDA J component/kaggle dataset.csv")
df


# In[3]:


# We're checking how many percentages we have
df['Type'].value_counts(normalize = True) * 100


# In[4]:


# We create a new Total column as a collection of spending over the years for each state
df1 = df.assign(Total = df.sum(axis=1))
df1.head(5)


# In[5]:


# We replace NaN values with zero
df1.fillna(0, inplace=True)
df1.head(5)


# In[6]:


df1.describe()


# In[7]:


# We divide values by 10^9 to get values in trillions of USD (more extensive)
columns=[str(i) for i in list((range(1960,2019)))]
columns=columns+["Total"]
for i in columns:
    df1[i]=df1[i]/1.e+9
df1=np.round(df1, decimals=2)
df1.head()


# In[8]:


# Sort by total consumption for Country type
df1.sort_values(by=['Type','Total'],ascending=[False,False],inplace=True)
df1=df1[df1['Type'].str.contains("Country")]


# In[9]:


# Top 20 countries in terms of total consumption with the elimination of unnecessary attributes
df2 = df1[:20]
df2
df3 = df2.drop(['Indicator Name', 'Code', 'Type'], axis=1)
new = df3.reset_index(drop=True)
new.head(20)


# In[10]:


# Visualization
plt.figure(figsize=(12,8))
sns.barplot(x = 'Total', y = 'Name', data = df3)
plt.title('Total Millitary Spending from 1960 to 2018')
plt.xlabel('Total in bilions USD')
plt.ylabel('Countries')
plt.grid()


# In[11]:


fig = px.pie(df2, values='Total', names='Name', title='Total military spendings in percentage from 1960 to 2018 ')
fig.show()


# In[12]:


fig = px.pie(df2, values='1960', names='Name', title='Military spendings in percentage in 1960')
fig.show()


# In[13]:


fig = px.scatter_geo(df2, locations="Code",hover_name="Name")
fig.update_layout(title="First 20 most powerful country")
fig.show()


# In[14]:


fig = px.scatter_geo(df2, locations = 'Code',hover_name="Name",size = '2018')
fig.show()


# In[15]:


df4 = df3.drop(['Total'], axis=1)


# In[16]:


Top20 = df4.set_index('Name')
Top20.index = Top20.index.rename('Year')
Top20 = Top20.T
Top20.head()


# In[17]:


plt.figure(figsize=(20,10))
plt.plot(Top20.index, Top20.values)
plt.ylabel('Spendings through year (in bilion USD)')
plt.title('Top 20 Countries  in Military Expenditure ')
plt.xticks(rotation=45)
plt.legend(Top20.columns)
plt.grid(True)
plt.show()


# In[18]:


# Percentage growth in consumption by year for the top 20 countries
PercUSA = (Top20['United States'].iloc[-1] - Top20['United States'].iloc[0])*100/Top20['United States'].iloc[0]
PercChina = (Top20['China'].iloc[-1] - Top20['China'].iloc[29])*100/Top20['China'].iloc[29]
PercRUS = (Top20['Russian Federation'].iloc[-1] - Top20['Russian Federation'].iloc[33])*100/Top20['Russian Federation'].iloc[33]
PercISR = (Top20['Israel'].iloc[-1] - Top20['Israel'].iloc[0])*100/Top20['Israel'].iloc[0]
PercITA = (Top20['Italy'].iloc[-1] - Top20['Italy'].iloc[0])*100/Top20['Italy'].iloc[0]
PercJPN = (Top20['Japan'].iloc[-1] - Top20['Japan'].iloc[0])*100/Top20['Japan'].iloc[0]
PercNET = (Top20['Netherlands'].iloc[-1] - Top20['Netherlands'].iloc[0])*100/Top20['Netherlands'].iloc[0]
PercPOL = (Top20['Poland'].iloc[-1] - Top20['Poland'].iloc[0])*100/Top20['Poland'].iloc[0]
PercSAU = (Top20['Saudi Arabia'].iloc[-1] - Top20['Saudi Arabia'].iloc[0])*100/Top20['Saudi Arabia'].iloc[0]
PercKOR = (Top20['South Korea'].iloc[-1] - Top20['South Korea'].iloc[0])*100/Top20['South Korea'].iloc[0]
PercSPA = (Top20['Spain'].iloc[-1] - Top20['Spain'].iloc[0])*100/Top20['Spain'].iloc[0]
PercTUR = (Top20['Turkey'].iloc[-1] - Top20['Turkey'].iloc[0])*100/Top20['Turkey'].iloc[0]
PercUK = (Top20['United Kingdom'].iloc[-1] - Top20['United Kingdom'].iloc[0])*100/Top20['United Kingdom'].iloc[0]
PercAUS = (Top20['Australia'].iloc[-1] - Top20['Australia'].iloc[0])*100/Top20['Australia'].iloc[0]
PercBRA = (Top20['Brazil'].iloc[-1] - Top20['Brazil'].iloc[0])*100/Top20['Brazil'].iloc[0]
PercCAN = (Top20['Canada'].iloc[-1] - Top20['Canada'].iloc[0])*100/Top20['Canada'].iloc[0]
PercFRA = (Top20['France'].iloc[-1] - Top20['France'].iloc[0])*100/Top20['France'].iloc[0]
PercGER = (Top20['Germany'].iloc[-1] - Top20['Germany'].iloc[0])*100/Top20['Germany'].iloc[0]
PercIND = (Top20['India'].iloc[-1] - Top20['India'].iloc[0])*100/Top20['India'].iloc[0]
PercIRA = (Top20['Iran'].iloc[-1] - Top20['Iran'].iloc[0])*100/Top20['Iran'].iloc[0]


# In[19]:


data = [['United States', PercUSA], ['China', PercChina], ['France', PercFRA], ['United Kingdom', PercUK], ['Germany', PercGER], ['Japan', PercJPN], ['Saudi Arabia', PercSAU], ['Russian Federation', PercRUS], ['India', PercIND], ['Italy', PercITA], ['South Korea', PercKOR], ['Brazil', PercBRA], ['Canada', PercCAN], ['Spain', PercSPA], ['Australia', PercAUS], ['Iran', PercIRA], ['Israel', PercISR], ['Turkey', PercTUR], ['Poland', PercPOL], ['Netherlands', PercNET]]


# In[20]:


percdf= pd.DataFrame(data, columns=['Country', 'Percentage growth'])
percdf.head(20)


# In[21]:


plt.figure(figsize=(15,8))
sns.barplot(x = 'Country', y = 'Percentage growth', data = percdf)
plt.xticks(rotation=45)


# In[22]:


model = percdf.join(new['Total'])
model


# In[23]:


plt.figure(figsize=(15,8))
sns.barplot(x = 'Total', y = 'Percentage growth', hue='Country',data = model)
plt.xticks(rotation = 90)


# In[ ]:




