#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import this libraryies just for practice, i dont need all of them
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
import matplotlib.pyplot as plt #used for potting graphs 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #data visualization library
# from sklearn.model_selection import train_test_split # used for splitting training and testing data
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
# from sklearn.metrics import confusion_matrix
# from sklearn.impute import SimpleImputer # used for handling missing data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
# from sklearn.preprocessing import StandardScaler # used for feature scaling


# In[2]:


df = pd.read_csv("C:/Users/Mayuresh/OneDrive/Documents/MSc_VIT/SEM 2/EDA J component/kaggle dataset.csv")
df


# In[4]:


# We're checking how many percentages we have
df['Type'].value_counts(normalize = True) * 100


# In[5]:


# We create a new Total column as a collection of spending over the years for each state
df1 = df.assign(Total = df.sum(axis=1))
df1.head(5)


# In[6]:


# We replace NaN values with zero
df1.fillna(0, inplace=True)
df1.head(5)


# In[7]:


df1.describe()


# In[8]:


# We divide values by 10^9 to get values in trillions of USD (more extensive)
columns=[str(i) for i in list((range(1960,2019)))]
columns=columns+["Total"]
for i in columns:
    df1[i]=df1[i]/1.e+9
df1=np.round(df1, decimals=2)
df1.head()


# In[9]:


# Sort by total consumption for Country type
df1.sort_values(by=['Type','Total'],ascending=[False,False],inplace=True)
df1=df1[df1['Type'].str.contains("Country")]


# In[10]:


# Top 20 countries in terms of total consumption with the elimination of unnecessary attributes
df2 = df1[:20]
df2
df3 = df2.drop(['Indicator Name', 'Code', 'Type'], axis=1)
new = df3.reset_index(drop=True)
new.head(20)


# In[11]:


# Visualization
plt.figure(figsize=(12,8))
sns.barplot(x = 'Total', y = 'Name', data = df3)
plt.title('Total Millitary Spending from 1960 to 2018')
plt.xlabel('Total in bilions USD')
plt.ylabel('Countries')
plt.grid()


# In[12]:


fig = px.pie(df2, values='Total', names='Name', title='Total military spendings in percentage from 1960 to 2018 ')
fig.show()


# In[13]:


fig = px.pie(df2, values='1960', names='Name', title='Military spendings in percentage in 1960')
fig.show()


# In[14]:


fig = px.pie(df2, values='2018', names='Name', title='Military spendings in percentage in 2018')
fig.show()


# In[15]:


fig = px.scatter_geo(df2, locations="Code",hover_name="Name")
fig.update_layout(title="First 20 most powerful country")
fig.show()


# In[16]:


fig = px.scatter_geo(df2, locations = 'Code',hover_name="Name",size = '2018')
fig.show()


# In[17]:


df4 = df3.drop(['Total'], axis=1)


# In[18]:


Top20 = df4.set_index('Name')
Top20.index = Top20.index.rename('Year')
Top20 = Top20.T
Top20.head()


# In[19]:


plt.figure(figsize=(20,10))
plt.plot(Top20.index, Top20.values)
plt.ylabel('Spendings through year (in bilion USD)')
plt.title('Top 20 Countries  in Military Expenditure ')
plt.xticks(rotation=45)
plt.legend(Top20.columns)
plt.grid(True)
plt.show()


# In[ ]:




