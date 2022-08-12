#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import this libraryies just for practice, i dont need all of them
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
import matplotlib.pyplot as plt #used for potting graphs 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #data visualization library
from sklearn.model_selection import train_test_split # used for splitting training and testing data
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


# In[3]:


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


# Countries of concern in the study in terms of total consumption with the elimination of unnecessary attributes
df2 = df1.apply(lambda row: row[df['Name'].isin(["India","China","Pakistan","Sri Lanka","Afghanistan","Bangladesh","Nepal"])])
df2
df3 = df2.drop(['Indicator Name', 'Code', 'Type'], axis=1)
new = df3.reset_index(drop=True)
new.head(7)


# In[11]:


# Visualization
plt.figure(figsize=(12,8))
sns.barplot(x = 'Total', y = 'Name', data = df3)
plt.title('Total Millitary Spending from 1960 to 2018')
plt.xlabel('Total in bilions USD')
plt.ylabel('Countries')
plt.grid()


# In[33]:


fig = px.pie(df2, values='Total', names='Name', title='Total military spendings in the subcontinent from 1960 to 2018 ')
fig.show()


# In[13]:


fig = px.pie(df2, values='2018', names='Name', title='Military spendings in percentage in 2018')
fig.show()


# In[16]:


fig = px.scatter_geo(df2, locations="Code",hover_name="Name", scope = 'asia')
fig.update_layout(title="Countries considered for the study")
fig.show()


# In[17]:


fig = px.scatter_geo(df2, locations = 'Code',hover_name="Name",size = '2018',  scope = 'asia')
fig.show()


# In[18]:


df4 = df3.drop(['Total'], axis=1)


# In[34]:


dfnew = df4.set_index('Name')
dfnew.index = dfnew.index.rename('Year')
dfnew = dfnew.T
dfnew.head()


# # Explore this decade wise

# In[20]:


plt.figure(figsize=(20,10))
plt.plot(dfnew.index, dfnew.values)
plt.ylabel('Spendings through year (in bilion USD)')
plt.title(' Military Expenditure of India and her neighbours ')
plt.xticks(rotation=45)
plt.legend(dfnew.columns)
plt.grid(True)
plt.show()


# In[21]:


decades = ["196", "197", "198", "199", "200", "201"]
rows = 2
cols = 3
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,8), sharey=True)
fig.suptitle("Decade wise comparison in trillions of US$", fontsize=16)

for i, decade in enumerate(decades):
    a, b = divmod(i, cols)
    df_decade = dfnew[dfnew.transpose().columns.str.startswith(decade)]
    df_decade.div(10**9).plot.line(ax=axes[a,b], legend=False, rot=30)
    plt.legend(dfnew.columns)


# In[22]:


decades = ["196", "197", "198", "199", "200", "201"]
rows = 2
cols = 3
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,10), sharey=True)
fig.suptitle("Decade wise comparison in trillions of US$", fontsize=16)

for i, decade in enumerate(decades):
    a, b = divmod(i, cols)
    df_decade = dfnew[dfnew.transpose().columns.str.startswith(decade)]
    df_decade.plot.bar(ax=axes[a,b], legend=True, rot=30)


# In[23]:


# Percentage growth of the countries
PercIND = (dfnew['India'].iloc[-1] - dfnew['India'].iloc[0])*100/dfnew['India'].iloc[0]
PercCHN = (dfnew['China'].iloc[-1] - dfnew['China'].iloc[29])*100/dfnew['China'].iloc[29]
PercPAK = (dfnew['Pakistan'].iloc[-1] - dfnew['Pakistan'].iloc[0])*100/dfnew['Pakistan'].iloc[0]
PercBAN = (dfnew['Bangladesh'].iloc[-1] - dfnew['Bangladesh'].iloc[14])*100/dfnew['Bangladesh'].iloc[14]
PercSRI = (dfnew['Sri Lanka'].iloc[-1] - dfnew['Sri Lanka'].iloc[0])*100/dfnew['Sri Lanka'].iloc[0]
PercNPL = (dfnew['Nepal'].iloc[-1] - dfnew['Nepal'].iloc[11])*100/dfnew['Nepal'].iloc[11]
PercAFG = (dfnew['Afghanistan'].iloc[-1] - dfnew['Afghanistan'].iloc[44])*100/dfnew['Afghanistan'].iloc[44]


# In[24]:


data = [['China', PercCHN], ['India', PercIND], ['Pakistan', PercPAK], ['Bangladesh', PercBAN], ['Sri Lanka', PercSRI], ['Nepal', PercNPL],['Afghanistan', PercAFG]]


# In[25]:


percdf= pd.DataFrame(data, columns=['Country', 'Percentage growth'])
percdf.head(7)


# In[26]:


plt.figure(figsize=(15,8))
sns.barplot(x = 'Country', y = 'Percentage growth', data = percdf)
plt.xticks(rotation=45)


# In[27]:


model = percdf.join(new['Total'])
model


# # Reason out and uderstand the plot

# In[28]:


plt.figure(figsize=(15,8))
sns.barplot(x = 'Total', y = 'Percentage growth', hue='Country',data = model)
plt.xticks(rotation = 0)


# In[32]:


dfn = model.sort_values(by = ['Percentage growth'],ascending = False).reset_index()

fig = px.bar(model, x = 'Country', y = 'Percentage growth', color = 'Percentage growth', color_continuous_scale = 'reds')

fig.update_layout(title = 'Percentage growth of the mil. spending of countries', title_x = 0.5, title_font = dict(size = 16, color = 'Darkred'))


# In[30]:


dfn = model.sort_values(by = ['Total'],ascending = False).reset_index()

fig = px.bar(model, x = 'Country', y = 'Total', color = 'Total', color_continuous_scale = 'greens')

fig.update_layout(title = 'Total mil. spending of countries 1960 - 2018', title_x = 0.5, title_font = dict(size = 16, color = 'Darkgreen'))


# In[31]:


dfn = model.sort_values(by=['Total'],ascending = False).reset_index()

fig = px.choropleth(model, locations = 'Country', locationmode = 'country names', color = 'Total', scope = 'asia', hover_name = 'Country',color_continuous_scale = 'reds')

fig.update_layout(title= 'Total mil. spending of countries 1960 - 2018', title_x = 0.5,title_font = dict(size = 16, color = 'Darkred'),geo = dict(showframe = False,showcoastlines = False,projection_type = 'equirectangular'))
fig.show()


# In[ ]:





# In[ ]:




