#!/usr/bin/env python
# coding: utf-8

# In[36]:


# importing the modules required for data training after the data cleaning


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn import svm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier


# In[8]:


# importing the data set from online

data = pd.read_csv("C:/Users/smkp8/Downloads/ML model for machine images/dataset.csv")
data.head()


# In[9]:


data.info()


# In[10]:


data.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)


# In[11]:


print(data["Target"].unique())


# In[ ]:





# In[16]:


new_data = data.copy()
new_data = new_data.drop(columns=['Nationality', 
                                  'Mother\'s qualification', 
                                  'Father\'s qualification', 
                                  'Educational special needs', 
                                  'International', 
                                  'Curricular units 1st sem (without evaluations)',
                                  'Unemployment rate', 
                                  'Inflation rate'], axis=1)
new_data.info()


# In[17]:


new_data['Target'].value_counts()


# In[18]:


x = new_data['Target'].value_counts().index
y = new_data['Target'].value_counts().values

df = pd.DataFrame({
    'Target': x,
    'Count_T' : y
})

fig = px.pie(df,
             names ='Target', 
             values ='Count_T',
            title='How many dropouts, enrolled & graduates are there in Target column')

fig.update_traces(labels=['Graduate','Dropout','Enrolled'], hole=0.4,textinfo='value+label', pull=[0,0.2,0.1])
fig.show()


# In[19]:


correlations = data.corr()['Target']
top_10_features = correlations.abs().nlargest(10).index
top_10_corr_values = correlations[top_10_features]

plt.figure(figsize=(10, 11))
plt.bar(top_10_features, top_10_corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Top 10 Features with Highest Correlation to Target')
plt.xticks(rotation=45)
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship between Age and Target')
plt.show()


# In[21]:


X = new_data.drop('Target', axis=1)
y = new_data['Target']


# In[22]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[42]:


# svm training
clf= svm.SVC(kernel='linear',probability=True)


# In[38]:


clf.fit(X_train, y_train)


# In[40]:


y_pred = clf.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[ ]:




