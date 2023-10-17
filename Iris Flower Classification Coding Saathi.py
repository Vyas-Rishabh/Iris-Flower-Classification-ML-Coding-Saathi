#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification

# ![iris-machinelearning.png](attachment:iris-machinelearning.png)

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[4]:


iris_data = pd.read_csv("iris.csv", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
iris_data.head()


# In[5]:


iris_data.tail()


# ### Statistical Data Analysis

# In[6]:


iris_data.describe()


# In[7]:


#Length of Data
iris_data.shape


# ### summary of a DataFrame

# In[8]:


iris_data.info()


# In[9]:


#Checking null value
iris_data.isnull().sum()


# In[10]:


iris_data['species'].value_counts()


# In[11]:


iris_data['sepal_width'].hist()


# In[12]:


iris_data['sepal_length'].hist()


# In[13]:


iris_data['petal_width'].hist()


# In[14]:


iris_data['petal_length'].hist()


# In[15]:


s = sns.FacetGrid(iris_data, height=8, hue="species")
s.map(plt.scatter, "petal_length", "petal_width")
s.add_legend()
sns.set_style("whitegrid")
plt.show()


# In[16]:


s = sns.FacetGrid(iris_data, height=8, hue="species")
s.map(plt.scatter, "sepal_length", "sepal_width")
s.add_legend()
sns.set_style("whitegrid")
plt.show()


# In[17]:


sns.pairplot(iris_data, height=2.5, hue="species")
plt.show()


# In[18]:


#Checking Correlation use of Heatmap
sns.heatmap(iris_data.corr(), annot=True)
plt.show()


# ### Split the data into training and testing

# In[19]:


from sklearn.model_selection import train_test_split

X = iris_data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris_data["species"]


# In[20]:


X


# In[21]:


y


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)


# ### Logistic regression model

# In[23]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[24]:


model.fit(X_train,y_train)


# In[25]:


#metrics to get performance
print('Accuracy',model.score(X_test,y_test)*100)


# ### K-Nearest Neighbours model

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[27]:


model.fit(X_train,y_train)


# In[28]:


#metrics to get performance
print('Accuracy',model.score(X_test,y_test)*100)


# ### Decision tree model

# In[29]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[30]:


model.fit(X_train,y_train)


# In[31]:


#metrics to get performance
print('Accuracy',model.score(X_test,y_test)*100)


# You can find this project on <a href="https://github.com/Vyas-Rishabh/Iris-Flower-Classification-ML-Coding-Saathi"><b>GitHub.</b></a>
