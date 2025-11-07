#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv(r"C:\Users\srine\Downloads\LiverT_dataset.csv")


# In[12]:


df.info()


# In[11]:


df.describe()


# In[4]:


#MEAN
print(df.mean())
print('\n')


# In[5]:


# MEDIAN 
print(df.median())
print('\n')


# In[6]:


print(df.mode())
print('\n')


# In[7]:


#VARIANCE
print(df.var())
print('\n')


# In[8]:


#STANDARD DEVIATION
print(df.std())
print('\n')


# In[9]:


#SKEWNESS 
df.skew()


# In[10]:


#KURTOSIS
df.kurt()


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'


# In[15]:


# Plot histograms for all columns
df.hist(figsize=(10, 8))  # Adjust the figure size as needed


# In[16]:


# Set titles and labels
plt.suptitle("Histograms of Dataset Columns")
plt.tight_layout()
plt.show()


# In[17]:


# Plot box plots for all columns
df.boxplot(figsize=(10, 8))  # Adjust the figure size as needed

# Set titles and labels
plt.suptitle("Box Plots of Dataset Columns")
plt.tight_layout()
plt.show()


# In[20]:


import seaborn as sns


# In[22]:


# loop through each pair of numerical columns and plot a scatterplot
for i, col1 in enumerate(df.columns):
    if df[col1].dtype == 'float64' or df[col1].dtype == 'int64':
        for j, col2 in enumerate(df.columns[i+1:]):
            if df[col2].dtype == 'float64' or df[col2].dtype == 'int64':
                # select the columns you want to plot and create numpy arrays
                column_data1 = df[col1].values
                column_data2 = df[col2].values

                # plot the scatterplot
                plt.scatter(column_data1, column_data2)

                # set labels and title
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.title('Scatterplot of ' + col1 + ' vs ' + col2)

                # show the plot
                plt.show()


# In[6]:


duplicate = df.duplicated()
duplicate

sum(duplicate)


# In[7]:


data = df.drop_duplicates(keep = 'last')
data


# In[8]:


duplicate = data.duplicated()
duplicate

sum(duplicate)


# In[11]:


import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'


# In[12]:


import seaborn as sns
sns.boxplot(data = data, orient = 'h')


# In[14]:


df.columns


# In[23]:


df.isna().sum()


# In[25]:


df = df.dropna()


# In[26]:


df.isna().sum()


# In[27]:


df.shape


# In[28]:


df.to_csv('cleanliver.csv',encoding='utf-8')
import os
os.getcwd()


# DESICION TREE

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[10]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[11]:


# Separate the features (X) and target variable (y)
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[12]:


X_encoded = pd.get_dummies(X)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[14]:


tree = DecisionTreeClassifier()


# In[15]:


tree.fit(X_train, y_train)


# In[16]:


y_pred = tree.predict(X_test)


# In[17]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# NAIVE BAYES

# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[19]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[20]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[22]:


nb = GaussianNB()


# In[23]:


nb.fit(X_train, y_train)


# In[24]:


y_pred = nb.predict(X_test)


# In[25]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# KNN

# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[28]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[29]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[31]:


# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[32]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[33]:


knn.fit(X_train, y_train)


# In[34]:


y_pred = knn.predict(X_test)


# In[35]:


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# LOGISTIC REGRESSION

# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[37]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[38]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[40]:


# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[41]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[42]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# RANDOM FORESTS

# In[43]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[44]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[45]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[47]:


# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[48]:


# Train the model
rf.fit(X_train, y_train)


# In[49]:


# Make predictions on the test set
y_pred = rf.predict(X_test)


# In[50]:


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# SVM

# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[52]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[53]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[55]:


# Create an SVM classifier
svm = SVC()


# In[56]:


# Train the SVM model
svm.fit(X_train, y_train)


# In[57]:


# Make predictions on the test set
y_pred = svm.predict(X_test)


# In[58]:


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[83]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# In[84]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[85]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[87]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[88]:


model = keras.Sequential([
    keras.layers.Dense(6, activation=tf.nn.relu, input_shape=(X_train.shape[1],)),
    keras.layers.Dense(6, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu)
])


# In[89]:


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_squared_error'])


# In[90]:


model.fit(X_train, y_train, epochs=100)


# In[ ]:




