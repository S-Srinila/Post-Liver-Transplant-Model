#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Load the dataset
df = pd.read_csv(r"C:\Users\srine\Downloads\LiverT_dataset.csv")
# Count the occurrences of each class
class_counts = df['Complications'].value_counts()
# Print the class counts
print("Class Counts:")
print(class_counts)
# Visualize class distribution
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Compute class imbalance ratio
imbalance_ratio = class_counts.max() / class_counts.min()

print("Imbalance Ratio:", imbalance_ratio)


# In[2]:


import pandas as pd
df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[3]:


X = df.drop('Complications', axis=1)
y = df['Complications']


# In[6]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'


# In[7]:


y.value_counts()
y.value_counts().plot.pie(autopct='%.2f')


# In[8]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

# Assuming X and y contain your data
# Encode categorical features using one-hot encoding
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Apply SMOTE to handle imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)


# In[9]:


# Count the number of samples in each class after SMOTE
class_counts_after_smote = pd.Series(y_resampled.value_counts(), name='Counts After SMOTE')


# In[10]:


# Plot the target variable distribution after SMOTE
plt.title('Target Variable Distribution After SMOTE')
class_counts_after_smote.plot.pie(autopct='%.2f%%')
plt.ylabel('')
plt.show()


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[12]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[13]:


# Separate the features (X) and target variable (y)
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[14]:


X_encoded = pd.get_dummies(X)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[16]:


tree = DecisionTreeClassifier()


# In[17]:


tree.fit(X_train, y_train)


# In[18]:


y_pred = tree.predict(X_test)


# In[19]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[21]:


# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[22]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[23]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[79]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[83]:


# Assuming "Age" as the independent variable and "Albumin" as the dependent variable
X = df['R_Age'].values.reshape(-1, 1)
y = df['R_Albumin_level'].values


# In[84]:


# Creating a Linear Regression model object
reg = LinearRegression()


# In[85]:


# Fitting the model to the data
reg.fit(X, y)


# In[86]:


# Making predictions on the test set
y_pred = reg.predict(X)


# In[87]:


import numpy as np

# Calculating the mean squared error and correlation coefficient
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)


# In[88]:


print('Root Mean Squared Error:', rmse)
print('Correlation Coefficient:', r2)


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[33]:


nb = GaussianNB()


# In[34]:


nb.fit(X_train, y_train)


# In[35]:


y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[37]:


# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[38]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[40]:


y_pred = knn.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[41]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[42]:


# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rf.fit(X_train, y_train)


# In[43]:


# Make predictions on the test set
y_pred = rf.predict(X_test)


# In[44]:


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[45]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[46]:


# Create an SVM classifier
svm = SVC()
# Train the SVM model
svm.fit(X_train, y_train)


# In[47]:


# Make predictions on the test set
y_pred = svm.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[69]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# In[70]:


df = pd.read_csv(r"C:\Users\srine\Downloads\cleanliver.csv")


# In[71]:


# Separate the features and target variable
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[ ]:


import pandas as pd

# Assuming your labels are in a column named 'label' in your dataframe
data['label'] = data['label'].astype(float)


# In[73]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=46))  # Update input_dim with the correct value
model.add(Dense(32, activation='relu'))  # Add hidden layer
model.add(Dense(1, activation='sigmoid'))  # Add output layer


# In[74]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[78]:


batch_size = 32
epochs = 10

model.fit(df.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test, y_test))


# In[ ]:




