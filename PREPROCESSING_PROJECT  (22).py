#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\srine\Downloads\LiverT_dataset.csv")


# In[3]:


sri=df.copy()
sri


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


#MEAN
print(df.mean())
print('\n')


# In[7]:


# MEDIAN 
print(df.median())
print('\n')


# In[8]:


print(df.mode())
print('\n')


# In[9]:


#VARIANCE
print(df.var())
print('\n')


# In[10]:


#STANDARD DEVIATION
print(df.std())
print('\n')


# In[11]:


#SKEWNESS 
df.skew()


# In[12]:


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


# In[18]:


import seaborn as sns


# In[19]:


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


# In[20]:


duplicate = df.duplicated()
duplicate

sum(duplicate)


# In[21]:


duplicate = df.duplicated()
duplicate

sum(duplicate)


# In[22]:


import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'


# In[23]:


import seaborn as sns
sns.boxplot(data = df, orient = 'h')


# In[24]:


df.columns


# In[25]:


df.isna().sum()


# In[26]:


df.dtypes


# In[27]:


df.isna().sum()


# In[28]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Specify the column with missing values
column_with_missing_values = 'R_BMI'

# Create a pipeline for handling missing values
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Apply the pipeline to impute missing values in the specified column
df[column_with_missing_values] = pipeline.fit_transform(df[[column_with_missing_values]])

# Print the modified DataFrame
print(df)


# In[29]:


# Specify the column with missing values
column_with_missing_values = 'R_Lympochyte'

# Create a pipeline for handling missing values
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Apply the pipeline to impute missing values in the specified column
df[column_with_missing_values] = pipeline.fit_transform(df[[column_with_missing_values]])

# Print the modified DataFrame
print(df)


# In[30]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['D_Hepatitis_B','R_Blood_Transfusion']] = imputer.fit_transform(df[['D_Hepatitis_B', 'R_Blood_Transfusion']])


# In[31]:


df.fillna(df.mode().iloc[0], inplace=True)


# In[32]:


df.isna().sum()


# In[33]:


categorical_columns = ['R_Gender', 'D_Gender', 'R_Etiology', 'R_Immunosuppressant_Medication']


# In[34]:


# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('one_hot_encoder', OneHotEncoder(), categorical_columns)
    ]
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Apply the pipeline to transform the data
transformed_data = pipeline.fit_transform(df)

# Get the new column names after one-hot encoding
new_column_names = preprocessor.named_transformers_['one_hot_encoder'].get_feature_names_out(categorical_columns)

# Create a new DataFrame with the transformed data and column names
transformed_df = pd.DataFrame(transformed_data, columns=new_column_names)

# Drop the original categorical columns from the original DataFrame
df_encoded = df.drop(categorical_columns, axis=1)

# Concatenate the encoded categorical columns with the remaining columns in the DataFrame
df = pd.concat([df_encoded, transformed_df], axis=1)


# In[35]:


# Specify the column to be one-hot encoded
categorical_column = 'D_Cause_of_Death'

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('one_hot_encoder', OneHotEncoder(), [categorical_column])
    ]
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Apply the pipeline to transform the data
transformed_data = pipeline.fit_transform(df)

# Get the new column names after one-hot encoding
new_column_names = preprocessor.named_transformers_['one_hot_encoder'].get_feature_names_out([categorical_column])

# Create a new DataFrame with the transformed data and column names
transformed_df = pd.DataFrame(transformed_data, columns=new_column_names)

# Drop the original categorical column from the original DataFrame
df_encoded = df.drop(categorical_column, axis=1)

# Concatenate the encoded categorical column with the remaining columns in the DataFrame
df = pd.concat([df_encoded, transformed_df], axis=1)


# In[36]:


print(df.columns)


# In[37]:


df= df.drop('Column1', axis = 1)
df


# In[38]:


X = df.drop('Complications', axis=1)
Y = df['Complications']


# In[39]:


X = pd.DataFrame(X)
X


# In[40]:


Y = pd.DataFrame(Y)
Y


# In[41]:


from sklearn.preprocessing import MinMaxScaler
pipeline = Pipeline([
    ('scaler', MinMaxScaler())  
])

scaled_data = pipeline.fit_transform(X)

X = pd.DataFrame(scaled_data, columns=X.columns)
X


# In[43]:


import pandas as pd
import os

df = pd.read_csv(r"C:\Users\srine\Downloads\LiverT_dataset.csv")

# Perform any necessary data cleaning or processing on the DataFrame

df.to_csv('cleanliver.csv', encoding='utf-8')

print("DataFrame saved as cleanliver.csv")

print("Current working directory:", os.getcwd())


# In[44]:


complications = Y['Complications']

class_counts = complications.value_counts()

plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Complications')
plt.ylabel('Count')
plt.title('Class Distribution of Complications')
plt.show()


# In[46]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, Y)

Complications_smote = y_smote['Complications']

class_counts_smote = Complications_smote.value_counts()

class_proportions = class_counts_smote / len(sri)
print("\nClass Proportions:")
print(class_proportions)

plt.bar(class_counts_smote.index, class_counts_smote.values)
plt.xlabel('Complications')
plt.ylabel('Count')
plt.title('Class Distribution after SMOTE')
plt.show()


# In[53]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[48]:


df = pd.read_csv(r"C:\Users\srine\Downloads\clean_liver.csv")


# In[49]:


# Separate the features (X) and target variable (y)
X = df.drop("Complications", axis=1)
y = df["Complications"]


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[54]:


# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[55]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[56]:


# Evaluate the model on the training set
y_train_pred = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[57]:


# Evaluate the model on the test set
y_test_pred = knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[58]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[125]:


from sklearn.metrics import precision_score, recall_score, f1_score

precision_knn = precision_score(y_test, y_test_pred, average='weighted')
recall_knn = recall_score(y_test, y_test_pred, average='weighted')
f1_knn = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_KNN:", precision_knn)
print("Recall_KNN:", recall_knn)
print("F1 Score_KNN:", f1_knn)


# In[59]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[60]:


nb = GaussianNB()


# In[61]:


nb.fit(X_train, y_train)


# In[64]:


# Evaluate the model on the training set
y_train_pred = nb.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[65]:


# Evaluate the model on the test set
y_test_pred = nb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[66]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[126]:


from sklearn.metrics import precision_score, recall_score, f1_score

precision_nb = precision_score(y_test, y_test_pred, average='weighted')
recall_nb = recall_score(y_test, y_test_pred, average='weighted')
f1_nb = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_NB:", precision_nb)
print("Recall_NB:", recall_nb)
print("F1 Score_NB:", f1_nb)


# In[67]:


from sklearn.linear_model import LogisticRegression


# In[68]:


# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[69]:


y_pred = model.predict(X_test)


# In[70]:


# Evaluate the model on the training set
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[71]:


# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[72]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[127]:


precision_model = precision_score(y_test, y_test_pred, average='weighted')
recall_model = recall_score(y_test, y_test_pred, average='weighted')
f1_model = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_MODEL:", precision_model)
print("Recall_MODEL:", recall_model)
print("F1 Score_MODEL:", f1_model)


# In[73]:


from sklearn.svm import SVC


# In[74]:


# Create an SVM classifier
svm = SVC()
# Train the SVM model
svm.fit(X_train, y_train)


# In[77]:


# Evaluate the model on the training set
y_train_pred = svm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[78]:


# Evaluate the model on the test set
y_test_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[79]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[128]:


precision_svm = precision_score(y_test, y_test_pred, average='weighted')
recall_svm = recall_score(y_test, y_test_pred, average='weighted')
f1_svm = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_SVM:", precision_svm)
print("Recall_SVM:", recall_svm)
print("F1 Score_SVM:", f1_svm)


# In[80]:


from sklearn.tree import DecisionTreeClassifier


# In[81]:


tree = DecisionTreeClassifier()


# In[82]:


tree.fit(X_train, y_train)


# In[83]:


y_pred = tree.predict(X_test)


# In[84]:


# Evaluate the model on the training set
y_train_pred = tree.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[85]:


# Evaluate the model on the test set
y_test_pred = tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[86]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[129]:


precision_tree = precision_score(y_test, y_test_pred, average='weighted')
recall_tree = recall_score(y_test, y_test_pred, average='weighted')
f1_tree = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_TREE:", precision_tree)
print("Recall_TREE:", recall_tree)
print("F1 Score_TREE:", f1_tree)


# In[87]:


from sklearn.ensemble import RandomForestClassifier


# In[88]:


# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rf.fit(X_train, y_train)


# In[89]:


y_pred = rf.predict(X_test)


# In[90]:


# Evaluate the model on the training set
y_train_pred = rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[91]:


# Evaluate the model on the test set
y_test_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[92]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[130]:


precision_rf = precision_score(y_test, y_test_pred, average='weighted')
recall_rf = recall_score(y_test, y_test_pred, average='weighted')
f1_rf = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_RF:", precision_rf)
print("Recall_RF:", recall_rf)
print("F1 Score_RF:", f1_rf)


# In[100]:


from sklearn.ensemble import GradientBoostingClassifier


# In[101]:


gbm = GradientBoostingClassifier()


# In[102]:


gbm.fit(X_train, y_train)


# In[103]:


y_pred = gbm.predict(X_test)


# In[104]:


# Evaluate the model on the training set
y_train_pred = gbm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[105]:


# Evaluate the model on the test set
y_test_pred = gbm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[106]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[131]:


precision_gbm = precision_score(y_test, y_test_pred, average='weighted')
recall_gbm = recall_score(y_test, y_test_pred, average='weighted')
f1_gbm = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_GBM:", precision_gbm)
print("Recall_GBM:", recall_gbm)
print("F1 Score_GBM:", f1_gbm)


# In[107]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


# In[108]:


mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)


# In[109]:


mlp.fit(X_train, y_train)


# In[110]:


y_pred = mlp.predict(X_test)


# In[111]:


# Evaluate the model on the training set
y_train_pred = gbm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)


# In[112]:


# Evaluate the model on the test set
y_test_pred = gbm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[113]:


print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[132]:


precision_mlp = precision_score(y_test, y_test_pred, average='weighted')
recall_mlp = recall_score(y_test, y_test_pred, average='weighted')
f1_mlp = f1_score(y_test, y_test_pred, average='weighted')
print("Precision_MLP:", precision_mlp)
print("Recall_MLP:", recall_mlp)
print("F1 Score_MLP:", f1_mlp)


# In[138]:


import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize lists or dictionaries to store the metrics for each model
models = ['model', 'tree', 'rf','mlp','gbm','svm','nb','knn']
train_precision = []
train_recall = []
train_f1 = []
test_precision = []
test_recall = []
test_f1 = []

# Calculate metrics for each model and append to the corresponding lists or dictionaries
for model in models:
    # Train set metrics
    train_precision.append(precision_score(y_train, y_train_pred, average='weighted'))
    train_recall.append(recall_score(y_train, y_train_pred, average='weighted'))
    train_f1.append(f1_score(y_train, y_train_pred, average='weighted'))
    
    # Test set metrics
    test_precision.append(precision_score(y_test, y_test_pred, average='weighted'))
    test_recall.append(recall_score(y_test, y_test_pred, average='weighted'))
    test_f1.append(f1_score(y_test, y_test_pred, average='weighted'))

# Create a DataFrame to store the metrics
data = {
    'Model': models,
    'Train Precision': train_precision,
    'Train Recall': train_recall,
    'Train F1 Score': train_f1,
    'Test Precision': test_precision,
    'Test Recall': test_recall,
    'Test F1 Score': test_f1
}
metrics_table = pd.DataFrame(data)

# Print the metrics table
metrics_table


# In[140]:


import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial'

# Calculate the accuracy values for train and test sets
train_accuracy = [0.73, 0.89, 0.99, 0.98, 1.0, 1.0, 1.0]
test_accuracy = [0.45, 0.87, 0.95, 0.97, 0.95, 0.97, 0.97]

# Create a list of model names
models = ['model', 'tree', 'rf', 'mlp', 'gbm', 'svm', 'nb']

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot train accuracy
ax1.plot(models, train_accuracy, marker='o')
ax1.set_ylabel('Train Accuracy')
ax1.set_title('Train Accuracy of Different Models')

# Plot test accuracy
ax2.plot(models, test_accuracy, marker='o')
ax2.set_xlabel('Models')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Test Accuracy of Different Models')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[143]:


import matplotlib.pyplot as plt

# Calculate the accuracy values for train and test sets
train_accuracy = [0.73, 0.89, 0.99, 0.98, 1.0, 1.0, 1.0]
test_accuracy = [0.45, 0.87, 0.95, 0.97, 0.95, 0.97, 0.97]

# Create a list of model names
models = ['model', 'tree', 'rf', 'mlp', 'gbm', 'svm', 'nb']

# Plot the test accuracies
plt.plot(models, test_accuracy, marker='o')

# Set the plot labels and title
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracies of Different Models')

# Find the best model
best_model_idx = test_accuracy.index(max(test_accuracy))
best_model = models[best_model_idx]
best_accuracy = test_accuracy[best_model_idx]

# Add a marker for the best model
plt.plot(best_model, best_accuracy, marker='o', color='red', label='Best Model')

# Add a legend
plt.legend()

# Show the plot
plt.show()

print("Best Model:", best_model)
print("Best Accuracy:", best_accuracy)


# In[144]:


from sklearn.neural_network import MLPClassifier
import joblib

# Create and train the best MLP model
best_model = MLPClassifier()  # Replace MLPClassifier() with your specific MLP model configuration
best_model.fit(X, y)  # Fit the model using the entire dataset

# Save the best model
joblib.dump(best_model, 'best_model.pkl')


# In[ ]:




