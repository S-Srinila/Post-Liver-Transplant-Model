#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\srine\Downloads\LiverT_dataset.csv")

df.info()

df.describe()

#MEAN
print(df.mean())
print('\n')

# MEDIAN 
print(df.median())
print('\n')

print(df.mode())
print('\n')

#VARIANCE
print(df.var())
print('\n')

#STANDARD DEVIATION
print(df.std())
print('\n')

#SKEWNESS 
df.skew()

#KURTOSIS
df.kurt()

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# Plot histograms for all columns
df.hist(figsize=(10, 8))  # Adjust the figure size as needed

# Set titles and labels
plt.suptitle("Histograms of Dataset Columns")
plt.tight_layout()
plt.show()

# Plot box plots for all columns
df.boxplot(figsize=(10, 8))  # Adjust the figure size as needed

# Set titles and labels
plt.suptitle("Box Plots of Dataset Columns")
plt.tight_layout()
plt.show()

import seaborn as sns

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

duplicate = df.duplicated()
duplicate

sum(duplicate)

duplicate = data.duplicated()
duplicate

sum(duplicate)

import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

import seaborn as sns
sns.boxplot(data = data, orient = 'h')

df.columns

df.isna().sum()

df.dtypes

df.isna().sum()

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

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['D_Hepatitis_B','R_Blood_Transfusion']] = imputer.fit_transform(df[['D_Hepatitis_B', 'R_Blood_Transfusion']])

df.fillna(df.mode().iloc[0], inplace=True)

df.isna().sum()

categorical_columns = ['R_Gender', 'D_Gender', 'R_Etiology', 'R_Immunosuppressant_Medication']

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

print(df.columns)

df= df.drop('Column1', axis = 1)
df

X = df.drop('Complications', axis=1)
Y = df['Complications']

X = pd.DataFrame(X)
X

Y = pd.DataFrame(Y)
Y

from sklearn.preprocessing import MinMaxScaler
pipeline = Pipeline([
    ('scaler', MinMaxScaler())  
])

scaled_data = pipeline.fit_transform(X)

X = pd.DataFrame(scaled_data, columns=X.columns)
X

df.to_csv('clean_liver.csv',encoding='utf-8')
import os
os.getcwd()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\srine\Downloads\clean_liver.csv")

X = df.drop('Complications', axis=1)
y = df['Complications']
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
y.value_counts()
y.value_counts().plot.pie(autopct='%.2f')


from imblearn.over_sampling import SMOTE
# Encode categorical features using one-hot encoding
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Apply SMOTE to handle imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
# Count the number of samples in each class after SMOTE
class_counts_after_smote = pd.Series(y_resampled.value_counts(), name='Counts After SMOTE')

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, y)

Complications_smote = y_smote['Complications']

class_counts_smote = Complications_smote.value_counts()

class_proportions = class_counts_smote / len(df)
print("\nClass Proportions:")
print(class_proportions)

plt.bar(class_counts_smote.index, class_counts_smote.values)
plt.xlabel('Complications')
plt.ylabel('Count')
plt.title('Class Distribution after SMOTE')
plt.show()

