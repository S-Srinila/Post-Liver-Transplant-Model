#!/usr/bin/env python
# coding: utf-8

# In[17]:


pip install streamlit


# In[1]:


import streamlit as st
import joblib
import pickle


# In[2]:


model = pickle.load(open(r"C:\Users\srine\Downloads\best_model.pkl", 'rb'))


# In[3]:


def main():
    st.title('Machine Learning Model Deployment')


# In[4]:


# Define the feature options
gender_options = ['Male', 'Female']
age_options = [30, 40, 50, 60, 70]
bmi_options = [20, 25, 30, 35, 40]
diabetes_options = ['Yes', 'No']
hypertension_options = ['Yes', 'No']
alcohol_abuse_options = ['Yes', 'No']
smoking_options = ['Yes', 'No']
lymphocyte_options = [3000, 4000, 5000, 6000, 7000]
hepatitis_b_options = ['Yes', 'No']
hepatitis_c_options = ['Yes', 'No']


# In[5]:


# UI elements for user input
st.title('Complication Prediction')
st.subheader('Enter Patient Information')


# In[6]:


gender = st.selectbox('Gender', gender_options)
age = st.selectbox('Age', age_options)
bmi = st.selectbox('BMI', bmi_options)
diabetes = st.selectbox('Diabetes', diabetes_options)
hypertension = st.selectbox('Hypertension', hypertension_options)
alcohol_abuse = st.selectbox('Alcohol Abuse', alcohol_abuse_options)
smoking = st.selectbox('Smoking', smoking_options)
lymphocyte = st.selectbox('Lymphocyte Count', lymphocyte_options)
hepatitis_b = st.selectbox('Hepatitis B', hepatitis_b_options)
hepatitis_c = st.selectbox('Hepatitis C', hepatitis_c_options)

submit_button = st.button('Predict')


# In[7]:


if submit_button:
    # Prepare the input data
    input_data = [[gender, age, bmi, diabetes, hypertension, alcohol_abuse, smoking, lymphocyte, hepatitis_b, hepatitis_c]]
    
    # Make the prediction
    prediction = model.predict(input_data)[0]


# In[8]:


# Map the prediction to the corresponding complication
complication_mapping = {
    0: 'Artery Thrombosis',
    1: 'Biliary Complications',
    2: 'Cardiovascular Complications',
    3: 'Infection',
    4: 'Metabolic Complications',
    5: 'No Complication',
    6: 'Portal Vein Thrombosis',
    7: 'Post-transplant Diabetes',
    8: 'Primary Graft Non-function',
    9: 'Rejection',
    10: 'Renal Dysfunction'
}


# In[10]:


# Add UI elements for user input
st.title('Complication Prediction')
st.subheader('Enter Patient Information')

gender = st.selectbox('Gender', gender_options)
age = st.selectbox('Age', age_options)
bmi = st.selectbox('BMI', bmi_options)
diabetes = st.selectbox('Diabetes', diabetes_options)
hypertension = st.selectbox('Hypertension', hypertension_options)
alcohol_abuse = st.selectbox('Alcohol Abuse', alcohol_abuse_options)
smoking = st.selectbox('Smoking', smoking_options)
lymphocyte = st.selectbox('Lymphocyte Count', lymphocyte_options)
hepatitis_b = st.selectbox('Hepatitis B', hepatitis_b_options)
hepatitis_c = st.selectbox('Hepatitis C', hepatitis_c_options)

submit_button = st.button('Predict')

if submit_button:
    # Prepare the input data
    input_data = [[gender, age, bmi, diabetes, hypertension, alcohol_abuse, smoking, lymphocyte, hepatitis_b, hepatitis_c]]

    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Map the prediction to the corresponding complication
    complication_mapping = {
        0: 'Artery Thrombosis',
        1: 'Biliary Complications',
        2: 'Cardiovascular Complications',
        3: 'Infection',
        4: 'Metabolic Complications',
        5: 'No Complication',
        6: 'Portal Vein Thrombosis',
        7: 'Post-transplant Diabetes',
        8: 'Primary Graft Non-function',
        9: 'Rejection',
        10: 'Renal Dysfunction'
    }

    # Display the predicted complication
    st.subheader('Prediction Result')
    st.write('Predicted Complication:', complication_mapping[prediction])


# In[11]:


if submit_button:
    # Prepare the input data
    input_data = [[gender, age, bmi, diabetes, hypertension, alcohol_abuse, smoking, lymphocyte, hepatitis_b, hepatitis_c]]

    # Make the prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    # Map the prediction to the corresponding complication
    complication_mapping = {
        0: 'Artery Thrombosis',
        1: 'Biliary Complications',
        2: 'Cardiovascular Complications',
        3: 'Infection',
        4: 'Metabolic Complications',
        5: 'No Complication',
        6: 'Portal Vein Thrombosis',
        7: 'Post-transplant Diabetes',
        8: 'Primary Graft Non-function',
        9: 'Rejection',
        10: 'Renal Dysfunction'
    }

    # Display the predicted complication and probability
    st.subheader('Prediction Result')
    st.write('Predicted Complication:', complication_mapping[prediction])
    st.write('Probability:', probability)


# In[12]:


if __name__ == '__main__':
    main()


# In[21]:


streamlit run app.py


# In[24]:


streamlit run C:\Users\srine\anaconda3\lib\site-packages\ipykernel_launcher.py


# In[ ]:




