import tensorflow as tf
import numpy as np
import pickle
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoders and scaler 

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
    
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# streamlit app 

st.title("Customer Churn Prediction")

#User_imput

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=700)

#prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary],
})


# One-hot encode the geography column

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

# prediction churn 
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write("Prediction Probability: ", prediction_proba)
if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")