import streamlit as st
import requests
import datetime
import json

st.title('PySpark Real Time Scoring API')

# PySpark API endpoint
url = 'http://pysparkapi:5000'
endpoint = '/api/'

# description and instructions
st.write('''A real time scoring API for PySpark model.''')

st.header('User Input features')

def user_input_features():
    input_features = {}
    input_features["age"] = st.slider('Age', 18, 95)
    input_features["job"] = st.selectbox('Job', ['management', 'technician', 'entrepreneur', 'blue-collar', \
           'unknown', 'retired', 'admin.', 'services', 'self-employed', \
           'unemployed', 'housemaid', 'student'])
    input_features["marital"] = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
    input_features["education"] = st.selectbox('Education Qualification', ['tertiary', 'secondary', 'unknown', 'primary'])
    input_features["default"] = st.selectbox('Have you defaulted before?', ['yes', 'no'])
    input_features["balance"]= st.slider('Current balance', -10000, 150000)
    input_features["housing"] = st.selectbox('Do you own a home?', ['yes', 'no'])
    input_features["loan"] = st.selectbox('Do you have a loan?', ['yes', 'no'])
    input_features["contact"] = st.selectbox('Best way to contact you', ['cellular', 'telephone', 'unknown'])
    date = st.date_input("Today's Date")
    input_features["day"] = date.day
    input_features["month"] = date.strftime("%b").lower()
    input_features["duration"] = st.slider('Duration', 0, 5000)
    input_features["campaign"] = st.slider('Campaign', 1, 63)
    input_features["pdays"] = st.slider('pdays', -1, 871)
    input_features["previous"] = st.slider('previous', 0, 275)
    input_features["poutcome"] = st.selectbox('poutcome', ['success', 'failure', 'other', 'unknown'])
    return [input_features]

json_data = user_input_features()

submit = st.button('Get predictions')
if submit:
    results = requests.post(url+endpoint, json=json_data)
    results = json.loads(results.text)
    st.header('Final Result')
    prediction = results["prediction"]
    probability = results["probability"]
    st.write("Prediction: ", int(prediction))
    st.write("Probability: ", round(probability,3))
