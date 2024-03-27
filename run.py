#  -*- coding: utf-8 -*-
# @Author  :   WEI
# @File    : run.py
# @Time    : 2024/3/25 21:46
# @Software: PyCharm
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib

st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Neuroblastoma Distant Metastasis Detection Assistant")
st.write("Hi, this is a simple program about neuroblastoma discrimination.")
st.write("You can enter the relevant variables and get a preliminary discriminatory result.")
st.write("Our results are not the truth, and intelligently assist you in the discrimination of the likelihood of "
         "cancer metastasis.")
st.write("The truth needs to be analyzed and summarized by professionals from the perspective of mechanism and so on.")
st.write('  ')
st.write('  ')
st.write('  ')

age = st.sidebar.selectbox('Age', ['≤1', '＞1'])
his = st.sidebar.selectbox('Histology', ['Neuroblastoma', 'Ganglioneuroblastoma'])
size = st.sidebar.selectbox('Tumor Size', ['≤3', '3-6', '6-10', '≥10', 'Unknown'])
grade = st.sidebar.selectbox('Tumor Grade', ['Grade I/II', 'Grade III', 'Grade IV', 'Unknown'])
site = st.sidebar.selectbox('Primary Site', ['Adrenal', 'Retroperitoneum', 'Other'])
sur = st.sidebar.selectbox('Surgery', ['Yes', 'No', 'Unknown'])
che = st.sidebar.selectbox('Chemotherapy', ['Yes', 'No/Unknown'])
rad = st.sidebar.selectbox('Radiotherapy', ['Yes', 'None/Unknown'])

data = [age, his, size, grade, site, sur, che, rad]
cols = ['Age', 'Histology', 'Tumor Size', 'Tumor grade', 'Primary Site', 'Surgery', 'Chemotherapy', 'Radiotherapy']
data_input = []
for i in range(len(cols)):
    col = cols[i]
    with open('LE/label_encoder_{}.pkl'.format(col), 'rb') as f:
        le = pickle.load(f)
        data_transform = le.transform(np.array(data[i]).reshape(-1, ))
        data_input.append(data_transform)

data_array = np.array(data_input).reshape(1, -1)
data_dataframe = pd.DataFrame(data_array, columns=cols)
model = joblib.load('models/LR.pkl')

if st.sidebar.button("Predict"):
    pre = model.predict_proba(data_dataframe)[:, 1][0]

    if pre > 0.5:
        st.markdown("""  
        <span style="color: black; font-size: 50px; background-color: #FFFF00;">  
        Probability of distant metastasis: {:.0f}% 
        </span>  
        """.format(pre * 100), unsafe_allow_html=True)
    else:
        st.markdown("""  
        <span style="color: black; font-size: 50px; background-color: #63B8FF;">  
        Probability of distant metastasis: {:.0f}% 
        </span>  
        """.format(pre * 100), unsafe_allow_html=True)
