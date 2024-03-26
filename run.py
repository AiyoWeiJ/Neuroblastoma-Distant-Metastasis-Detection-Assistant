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

st.title("Cancer Metastasis Detection Assistant")
st.write("Hi, this is a simple program about cancer metastasis discrimination.")
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
        data_transform = le.transform(np.array(data[i]).reshape(-1,))
        data_input.append(data_transform)

data_array = np.array(data_input).reshape(1, -1)
data_dataframe = pd.DataFrame(data_array, columns=cols)
model = joblib.load('models/LR.pkl')


styled_text_1 = """  
<style>  
.large-text {  
    font-size: 24px; /* 设置字体大小 */  
    color: white; /* 设置字体颜色，与 Streamlit 的成功消息背景相协调 */  
    background-color: #FFA500; /* 设置背景颜色，与 Streamlit 的成功消息背景相同 */  
    padding: 10px; /* 添加内边距，以模仿 Streamlit 的成功消息样式 */  
    border-radius: 4px; /* 添加圆角 */  
}  
</style>  
<div class="large-text">  
According to our calculation, there is a large transfer risk! 
</div>  
"""
styled_text_0 = """  
<style>  
.large-text {  
    font-size: 24px; /* 设置字体大小 */  
    color: white; /* 设置字体颜色，与 Streamlit 的成功消息背景相协调 */  
    background-color: #28a745; /* 设置背景颜色，与 Streamlit 的成功消息背景相同 */  
    padding: 10px; /* 添加内边距，以模仿 Streamlit 的成功消息样式 */  
    border-radius: 4px; /* 添加圆角 */  
}  
</style>  
<div class="large-text">  
According to our calculations, there is a small transfer risk! 
</div>  
"""


if st.sidebar.button("Predict"):
    pre = model.predict(data_dataframe)
    if pre:
        st.markdown(styled_text_1, unsafe_allow_html=True)
    else:
        st.markdown(styled_text_0, unsafe_allow_html=True)

