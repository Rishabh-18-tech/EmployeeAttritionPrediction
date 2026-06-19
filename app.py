import os
from datetime import datetime
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title='Employee Attrition Predictor',
    page_icon='👨‍💼',
    layout='wide',
)

@st.cache_resource
def load_model():
    with open('model/attrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.markdown('''<h1 style="text-align: center;">👨‍💼  Employee Attrition Prediction System </h1>''',unsafe_allow_html=True)
st.markdown('⇨ Predict whether an employee is likely to leave an organization.')

st.divider()

st.subheader('📋 Enter Employee Details')
st.subheader('👤 Employee Information')
employee_name = st.text_input('Employee Name', placeholder='Enter employee name here...')

if employee_name == '':
    st.warning('Enter Employee Name to proceed')

st.divider()


col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=60, value=30)
    overtime = st.selectbox('OverTime',
                options=[0, 1],
                format_func=lambda x: 'Yes' if x == 1 else 'No')
    marital_status = st.selectbox('Marital Status',
                options=[0, 1, 2],
                format_func=lambda x: {0:'Divorced', 1:'Married', 2:'Single'}[x])
    stock_option = st.selectbox('Stock Option Level', options=[0, 1, 2, 3],
                                format_func=lambda x: {0:'No Stock', 1:'Low', 2:'Medium', 3:'High'}[x])
    job_level = st.selectbox('Job Level', options=[1, 2, 3, 4, 5],
                             format_func=lambda x: {1:'fresher', 2:'Junior', 3:'Mid 5-10year', 4:'Senior', 5:'director/Executive'}[x])

with col2:
    job_satisfaction = st.selectbox('Job Satisfaction',
                options=[1, 2, 3, 4],
                format_func=lambda x: {1:'Low', 2:'Medium', 3:'High', 4:'Very High'}[x])
    monthly_income = st.number_input('Monthly Income',
                min_value=1009, max_value=16581, value=5000)
    years_at_company = st.number_input('Years At Company',
                min_value=0, max_value=32, value=5)
    years_with_manager = st.number_input('Years With Current Manager',
                min_value=0, max_value=int(years_at_company), value=0)
    total_working_years = st.number_input('Total Working Years',
                min_value=int(years_at_company), max_value=29, value=years_at_company)

st.divider()

if st.button('🔍 Predict Attrition', use_container_width=True):
    input_data = pd.DataFrame([{
        'OverTime':             overtime,
        'MaritalStatus':        marital_status,
        'StockOptionLevel':     stock_option,
        'Age':                  age,
        'JobLevel':             job_level,
        'YearsWithCurrManager': years_with_manager,
        'JobSatisfaction':      job_satisfaction,
        'YearsAtCompany':       years_at_company,
        'MonthlyIncome':        monthly_income,
        'TotalWorkingYears':    total_working_years
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error('‼️ ----  High Attrition Risk! ---- ‼️')
        st.metric('Attrition Probability', f'{round(probability*100, 2)}%')
        st.warning('''
                   →  Consider salary revision or bonous          
                    →  Offer promotion.      
                   →  Provide better work-life balance.
                   ''')
    else:
        st.success('❤️ Low Attrition Risk ❤️')
        st.metric('Attrition Probability', f'{round(probability*100, 2)}%')
        st.info('😊 Employee is likely to stay.')

st.divider()
st.caption('Built by Rishabh Shukla  using Streamlit | IBM HR Analytics Kaggle DataSet| MNNIT Internship 2026')
st.markdown('''[LinkedIn](https://www.linkedin.com/in/rishabh-shukla-6b6571327)   |  [GitHub](https://github.com/Rishabh-18-tech)''')
st.sidebar.markdown('Deployed on : [Streamlit Cloud](https://employee-attrition-rishabh.streamlit.app/)')
st.sidebar.info(' 📊 Project Details:')
st.sidebar.info('''
The model is trained on IBM HR Analytics Employee Attrition & Performance dataset from Kaggle.
                
The model is built using Random Forest Classifier and is deployed using Streamlit for interactive predictions.''')

st.sidebar.success('''Accuracy : 81.29%
                   
ROC-AUC  : 75.77%
                   
Model : Random Forest Classifier

Deployed on : Streamlitcloud

Developed by Rishabh Shukla
                   
''')
st.markdown('Deployed on : [Streamlit Cloud](https://employee-attrition-rishabh.streamlit.app/)')
st.markdown("[📩Email :10.rajasva@gmail.com ](mailto:10.rajasva@gmail.com)")