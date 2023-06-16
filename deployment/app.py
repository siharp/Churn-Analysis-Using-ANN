import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# import preproses
preproses = pickle.load(open("preproses.pkl", "rb"))

# import model
model = load_model('model.h5')

#title
st.title("Customer Churn Predictions")
st.write("Created by Sihar Pangaribuan")

# User imput
user_id = st.text_input('Input ID of a customer', value='')
age = st.number_input(label='Age of a customer', min_value=10, max_value=64, value=10, step=1)
gender = st.selectbox(label='Gender of a customer', options=['F','M'])
region_category = st.selectbox(label='Select Region that a customer belongs to', options=['City', 'Village', 'Town'])
membership_category = st.selectbox(label='Select Category of the membership that a customer is using', options=['No Membership', 'Basic Membership', 'Silver Membership', 'Premium Membership', 'Gold Membership', 'Platinum Membership'])
joining_date = st.text_input('Date when a customer became a member', value='')
joined_through_referral = st.selectbox(label='Whether a customer joined using any referral code or ID ?', options=['Yes','No'])
preferred_offer_types = st.selectbox(label='Select Type of offer that a customer prefers', options=['Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'])
medium_of_operation = st.selectbox(label='Select Medium of operation that a customer uses for transactions', options=['Desktop', 'Smartphone', 'Both'])
internet_option = st.selectbox(label='Select Type of internet service a customer uses', options=['Wi-Fi', 'Fiber_Optic', 'Mobile_Data'])
last_visit_time = st.text_input('Input The last time a customer visited the website', value='')
days_since_last_login = st.number_input(label='Imput Number of days since a customer last logged into the website', min_value=-999, max_value=26, value=-999, step=1)
avg_time_spent = st.number_input(label='Imput Average time spent by a customer on the website', min_value=0.0, max_value=3235.6, value=0.0, step=0.1)
avg_transaction_value = st.number_input(label='Imput Average transaction value of a customer', min_value=800.46, max_value=99914.05, value=800.46, step=0.1)
avg_frequency_login_days = st.number_input(label='Imput Number of times a customer has logged in to the website', min_value=0.0, max_value=73.07, value=0.0, step=0.1)
points_in_wallet = st.number_input(label='Imput Points awarded to a customer on each transaction', min_value=0.0, max_value=2069.06, value=0.0, step=0.1)
used_special_discount = st.selectbox(label='Whether a customer uses special discounts offered?', options=['Yes','No'])
offer_application_preference = st.selectbox(label='Whether a customer prefers offers?', options=['Yes','No'])
past_complaint = st.selectbox(label='Whether a customer has raised any complaints?', options=['Yes','No'])
complaint_status = st.selectbox(label='Select the complaint status', options=['No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'])
feedback = st.selectbox(label='Select the feedback', options=['Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality', 'No reason specified', 'Products always in Stock', 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'])

# Convert ke data frame
data = pd.DataFrame({
    'user_id':[user_id],
    'age':[age],
    'gender':[gender],
    'region_category':[region_category],
    'membership_category':[membership_category],
    'joining_date':[joining_date],
    'joined_through_referral':[joined_through_referral],
    'preferred_offer_types':[preferred_offer_types],
    'medium_of_operation':[medium_of_operation],
    'internet_option':[internet_option],
    'last_visit_time':[last_visit_time],
    'days_since_last_login':[days_since_last_login],
    'avg_time_spent':[avg_time_spent],
    'avg_transaction_value':[avg_transaction_value],
    'avg_frequency_login_days':[avg_frequency_login_days],
    'points_in_wallet':[points_in_wallet],
    'used_special_discount':[used_special_discount],
    'offer_application_preference':[offer_application_preference],
    'past_complaint':[past_complaint],
    'complaint_status':[complaint_status],
    'feedback':[feedback]
            })

# Transfom data
data = preproses.transform(data)

# model predict
if st.button('Predict'):
    prediction = model.predict(data).tolist()[0]

    if prediction == 1:
        prediction = 'Froud'
    else:
        prediction = 'Not Froud'

    st.write('The Prediction is: ')
    st.write(prediction)
