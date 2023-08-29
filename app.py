import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st


start = '2010-06-22'
end = datetime.now().strftime('%Y-%m-%d')

st.title("Stock Trend Prediction")
user_input = st.text_input('Énter Stock Tricker','AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data From 2010 -2023')
st.write(df.describe())

st.subheader('Çlosing Price VS Time Chart ')
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig2)

st.subheader('Çlosing Price VS Time Chart with 100MA &  200MA')
ma100 = df.Close.rolling(100).mean() 
ma200 = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12,6))
plt.plot(100)
plt.plot(200)
plt.plot(df.Close)
st.pyplot(fig3)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scaler_factor = 1/scaler[0]
y_predicted = y_predicted*scaler_factor
y_test = y_test*scaler_factor

st.subheader('Prediction vs Original')
fig1 =plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig1)
