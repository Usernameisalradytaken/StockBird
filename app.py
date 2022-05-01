import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import load_model
import streamlit as st
from datetime import date


st.image('stock.png',"",None,'')
st.title("Stock Bird")

start = '2020-05-27'
end = date.today()
#start = '2015-06-08'
#end = '2020-06-05'
# Reliance_Stock
st.title("Stocks Predication")
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input, 'yahoo', start ,end)
df = df.dropna()

st.subheader("DATA")
st.write(df.describe())

# closing price 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
m100 = df.Close.rolling(100).mean()
m200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(m100)
plt.plot(m200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)



scaler = MinMaxScaler(feature_range = (0,1))
df_close = scaler.fit_transform(np.array(df.Close).reshape(-1,1))

# training and testing 
training_size = int(len(df_close) * 0.60)
test_size = len(df_close) - training_size
train_data, test_data = df_close[0:training_size,:], df_close[training_size:len(df_close),:1]


def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)

# spliting data into x_train and y_train
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


# loding the model

model = load_model('keras_model.h5')
# previous 100 day data
time_stamp = 100
x_input = test_data[(len(test_data)-time_stamp):].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output=[]
# i have changes n_steps from 100 to x_input.shape
n_steps = time_stamp
nextNumberOfDays = 20
i=0

while(i<nextNumberOfDays):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

day_new = np.arange(1,time_stamp+1)
#30 days we will predict
day_pred = np.arange(time_stamp+1,(time_stamp+nextNumberOfDays+1))
print(lst_output)

st.subheader('30 days predicting')
df3 = df_close.tolist()
df3.extend(lst_output)
fig = plt.figure(figsize=(12,6))
plt.plot(df3)
st.pyplot(fig)

st.subheader('30 days predicting')
# this will plot read data
# this will plot predict data
fig2 = plt.figure(figsize=(12,6))
plt.plot(day_new, scaler.inverse_transform(df_close[(len(df_close)-time_stamp):]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
st.pyplot(fig2)
