import os
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# File path for the model
model_path = "C:\\Users\\pasun\\OneDrive\\Desktop\\py\\Stock Predictions Model.keras"

# Check if the file exists
if not os.path.isfile(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the file exists and the path is correct.")
else:
    # Load the pre-trained model
    model = load_model(model_path)

    # Streamlit app
    st.header("Stock Price Predictor")

    # Input for stock symbol
    stock_symbol = st.text_input("Enter stock Name", "GOOG")
    start_date = "2012-01-01"
    end_date = "2022-12-31"

    # Download stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Display stock data
    st.subheader("Stock Data")
    st.write(data)

    # Prepare training and testing datasets
    data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler on the training data and transform both training and test data
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # Prepare the full test data with the last 100 days of training data
    pas_100_days = data_train[-100:]
    data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_full_scaled = scaler.transform(data_test_full)

    # Plotting Moving Averages
    st.subheader("Price vs MA50")
    ma_50_days = data['Close'].rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, "r", label='MA50')
    plt.plot(data['Close'], "g", label='Close Price')
    plt.legend()
    st.pyplot(fig1)

    st.subheader("Price vs MA50 vs MA100")
    ma_100_days = data['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(ma_50_days, "r", label='MA50')
    plt.plot(ma_100_days, "b", label='MA100')
    plt.plot(data['Close'], "g", label='Close Price')
    plt.legend()
    st.pyplot(fig2)

    st.subheader("Price vs MA50 vs MA100 vs MA200")
    ma_200_days = data['Close'].rolling(200).mean()
    fig3 = plt.figure(figsize=(10, 8))
    plt.plot(ma_50_days, "r", label='MA50')
    plt.plot(ma_100_days, "b", label='MA100')
    plt.plot(ma_200_days, "y", label='MA200')
    plt.plot(data['Close'], "g", label='Close Price')
    plt.legend()
    st.pyplot(fig3)

    # Creating the dataset for the model
    x = []
    y = []
    for i in range(100, data_test_full_scaled.shape[0]):
        x.append(data_test_full_scaled[i-100:i])
        y.append(data_test_full_scaled[i, 0])
    x, y = np.array(x), np.array(y)

    # Predicting with the model
    predictions = model.predict(x)
    scale = 1 / scaler.scale_[0]
    predictions = predictions * scale
    y = y * scale

    # Display predictions
    st.subheader("Original Price vs Predicted Price")
    fig4 = plt.figure(figsize=(10, 8))
    plt.plot(y, "b", label="Original Price")
    plt.plot(predictions, "r", label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig4)

    # Debugging: Display the first few predictions and actual values
    st.subheader("Debugging Information")
    st.write("First 10 Actual Prices:", y[:10])
    st.write("First 10 Predicted Prices:", predictions[:10])
