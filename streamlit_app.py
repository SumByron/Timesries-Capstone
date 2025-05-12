import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

st.set_page_config(page_title="Kenya Real GDP Analysis", layout="centered")
st.title("📈 Kenya Real GDP Time Series Analysis")

# Fetch data
@st.cache_data

def load_data():
    url = 'http://api.worldbank.org/v2/country/KE/indicator/NY.GDP.MKTP.KD?format=json&per_page=1000'
    response = requests.get(url)
    data = response.json()
    records = data[1]
    df = pd.DataFrame(records)[['date', 'value']].dropna()
    df.columns = ['Year', 'Real_GDP']
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index('Year', inplace=True)
    df.sort_index(inplace=True)
    df['Real_GDP'] = pd.to_numeric(df['Real_GDP'], errors='coerce')
    return df

st.sidebar.markdown("### Options")
show_data = st.sidebar.checkbox("Show raw data")

# Load and display data
df = load_data()
if show_data:
    st.subheader("Raw GDP Data")
    st.write(df.tail())

# Plot the data
st.subheader("Real GDP Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
df.plot(ax=ax, legend=False)
ax.set_title("Kenya Real GDP (constant 2015 US$)")
ax.set_ylabel("GDP")
ax.grid(True)
st.pyplot(fig)

# ADF Test
st.subheader("ADF Test for Stationarity")
result = adfuller(df['Real_GDP'])
st.write("ADF Statistic:", result[0])
st.write("p-value:", result[1])

if result[1] < 0.05:
    st.success("The time series is likely stationary (reject null hypothesis).")
else:
    st.warning("The time series is likely non-stationary (fail to reject null hypothesis).")

# Forecasting Section
st.subheader("Forecasting Kenya's GDP")
forecast_years = st.slider("Select number of years to forecast", 1, 20, 5)

# ARIMA Forecast
st.markdown("### ARIMA Forecast")
try:
    model_arima = ARIMA(df['Real_GDP'], order=(1, 1, 1))
    model_fit = model_arima.fit()
    forecast_arima = model_fit.forecast(steps=forecast_years)
    forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=forecast_years, freq='Y')
    forecast_series = pd.Series(forecast_arima, index=forecast_index)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df['Real_GDP'].plot(ax=ax2, label="Historical")
    forecast_series.plot(ax=ax2, label="Forecast", style='--')
    ax2.set_title("ARIMA Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
except Exception as e:
    st.error(f"ARIMA Forecast Error: {e}")

# Prophet Forecast
st.markdown("### Prophet Forecast")
try:
    df_prophet = df.reset_index().rename(columns={'Year': 'ds', 'Real_GDP': 'y'})
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=forecast_years, freq='Y')
    forecast = model_prophet.predict(future)
    fig3 = model_prophet.plot(forecast)
    st.pyplot(fig3)
except Exception as e:
    st.error(f"Prophet Forecast Error: {e}")
