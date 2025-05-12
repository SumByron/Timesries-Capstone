import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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
