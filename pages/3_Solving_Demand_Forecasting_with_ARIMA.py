import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta

st.set_page_config(
    page_title="Demand Forecasting: ARIMA"
)

st.title("Solving Demand Forecasting with ARIMA")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
Most business data, especially in supply chain, is time-series data with trends, seasonality, and cycles. Simple models may fail to capture these patterns, leading to less accurate forecasts.
""")

# -- Simulated Time Series Data --
st.subheader("Simulated Time Series Data")
st.write("""
For this project, we've simulated a time-series dataset that mimics real sales data with an upward trend and a clear seasonal pattern, making it a much better fit for an ARIMA model.
""")

# -- ARIMA Model Explanation --
st.write("""
The **ARIMA** (AutoRegressive Integrated Moving Average) model is a powerful statistical tool for time-series forecasting. It uses historical data to identify and model the patterns of a time series, enabling accurate predictions of future values based on past behavior.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data manipulation and creating a time series.
- **Numpy** for data simulation.
- **Statsmodels** for building the ARIMA model.
- **Matplotlib** for data visualization.
""")

# -- Code and Model Demonstration --
@st.cache_data
def generate_and_forecast_data():
    start_date = date(2022, 1, 1)
    end_date = date(2023, 12, 31)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    sales_data = 1000 + (dates.dayofyear + 365*(dates.year-2022)) * 1.5
    sales_data += (np.sin(dates.dayofyear / 90 * 2 * np.pi) * 300)
    sales_data += np.random.normal(0, 75, len(dates))
    
    df = pd.DataFrame({'Date': dates, 'Sales': sales_data})
    df.set_index('Date', inplace=True)

    model = ARIMA(df['Sales'], order=(5,1,0))
    model_fit = model.fit()
    
    return df, model_fit

if st.button("Generate New Data"):
    st.cache_data.clear()
    st.rerun()

df, model_fit = generate_and_forecast_data()

st.subheader("Simulated Historical Sales Data")
st.line_chart(df['Sales'].tail(90))

st.subheader("Make a Forecast")
periods_to_forecast = st.slider('Number of days to forecast', min_value=7, max_value=90, value=30)

forecast = model_fit.forecast(steps=periods_to_forecast)
forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(1, periods_to_forecast + 1)]
forecast_df = pd.DataFrame(forecast.values, index=forecast_dates, columns=['Forecasted Sales'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Sales'].tail(120), label='Historical Sales')
ax.plot(forecast_df['Forecasted Sales'], label='Forecast', color='red')
ax.set_title('Historical Sales vs. ARIMA Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
ax.grid(True)
st.pyplot(fig)
