import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go  # <-- This is the new, corrected line
from datetime import date, timedelta

st.set_page_config(
    page_title="Demand Forecasting: Prophet"
)

st.title("Solving Demand Forecasting with Prophet")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
Forecasting real-world data is often complicated by multiple factors like weekly sales cycles, yearly seasonality, and specific holidays. Traditional models can be difficult to tune for these complexities. Prophet is designed to handle them out-of-the-box.
""")

# -- Simulated Time Series Data --
st.subheader("Simulated Time Series Data")
st.write("""
This project uses simulated data with a clear yearly trend and a weekly seasonal pattern, making it an ideal candidate for Prophet's forecasting capabilities.
""")

# -- Prophet Model Explanation --
st.subheader("Prophet Model")
st.write("""
Prophet is an open-source forecasting library developed by Meta. It works by decomposing a time series into three main components: **trend**, **seasonality**, and **holidays**. It's known for being easy to use, robust to missing data, and effective at capturing multiple seasonalities.
""")

# -- Industry Applicability Section --
st.subheader("Industry Applicability")
st.write("""
Prophet is a go-to tool for a wide range of applications:
- **E-commerce & Retail:** Forecasting daily or weekly product sales, especially when sales are heavily influenced by holidays like Black Friday or Christmas.
- **Web Traffic Analysis:** Predicting website visits, which often have strong weekly and yearly patterns.
- **Operations:** Forecasting resource needs or customer service call volumes, which can vary significantly by day of the week and time of year.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data manipulation.
- **Numpy** for data simulation.
- **Prophet** for building the forecasting model.
- **Plotly** and **Matplotlib** for interactive data visualization.
""")

# -- Code and Model Demonstration --
@st.cache_data
def generate_and_forecast_data(forecast_periods):
    # Simulate time series data with trend and seasonality
    dates = pd.date_range(start='2020-01-01', periods=1095, freq='D')
    sales_data = 500 + np.arange(len(dates)) * 0.75  # Upward trend
    sales_data += np.sin(dates.dayofyear / 365 * 2 * np.pi) * 100  # Yearly seasonality
    sales_data += np.sin(dates.dayofweek / 7 * 2 * np.pi) * 50  # Weekly seasonality
    sales_data += np.random.normal(0, 20, len(dates)) # Random noise
    
    df = pd.DataFrame({'ds': dates, 'y': sales_data})
    
    # Train Prophet model
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    
    # Create future dataframe and forecast
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    
    return df, model, forecast

if st.button("Generate New Data"):
    st.cache_data.clear()
    st.rerun()

periods_to_forecast = st.slider('Number of days to forecast', min_value=7, max_value=180, value=90)
df, model, forecast = generate_and_forecast_data(periods_to_forecast)

st.subheader("Historical Sales Data & Forecast")
st.write("This plot shows the historical data (blue dots) and the Prophet model's forecast (dark blue line) with its uncertainty interval (light blue shade).")

fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Prophet Model Components")
st.write("Prophet automatically decomposes the forecast into its individual components: trend, yearly seasonality, and weekly seasonality.")
fig_components = plot_components_plotly(model, forecast)
st.plotly_chart(fig_components, use_container_width=True)

st.subheader("Raw Forecast Data")
st.write("The table below shows the full historical and forecasted data, including upper and lower bounds for the prediction.")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_forecast + 5))
