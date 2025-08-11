import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Demand Forecasting"
)

st.title("Solving Demand Forecasting with Linear Regression")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
In a supply chain, accurate demand forecasting is the bedrock of operational efficiency. Without a reliable forecast, companies risk making costly decisions that lead to either:
- **Excess Inventory:** High carrying costs and potential obsolescence.
- **Stockouts:** Lost sales and a damaged customer experience.
""")

# -- The Concept: Simulated Data --
st.subheader("Simulated Data for Our Playground")
st.write("""
I used a simulated dataset to demonstrate the core concept.
""")

# -- Concept Explanation Section --
st.subheader("Supervised Learning & Linear Regression")
st.write("""
This project uses **Supervised Learning**, a type of machine learning where we train a model on a labeled dataset. We give the model both the input (Units Sold) and the correct output (Total Revenue). The model's job is to learn the relationship between these two variables.

We use **Linear Regression** to find the "line of best fit" that represents this relationship. This simple yet powerful concept allows us to make predictions on new data.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data manipulation.
- **Numpy** for data simulation.
- **Scikit-learn** for building the machine learning model.
- **Matplotlib** for data visualization.
""")

# -- Code and Model Demonstration --
@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    units_sold_data = np.random.normal(loc=500, scale=150, size=500).astype(int)
    units_sold_data[units_sold_data < 0] = 0
    total_revenue_data = (units_sold_data * 150) + np.random.normal(loc=0, scale=15000, size=500)
    df = pd.DataFrame({
        'Units Sold': units_sold_data,
        'Total Revenue': total_revenue_data
    })
    X = df[['Units Sold']]
    y = df['Total Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X, y, df

model, X, y, df = generate_and_train_model()

# -- Display the Raw Data --
st.subheader("Raw Simulated Data")
st.write("The table below shows the data our model was trained on. Each row represents a sales record with Units Sold and its corresponding Total Revenue.")
st.dataframe(df.head(10))

st.subheader("Make a Prediction")
st.info('Adjust the slider below to see the predicted revenue for a given number of units sold.')
units_sold = st.slider('Units Sold', min_value=1, max_value=1000, value=500, step=10)
predicted_revenue = model.predict([[units_sold]])
st.metric(label=f"Predicted Revenue for {units_sold:,} units", value=f"${predicted_revenue[0]:,.2f}")

st.subheader("Visualizing the Model")
st.write("This chart shows our model's 'line of best fit' and highlights your prediction as you adjust the slider.")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color='blue', label='Simulated Sales Data')
ax.plot(X, model.predict(X), color='red', linewidth=2, label='Linear Regression Predictions')
ax.scatter(units_sold, predicted_revenue, color='green', s=200, label='Your Prediction')
ax.axvline(units_sold, color='green', linestyle='--', linewidth=1)
ax.axhline(predicted_revenue, color='green', linestyle='--', linewidth=1)
ax.set_title('Simulated Units Sold vs. Total Revenue')
ax.set_xlabel('Units Sold')
ax.set_ylabel('Total Revenue')
ax.legend()
ax.grid(True)
st.pyplot(fig)
