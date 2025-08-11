import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Demand Forecasting - Random Forest"
)

st.title("Solving Demand Forecasting with Random Forest")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
While linear regression is a great starting point, real-world demand forecasting involves complex, non-linear factors that a simple model can't capture. Variables like seasonality, promotions, and market events often have a non-linear impact on sales.
""")

# -- The Concept: Simulated Data --
st.subheader("Simulated Data for Our Playground")
st.write("""
This project uses a simulated dataset, but this time with an added 'promotional_campaign' feature to show how a more advanced model can handle multiple variables.
""")

# -- Concept Explanation --
st.subheader("The Concept: Random Forest (Supervised Learning)")
st.write("""
The **Random Forest** algorithm is a powerful **Supervised Learning** method that builds multiple "decision trees" and combines their outputs to make a more accurate prediction. Think of it as a committee of experts: each tree analyzes the data differently, and their collective judgment is more robust than any single one. This makes it highly effective at capturing complex, non-linear relationships in the data.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data manipulation.
- **Numpy** for data simulation.
- **Scikit-learn** for building the Random Forest model.
- **Matplotlib** for data visualization.
""")

# -- Code and Model Demonstration --
@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    units_sold_data = np.random.normal(loc=500, scale=150, size=500).astype(int)
    units_sold_data[units_sold_data < 0] = 0

    # Add a new feature: promotional_campaign (0 or 1)
    promotional_campaign = np.random.choice([0, 1], size=500, p=[0.7, 0.3])

    # Total Revenue now depends on units_sold and the campaign
    total_revenue_data = (units_sold_data * 150) + (promotional_campaign * 20000) + np.random.normal(loc=0, scale=10000, size=500)

    df = pd.DataFrame({
        'Units Sold': units_sold_data,
        'Promotional Campaign': promotional_campaign,
        'Total Revenue': total_revenue_data
    })

    X = df[['Units Sold', 'Promotional Campaign']]
    y = df['Total Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X, y, df

model, X, y, df = generate_and_train_model()

st.subheader("Raw Simulated Data")
st.write("The table below shows our simulated data, now with an added 'Promotional Campaign' variable.")
st.dataframe(df.head(10))

st.subheader("Make a Prediction")
st.info('Adjust the inputs below to see the predicted revenue using a Random Forest model.')
col1, col2 = st.columns(2)
with col1:
    units_sold = st.slider('Units Sold', min_value=1, max_value=1000, value=500, step=10)
with col2:
    campaign = st.checkbox('Promotional Campaign Active?')
    campaign_value = 1 if campaign else 0

predicted_revenue = model.predict([[units_sold, campaign_value]])
st.metric(label=f"Predicted Revenue for {units_sold:,} units", value=f"${predicted_revenue[0]:,.2f}")

st.subheader("Visualizing the Prediction")
st.write("This chart shows how the Random Forest model captures the different revenue patterns for promotional and non-promotional campaigns.")
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Units Sold',
    y='Total Revenue',
    hue='Promotional Campaign',
    palette='deep',
    ax=ax,
    s=100
)

ax.scatter(units_sold, predicted_revenue, color='red', s=200, label='Your Prediction', zorder=5)
ax.axvline(units_sold, color='red', linestyle='--', linewidth=1)
ax.axhline(predicted_revenue, color='red', linestyle='--', linewidth=1)

ax.set_title('Simulated Units Sold vs. Total Revenue (Random Forest)')
ax.set_xlabel('Units Sold')
ax.set_ylabel('Total Revenue')
ax.legend(title='Promotional Campaign')
ax.grid(True)
st.pyplot(fig)
