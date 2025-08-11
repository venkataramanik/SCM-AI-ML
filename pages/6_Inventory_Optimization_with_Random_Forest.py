import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Inventory Optimization"
)

st.title("Inventory Optimization with a Machine Learning Classifier")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
An inventory manager's core job is to balance inventory costs with customer service. Holding too much stock is expensive, while holding too little leads to stockouts and lost sales. This project shows how a machine learning model can automate the decision of when to place a new order.
""")

# -- Machine Learning Classification --
st.subheader("Machine Learning Classification")
st.write("""
This project uses **Classification**, a type of supervised learning where a model learns to predict a categorical label. Based on historical data, our model learns to classify a given inventory situation as `Order`, `Hold`, or `Urgent Order`.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data manipulation and creating a dataframe.
- **Numpy** for data simulation.
- **Scikit-learn** for building the **Random Forest Classifier** model.
- **Matplotlib** for data visualization.
""")

# -- Code and Model Demonstration --
@st.cache_data
def generate_and_train_model():
    # Simulate inventory data
    sales = np.random.normal(loc=100, scale=30, size=1000).astype(int)
    sales[sales < 0] = 0
    stock = np.random.normal(loc=500, scale=100, size=1000).astype(int)
    stock[stock < 0] = 0
    lead_time = np.random.normal(loc=7, scale=2, size=1000).astype(int)
    lead_time[lead_time < 0] = 1

    df = pd.DataFrame({
        'Daily Sales': sales,
        'Current Stock': stock,
        'Lead Time (days)': lead_time
    })

    # Create the target variable: 'Order Recommendation'
    # This is a simplified logic to generate the target labels for training
    df['Order Recommendation'] = 'Hold'
    df.loc[df['Current Stock'] < df['Daily Sales'] * 10, 'Order Recommendation'] = 'Order'
    df.loc[df['Current Stock'] < df['Daily Sales'] * 5, 'Order Recommendation'] = 'Urgent Order'

    # Prepare data for the classifier
    X = df[['Daily Sales', 'Current Stock', 'Lead Time (days)']]
    y = df['Order Recommendation']

    # Train a Random Forest Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = generate_and_train_model()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

if st.button("Generate New Data"):
    st.cache_data.clear()
    st.rerun()

st.subheader("Interactive Order Recommendation")
st.write(f"Model Accuracy on Test Data: **{accuracy:.2f}**")
st.info("Adjust the inputs below to get a real-time inventory recommendation from the model.")

col1, col2, col3 = st.columns(3)
with col1:
    daily_sales = st.number_input('Average Daily Sales', min_value=1, max_value=500, value=100)
with col2:
    current_stock = st.number_input('Current Stock Level', min_value=0, max_value=2000, value=500)
with col3:
    lead_time = st.number_input('Supplier Lead Time (days)', min_value=1, max_value=20, value=7)

user_data = pd.DataFrame([[daily_sales, current_stock, lead_time]], columns=['Daily Sales', 'Current Stock', 'Lead Time (days)'])
recommendation = model.predict(user_data)[0]

# Display the recommendation with a corresponding color
if recommendation == 'Urgent Order':
    st.markdown(f"### <span style='color:red;'>Recommendation: {recommendation}</span>", unsafe_allow_html=True)
elif recommendation == 'Order':
    st.markdown(f"### <span style='color:orange;'>Recommendation: {recommendation}</span>", unsafe_allow_html=True)
else:
    st.markdown(f"### <span style='color:green;'>Recommendation: {recommendation}</span>", unsafe_allow_html=True)

# Visualizing Feature Importance
st.subheader("Model Insights: Feature Importance")
st.write("This chart shows which factors the model considers most important when making a recommendation.")
importances = model.feature_importances_
feature_names = ['Daily Sales', 'Current Stock', 'Lead Time (days)']
fig, ax = plt.subplots()
ax.barh(feature_names, importances)
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
ax.grid(axis='x', linestyle='--')
st.pyplot(fig)
