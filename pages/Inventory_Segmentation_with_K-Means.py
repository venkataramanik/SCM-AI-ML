import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Inventory Segmentation"
)

st.title("Inventory Segmentation with K-Means Clustering")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
Not all products are equally important. Inventory managers often use strategies like ABC analysis to categorize products based on their value, volume, or criticality. This allows them to allocate resources and attention where they matter most. This project demonstrates how K-Means clustering can automate this segmentation process.
""")

# -- K-Means Clustering --
st.subheader("K-Means Clustering")
st.write("""
**K-Means** is an unsupervised machine learning algorithm that finds groups (or clusters) in a dataset. It works by iteratively assigning data points to clusters and adjusting the cluster's center until the groups are well-defined. We'll use it here to segment products into different categories based on their simulated attributes.
""")

# -- Industry Applicability Section --
st.subheader("Industry Applicability")
st.write("""
- **Retail & Distribution:** Segmenting a product catalog to apply different re-ordering policies. For example, high-volume products (Cluster 1) may require a safety stock, while low-volume products (Cluster 3) can be managed with a simple re-order point.
- **Manufacturing:** Categorizing raw materials based on usage rate and cost to optimize procurement and storage.
- **Warehouse Management:** Grouping items by their picking frequency or size to optimize warehouse layout and reduce picking time.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data manipulation.
- **Numpy** for data simulation.
- **Scikit-learn** for building the **K-Means** clustering model.
- **Matplotlib** and **Seaborn** for data visualization.
""")

# -- Code and Model Demonstration --
@st.cache_data
def generate_and_cluster_data(n_clusters):
    # Simulate inventory data for different products
    # We will create 3 distinct groups of products
    np.random.seed(42)
    
    # Cluster 1: High-Value, High-Volume
    products1 = pd.DataFrame({
        'Product ID': range(1, 101),
        'Daily Sales': np.random.normal(loc=150, scale=30, size=100),
        'Unit Cost': np.random.normal(loc=120, scale=20, size=100)
    })
    
    # Cluster 2: Medium-Value, Medium-Volume
    products2 = pd.DataFrame({
        'Product ID': range(101, 251),
        'Daily Sales': np.random.normal(loc=50, scale=15, size=150),
        'Unit Cost': np.random.normal(loc=50, scale=10, size=150)
    })
    
    # Cluster 3: Low-Value, Low-Volume (or slow-moving)
    products3 = pd.DataFrame({
        'Product ID': range(251, 501),
        'Daily Sales': np.random.normal(loc=10, scale=5, size=250),
        'Unit Cost': np.random.normal(loc=10, scale=3, size=250)
    })
    
    df = pd.concat([products1, products2, products3]).reset_index(drop=True)
    df['Daily Sales'] = df['Daily Sales'].astype(int)
    df['Unit Cost'] = df['Unit Cost'].round(2)
    df.loc[df['Daily Sales'] < 1, 'Daily Sales'] = 1
    
    # Prepare data for clustering - we use Sales and Cost
    X = df[['Daily Sales', 'Unit Cost']]
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    return df, kmeans.cluster_centers_

if st.button("Generate New Data"):
    st.cache_data.clear()
    st.rerun()

# Use a slider to choose the number of clusters
n_clusters = st.slider('Select number of clusters', min_value=2, max_value=5, value=3)

df, centers = generate_and_cluster_data(n_clusters)

st.subheader("Product Segmentation Visualization")
st.write("This scatter plot shows all products, colored by their assigned cluster. The squares represent the cluster centers.")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Daily Sales',
    y='Unit Cost',
    hue='Cluster',
    palette='deep',
    ax=ax,
    s=100
)
ax.scatter(centers[:, 0], centers[:, 1], marker='s', s=200, color='red', label='Cluster Centers')
ax.set_title(f'K-Means Clustering of Products ({n_clusters} Clusters)')
ax.set_xlabel('Daily Sales')
ax.set_ylabel('Unit Cost')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Product Details by Cluster")
st.write("Select a cluster to view the details of the products that fall into that category.")

cluster_options = sorted(df['Cluster'].unique())
selected_cluster = st.selectbox('Choose a Cluster', options=cluster_options)

cluster_df = df[df['Cluster'] == selected_cluster].drop(columns=['Cluster']).reset_index(drop=True)
st.dataframe(cluster_df)
