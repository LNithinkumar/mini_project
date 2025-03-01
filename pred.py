import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load Dataset (with only one store)
@st.cache_data
def load_data():
    data = pd.read_csv("black_friday_sales.csv")  # Path to the CSV with data for one store
    return data

# Preprocess Data
def preprocess_data(data):
    if 'Date' in data.columns:
        data['Year'] = pd.to_datetime(data['Date']).dt.year
    else:
        st.write("No 'Date' column found. Please ensure the data contains date or year information.")
    
    data['Product_Category_2'] = data['Product_Category_2'].fillna(data['Product_Category_2'].mode()[0])
    data['Product_Category_3'] = data['Product_Category_3'].fillna(data['Product_Category_3'].mode()[0])
    
    data.dropna(inplace=True)
    
    return data

# Load the data
data = load_data()

# Preprocess the data
data = preprocess_data(data)

# Group data by Product_ID and Year for sales prediction
if 'Product_ID' in data.columns:
    product_sales = data.groupby(['Product_ID', 'Year']).agg({'Purchase': 'sum'}).reset_index()
else:
    st.error("Column 'Product_ID' not found in the dataset. Please check your CSV file.")

product_ids = product_sales['Product_ID'].unique()

# Menu for navigation
menu = ["Home", "Predicted Sales", "Suggest Products to Order More", "Sales Visualization"]
choice = st.sidebar.selectbox("Select an Option", menu)

latest_year = product_sales['Year'].max()
next_year = latest_year + 1

# Home page content
def home_page():
    st.title("Black Friday Sales Prediction")
    st.subheader("Welcome to the Dashboard")
    st.markdown("### About")
    st.markdown(
        "This dashboard predicts product sales for the upcoming year, "
        "suggests products to order more, and visualizes sales data.")

# Predicted Sales Page
def predicted_sales():
    predictions = []
    for product_id in product_ids:
        product_data = product_sales[product_sales['Product_ID'] == product_id]
        X = product_data['Year'].values.reshape(-1, 1)
        y = product_data['Purchase']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        input_data = np.array([[next_year]])
        predicted_sales = model.predict(input_data)

        predictions.append((product_id, predicted_sales[0]))

    predictions_df = pd.DataFrame(predictions, columns=['Product_ID', 'Predicted_Sales'])

    st.title("Predicted Sales")
    st.subheader(f"For All Products in {next_year}")

    # Apply custom CSS for table styling
    st.markdown("""
        <style>
            .dataframe {
                font-size: 18px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(predictions_df, use_container_width=True)

# Suggest Products to Order More Page
def suggest_products_to_order_more():
    predictions = []
    for product_id in product_ids:
        product_data = product_sales[product_sales['Product_ID'] == product_id]
        X = product_data['Year'].values.reshape(-1, 1)
        y = product_data['Purchase']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        input_data = np.array([[next_year]])
        predicted_sales = model.predict(input_data)

        predictions.append((product_id, predicted_sales[0]))

    predictions_df = pd.DataFrame(predictions, columns=['Product_ID', 'Predicted_Sales'])
    predictions_df = predictions_df.sort_values(by='Predicted_Sales', ascending=False)
    top_5_products = predictions_df.head(5)

    st.title("Suggested Products to Order More")
    st.subheader(f"Top 5 Products for {next_year}")

    # Apply custom CSS for table styling
    st.markdown("""
        <style>
            .dataframe {
                font-size: 18px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(top_5_products, use_container_width=True)

    top_product = top_5_products.iloc[0]
    second_top_product = top_5_products.iloc[1]
    least_product = predictions_df.iloc[-1]
    
    st.write(f"**Recommendation**: Order **Product {top_product['Product_ID']}** in higher quantities.")
    st.write(f"**Keep in Stock**: Ensure stock of **Product {second_top_product['Product_ID']}**.")
    st.write(f"**Minimal Stock**: Keep minimal units of **Product {least_product['Product_ID']}**.")

# Sales Visualization Page
def sales_visualization():
    predictions = []
    for product_id in product_ids:
        product_data = product_sales[product_sales['Product_ID'] == product_id]
        X = product_data['Year'].values.reshape(-1, 1)
        y = product_data['Purchase']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        input_data = np.array([[next_year]])
        predicted_sales = model.predict(input_data)

        predictions.append((product_id, predicted_sales[0]))

    predictions_df = pd.DataFrame(predictions, columns=['Product_ID', 'Predicted_Sales'])

    st.title("Sales Visualization")
    st.subheader("Predicted Sales Distribution")

    n_products = st.slider("Select number of products", 1, 10, 5)
    top_products = predictions_df.sort_values(by='Predicted_Sales', ascending=False).head(n_products)

    fig1, ax1 = plt.subplots()
    ax1.pie(top_products['Predicted_Sales'], labels=top_products['Product_ID'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Menu Options
if choice == "Home":
    home_page()
elif choice == "Predicted Sales":
    predicted_sales()
elif choice == "Suggest Products to Order More":
    suggest_products_to_order_more()
elif choice == "Sales Visualization":
    sales_visualization()
    