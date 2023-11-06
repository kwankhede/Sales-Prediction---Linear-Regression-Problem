# app.py
import streamlit as st
import pandas as pd
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the advertising data
data = pd.read_csv("data/advertising.csv")

# Create a linear regression model
X = data[["TV"]]
y = data["Sales"]
model = LinearRegression()
model.fit(X, y)

# Use the 'Agg' backend for Matplotlib
matplotlib.use('Agg')

# Streamlit app
st.title("Advertising Analysis")
st.write("This app explores the relationship between TV advertising and sales.")

# Sidebar for user input
st.sidebar.header("Choose Advertising Budget")
tv_budget = st.sidebar.slider("TV Budget ($)", min_value=0, max_value=300, step=10)

# Predict sales based on TV budget
predicted_sales = model.predict([[tv_budget]])

# Display results
st.write(f"TV Budget: ${tv_budget}")
st.write(f"Predicted Sales: {predicted_sales[0]:.2f}")

# Create a scatter plot using st.pyplot()
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data["TV"], y=data["Sales"])
plt.xlabel("TV Budget ($)")
plt.ylabel("Sales")
plt.title("TV Budget vs. Sales")
plt.axvline(x=tv_budget, color="r", linestyle="--", label=f"TV Budget: ${tv_budget}")
plt.legend()
st.pyplot(plt)
