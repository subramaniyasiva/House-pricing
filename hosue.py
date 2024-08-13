import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file = pd.read_csv(r"C:\Users\ASUS\Downloads\archive (13)\Housing.csv")

# Select relevant features and the target variable
X = file[['area', 'bedrooms', 'bathrooms']]
y = file['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

# Optional: Print the coefficients and intercept
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
for i, col in enumerate(X.columns):
    print(f"{col}: {model.coef_[i]}")
def predict_price(area, bedrooms, bathrooms):
    # Create a DataFrame with the user input
    input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['area', 'bedrooms', 'bathrooms'])
    
    # Use the model to make predictions
    predicted_price = model.predict(input_data)
    
    return predicted_price[0]

# Example usage:
user_area = float(input("Enter the area: "))
user_bedrooms = int(input("Enter the number of bedrooms: "))
user_bathrooms = int(input("Enter the number of bathrooms: "))

predicted_price = predict_price(user_area, user_bedrooms, user_bathrooms)
print(f"The predicted price is: ${predicted_price:.2f}")
