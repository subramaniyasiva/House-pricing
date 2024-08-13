from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

app = Flask(__name__)

# Load the dataset
file = pd.read_csv(r"C:\Users\ASUS\Downloads\archive (13)\Housing.csv")


X = file[['area', 'bedrooms', 'bathrooms']]
y = file['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)


pickle.dump(model, open('model.pkl', 'wb'))


form_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Housing Price Prediction</title>
</head>
<body>
    <h1>Predict Housing Price</h1>
    <form action="/predict" method="post">
        <label for="area">Area (in sq ft):</label>
        <input type="text" id="area" name="area"><br><br>
        <label for="bedrooms">Number of Bedrooms:</label>
        <input type="text" id="bedrooms" name="bedrooms"><br><br>
        <label for="bathrooms">Number of Bathrooms:</label>
        <input type="text" id="bathrooms" name="bathrooms"><br><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
'''


result_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p>The predicted price is: ${{ prediction }}</p>
    <a href="/">Go back to the form</a>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
  
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    
 
    input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['area', 'bedrooms', 'bathrooms'])
    predicted_price = model.predict(input_data)[0]
    
   
    return render_template_string(result_html, prediction=predicted_price)

if __name__ == "__main__":
    app.run(port=5001)
