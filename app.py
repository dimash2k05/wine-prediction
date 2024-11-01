from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

wine_data = pd.read_csv('winequality.csv')
numeric_cols = wine_data.select_dtypes(include=[np.number]).columns
wine_data_cleaned = wine_data.copy()
wine_data_cleaned[numeric_cols] = wine_data_cleaned[numeric_cols].fillna(wine_data_cleaned[numeric_cols].median())
wine_data_encoded = pd.get_dummies(wine_data_cleaned, columns=['type'])

x = wine_data_encoded.drop('quality', axis=1)
y = wine_data_encoded['quality']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model_baseline = LogisticRegression(max_iter=1000)
model_baseline.fit(x_scaled, y)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        type_red = 0
        type_white = 0
        if request.form['type'] == 'white':
            type_white = 1
        else:
            type_red = 1
        features = [
            float(request.form['fixed acidity']),
            float(request.form['volatile acidity']),
            float(request.form['citric acid']),
            float(request.form['residual sugar']),
            float(request.form['chlorides']),
            float(request.form['free sulfur dioxide']),
            float(request.form['total sulfur dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol']),
            type_red,
            type_white
        ]
        features_reshaped = np.array(features).reshape(1, -1)
        for column in features_reshaped:
            print(column)
        prediction = model_baseline.predict(features_reshaped)
        return render_template('index.html', prediction=prediction[0])
    except Exception as err:
        return f'Error: {str(err)}'  
    
if __name__ == "__main__":
    app.run(debug=True)