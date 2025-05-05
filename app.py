
from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and encoder
model = joblib.load('random_forest_model.pkl')  # Or 'logistic_regression_model.pkl'
encoder = joblib.load('encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input from form
    input_data = {
        'PROP_TYPE': request.form['PROP_TYPE'],
        'PROP_AGE_BAND': int(request.form['PROP_AGE_BAND']),
        'FLOOR_AREA_BAND': int(request.form['FLOOR_AREA_BAND']),
        'CONSERVATORY_FLAG': request.form.get('CONSERVATORY_FLAG', '0'),  # default to '0' if empty
        'COUNCIL_TAX_BAND': request.form['COUNCIL_TAX_BAND'],
        'REGION': request.form['REGION'],
        'EPC': request.form['EPC']
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Transform using encoder
    encoded_input = encoder.transform(input_df)

    # Predict
    prediction = model.predict(encoded_input)[0]

    # Display result
    result_text = f"Predicted class: {prediction}"
    return render_template('index.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True)
