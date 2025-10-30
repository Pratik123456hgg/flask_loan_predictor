import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model, scaler, and column list
try:
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('loan_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("✅ Model, scaler, and columns loaded successfully.")
except FileNotFoundError:
    print("🚨 Error: Model/scaler/column files not found. Run the notebook cell to save them.")
    exit()

@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    
    # 1. Get data from the form
    form_data = {
        'person_age': float(request.form['person_age']),
        'person_income': float(request.form['person_income']),
        'loan_amnt': float(request.form['loan_amnt']),
        'loan_percent_income': float(request.form['loan_percent_income']),
        'credit_score': float(request.form['credit_score']),
        'person_emp_exp': float(request.form['person_emp_exp']),
        'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length']),
        
        # --- THIS IS THE UPDATED PART ---
        # Get the value from the new dropdown (it will be '0' or '1')
        'previous_loan_defaults_on_file': int(request.form['previous_loan_defaults_on_file']),
        
        # --- Hardcoded values from your notebook's prediction cell ---
        'person_gender': 1,
        'person_education': 2,
        'person_home_ownership': 1,
        'loan_intent': 2,
        'loan_int_rate': 10.0,
    }

    # 2. Create a DataFrame in the correct order
    input_df = pd.DataFrame([form_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 3. Scale the input data
    input_scaled = scaler.transform(input_df)

    # 4. Get probability and prediction
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob > 0.5 else 0

    # 5. Format the output
    result = "Loan Approved" if prediction == 1 else "Loan Rejected"
    probability_score = f"{prob*100:.2f}%"

    return render_template('result.html', 
                           prediction_text=result, 
                           probability_score=probability_score)

if __name__ == "__main__":
    app.run(debug=True)