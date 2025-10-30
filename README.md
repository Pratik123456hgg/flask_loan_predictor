# Flask Loan Prediction Model

A web application built with Python and Flask that predicts the probability of a loan being approved based on applicant details.

This was a development project for my Computer Engineering studies at PCCOE. The model is a Logistic Regression classifier trained on a loan dataset, using SMOTE to handle class imbalance.



## Tech Stack
* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas, Joblib
* **Data Handling:** SMOTE (for class imbalance)
* **Frontend:** HTML, CSS

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/flask-loan-predictor.git](https://github.com/YOUR_USERNAME/flask-loan-predictor.git)
    cd flask-loan-predictor
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the required packages:
    ```bash
    pip install Flask joblib scikit-learn pandas
    ```

4.  Run the app:
    ```bash
    python app.py
    ```

5.  Open your browser and go to `http://127.0.0.1:5000`

---
*Developed by Pratik Bandgar*
