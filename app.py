from flask import Flask, render_template, request, jsonify
import numpy as np
from backend.loan_prediction import load_model, predict_loan_status, check_bank_eligibility

app = Flask(__name__, template_folder='backend/templates')


model = load_model()

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    loan_type = data['loan_type']
    age = int(data['age'])
    self_employed = int(data['self_employed'])
    income_annum = float(data['income_annum'])
    loan_amount = float(data['loan_amount'])
    loan_term = int(data['loan_term'])
    cibil_score = int(data['cibil_score'])
    education = int(data['education'])

    
    features = np.array([[education, self_employed, income_annum, loan_amount, loan_term, cibil_score]])
    
    
    loan_status = predict_loan_status(model, features)

    
    eligible_banks = []
    if loan_status == 'Approved':
        eligible_banks = check_bank_eligibility(
            loan_type, age, income_annum, self_employed, loan_amount, loan_term, cibil_score
        )

    return jsonify({
        'loan_status': loan_status,
        'eligible_banks': eligible_banks
    })

if __name__ == '__main__':
    app.run(debug=True)
