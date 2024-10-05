import pickle
import numpy as np

def load_model(model_path='backend/lightgbm_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_loan_status(model, features):
    prediction = model.predict(features)
    loan_status = 'Approved' if prediction == 1 else 'Rejected'
    return loan_status

def check_bank_eligibility(loan_type, age, income, self_employed, loan_amount, loan_term, cibil_score):
    eligible_banks = []

    # Personal Loan Eligibility
    if loan_type == 'personal':
        # HDFC Bank
        if 21 <= age <= 60 and income >= 25000:
            eligible_banks.append('HDFC Bank')
        # ICICI Bank
        if (self_employed and income >= 1500000) or (not self_employed and income >= 30000):
            eligible_banks.append('ICICI Bank')
        # SBI Bank
        if 21 <= age <= 58 and income >= 15000:
            eligible_banks.append('SBI Bank')
        # Union Bank
        if income >= 1200000:
            eligible_banks.append('Union Bank')
        # Axis Bank
        if 21 <= age <= 60 and income >= 15000:
            eligible_banks.append('Axis Bank')
    
    # Home Loan Eligibility
    elif loan_type == 'home':
        # HDFC Bank
        if 21 <= age <= 65 and income >= 10000:
            eligible_banks.append('HDFC Bank')
        # ICICI Bank
        if 21 <= age <= 65 and income >= 10000:
            eligible_banks.append('ICICI Bank')
        # SBI Bank
        if 18 <= age <= 70:
            eligible_banks.append('SBI Bank')
        # Union Bank
        if income >= 1200000:
            eligible_banks.append('Union Bank')
        # Axis Bank
        if 21 <= age <= 65 and income >= 10000:
            eligible_banks.append('Axis Bank')

    # Education Loan Eligibility
    elif loan_type == 'education':
        # HDFC Bank
        if 16 <= age <= 35:
            eligible_banks.append('HDFC Bank')
        # ICICI Bank
        if 18 <= age <= 35:
            eligible_banks.append('ICICI Bank')
        # SBI Bank
        if 18 <= age <= 70:
            eligible_banks.append('SBI Bank')
        # Union Bank
        if 18 <= age <= 75:
            eligible_banks.append('Union Bank')
        # Axis Bank
        if 18 <= age <= 35:
            eligible_banks.append('Axis Bank')

    # Business Loan Eligibility
    elif loan_type == 'business':
        # HDFC Bank
        if self_employed and income >= 150000 and loan_amount <= 4000000:
            eligible_banks.append('HDFC Bank')
        # ICICI Bank
        if self_employed and income >= 4000000:
            eligible_banks.append('ICICI Bank')
        # SBI Bank
        if 18 <= age <= 65:
            eligible_banks.append('SBI Bank')
        # Union Bank
        if income >= 1200000:
            eligible_banks.append('Union Bank')
        # Axis Bank
        if self_employed and income >= 3000000 and age >= 21:
            eligible_banks.append('Axis Bank')

    return eligible_banks
