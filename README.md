##Loan Eligibility Prediction System

The Loan Eligibility Prediction System is a machine learning-powered web application designed to predict loan approval based on applicant data such as income, loan amount, education, employment status, and credit score. This project is built using various machine learning algorithms and provides visualizations to compare the model performances.

#Features
//Multiple Machine Learning Models:

Implements Decision Tree, Extra Trees, XGBoost, and LightGBM classifiers.
Predicts loan eligibility based on key financial and personal data.

//Model Evaluation:

Compares model performance using accuracy metrics.
Visualizations include:
Horizontal Bar Chart for accuracy comparison.
Confusion Matrix for model prediction evaluation.
ROC Curves for performance evaluation.
Box Plot to show accuracy distribution across models.

//Flask Web Interface:
Simple and intuitive web interface for users to input data and get real-time loan approval predictions.

//Bank Recommendations:
Provides bank suggestions based on loan eligibility predictions.

//Installation
Clone the repository:

bash

-git clone https://github.com/perlamadhava/LoanEligibilityPrediction-ML.git
-cd LoanEligibilityPrediction-ML
Create and activate a virtual environment (optional but recommended):

bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
Install the required dependencies:

bash

pip install -r requirements.txt
Run the application:

bash

python app.py
Access the web application: Open your browser and go to http://127.0.0.1:5000/.

//Technologies Used
Backend: Flask, Python
Libraries: pandas, scikit-learn, XGBoost, LightGBM, Matplotlib
Machine Learning: Decision Tree, Extra Trees, XGBoost, LightGBM
Visualization: Matplotlib for generating model comparison charts
Project Structure
php

LoanEligibilityPrediction-ML/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── model_training.py    # Script to train models and generate visualizations
│   └── static/              # Contains generated images (confusion matrices, ROC curves, etc.)
├── dataset/
│   └── loan_approval_dataset.csv  # Dataset used for training the models
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)

//Future Enhancements
Improve the model accuracy by hyperparameter tuning.
Add more financial data features to improve prediction quality.
Implement additional model performance metrics like precision, recall, and F1 score.

//License
This project is licensed under the MIT License. See the LICENSE file for details.
