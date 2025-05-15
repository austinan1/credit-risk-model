# Credit Risk Assessment Tool

## Overview

This Credit Risk Assessment Tool is a web-based dashboard that predicts the probability of loan default using machine learning. The model achieves an AUC (Area Under the ROC Curve) of over 0.85, making it a reliable tool for credit risk assessment.

The dashboard uses an ensemble of machine learning models:
- Balanced Random Forest
- XGBoost
- Random Forest

The final prediction is a weighted average of these models, providing a robust risk assessment.

## Features

- **Real-time Credit Risk Assessment**: Enter applicant information and get an immediate risk prediction
- **Risk Categorization**: Applicants are classified into risk categories (Very Low, Low, Moderate, High, Very High)
- **Visual Risk Representation**: Visual indicators make risk levels easy to interpret
- **Key Risk Factor Identification**: The tool highlights factors contributing to the risk level
- **Recommendations**: Provides suggestions based on the applicant's risk profile

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - joblib
  - matplotlib
  - seaborn
  - streamlit

### Installation

1. Clone this repository or download the files
```bash
git clone https://github.com/austinan1/credit-risk-assessment.git
cd credit-risk-assessment
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Ensure you have the trained models
The application requires trained models in the `models` directory. If you're using the notebook to train models, make sure to run the model export cells. Alternatively, you can use the script to create dummy models for testing:
```bash
python create_dummy_models.py
```

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

This will open the dashboard in your default web browser.

## Usage

1. Enter the applicant's information in the form:
   - Revolving Utilization of Unsecured Lines
   - Age
   - Number of Times 30-59 Days Past Due
   - Debt Ratio
   - Monthly Income
   - Number of Open Credit Lines and Loans
   - Number of Times 90 Days Late
   - Number of Real Estate Loans or Lines
   - Number of Times 60-89 Days Past Due
   - Number of Dependents

2. Click "Predict Default Risk"

3. Review the results:
   - Default Probability
   - Risk Category
   - Visualization of the risk level
   - Key risk factors
   - Recommendation

## Model Performance

The ensemble model achieves an AUC of over 0.85 on the test set, indicating excellent discriminatory power between defaulters and non-defaulters. The model has been optimized for both predictive power and interpretability.

## Files in this Repository

- `app.py`: The main Streamlit dashboard application
- `prediction.py`: Contains the prediction functionality
- `models/`: Directory containing trained models
  - `balanced_rf_model.pkl`: Balanced Random Forest model
  - `xgb_model.pkl`: XGBoost model
  - `rf_model.pkl`: Random Forest model
  - `preprocessing_info.pkl`: Contains preprocessing information

## Troubleshooting

If you encounter the error "Model files not found", ensure:
1. The `models` directory exists in the same location as `app.py`
2. All model files have been exported correctly
3. You have the necessary permissions to read the files

## Future Improvements

- Add batch processing capabilities for multiple applicants
- Implement model monitoring for drift detection
- Add user authentication for secure access
- Expand feature set to incorporate additional risk factors
- Create API endpoints for integration with other systems


## Acknowledgements

- The model is trained on credit risk data similar to datasets from Kaggle
- Thanks to the scikit-learn, imbalanced-learn, and XGBoost teams for their excellent libraries

---

For questions or support, please open an issue on this repository or contact austina@andrew.cmu.edu.
```