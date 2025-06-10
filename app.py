
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from prediction import predict_credit_risk

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
    }
    .risk-very-low {
        color: #10B981;
        font-weight: bold;
    }
    .risk-low {
        color: #84CC16;
        font-weight: bold;
    }
    .risk-moderate {
        color: #F59E0B;
        font-weight: bold;
    }
    .risk-high {
        color: #EF4444;
        font-weight: bold;
    }
    .risk-very-high {
        color: #7F1D1D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Credit Risk Assessment Tool</p>', unsafe_allow_html=True)

st.write("""
This application predicts the probability of a loan applicant defaulting on their payment.
The model achieves an AUC of over 0.85, providing reliable risk assessments.
""")

# Sidebar with info
with st.sidebar:
    st.title("About")
    st.info("""
    This tool uses an ensemble of machine learning models to predict credit risk.
    
    **Models used:**
    - Balanced Random Forest
    - XGBoost
    - Random Forest
    
    The final prediction is a weighted average of these models.
    """)
    
    st.markdown("### How to use")
    st.write("""
    1. Enter applicant information in the form
    2. Click 'Predict Default Risk'
    3. Review the results and risk assessment
    """)
    
    st.markdown("### Risk Categories")
    st.markdown("- **Very Low Risk**: < 5% default probability")
    st.markdown("- **Low Risk**: 5-15% default probability")
    st.markdown("- **Moderate Risk**: 15-30% default probability")
    st.markdown("- **High Risk**: 30-50% default probability")
    st.markdown("- **Very High Risk**: > 50% default probability")

# Main form
st.markdown('<p class="sub-header">Applicant Information</p>', unsafe_allow_html=True)

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    revolving_utilization = st.number_input('Revolving Utilization of Unsecured Lines', 
                                          min_value=0.0, max_value=20.0, value=0.5,
                                          help="Total balance on credit cards and personal lines of credit divided by the sum of credit limits")
    
    age = st.number_input('Age', 
                        min_value=18, max_value=100, value=45,
                        help="Age of the borrower in years")
    
    num_30_59_days_late = st.number_input('Number of Times 30-59 Days Past Due', 
                                        min_value=0, max_value=20, value=0,
                                        help="Number of times the borrower has been 30-59 days past due on a payment")
    
    debt_ratio = st.number_input('Debt Ratio', 
                               min_value=0.0, max_value=10.0, value=0.5,
                               help="Monthly debt payments divided by monthly gross income")
    
    monthly_income = st.number_input('Monthly Income ($)', 
                                   min_value=0, max_value=100000, value=5000,
                                   help="Monthly income in dollars")

with col2:
    num_open_credit_lines = st.number_input('Number of Open Credit Lines and Loans', 
                                          min_value=0, max_value=30, value=8,
                                          help="Number of open loans (installment like car loan or mortgage) and lines of credit (e.g. credit cards)")
    
    num_90_days_late = st.number_input('Number of Times 90 Days Late', 
                                      min_value=0, max_value=20, value=0,
                                      help="Number of times the borrower has been 90 or more days past due")
    
    num_real_estate_loans = st.number_input('Number of Real Estate Loans or Lines', 
                                          min_value=0, max_value=20, value=1,
                                          help="Number of mortgage and real estate loans including home equity lines of credit")
    
    num_60_89_days_late = st.number_input('Number of Times 60-89 Days Past Due', 
                                        min_value=0, max_value=20, value=0,
                                        help="Number of times the borrower has been 60-89 days past due on a payment")
    
    num_dependents = st.number_input('Number of Dependents', 
                                   min_value=0, max_value=10, value=0,
                                   help="Number of dependents in family excluding themselves (spouse, children, etc.)")

# Create applicant data dictionary
applicant_data = {
    'revolvingutilizationofunsecuredlines': revolving_utilization,
    'age': age,
    'numberoftime3059dayspastduenotworse': num_30_59_days_late,
    'debtratio': debt_ratio,
    'monthlyincome': monthly_income,
    'numberofopencreditlinesandloans': num_open_credit_lines,
    'numberoftimes90dayslate': num_90_days_late,
    'numberrealestateloansorlines': num_real_estate_loans,
    'numberoftime6089dayspastduenotworse': num_60_89_days_late,
    'numberofdependents': num_dependents
}

# Prediction section
st.markdown('---')
predict_button = st.button('Predict Default Risk', use_container_width=True)

if predict_button:
    # Show a spinner while making prediction
    with st.spinner('Calculating risk assessment...'):
        # Make prediction
        try:
            result = predict_credit_risk(applicant_data)
            
            # Display results
            default_prob = result['default_probability'].iloc[0]
            risk_category = result['risk_category'].iloc[0]
            
            st.markdown('<p class="sub-header">Risk Assessment Results</p>', unsafe_allow_html=True)
            
            # Show probability and risk category
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Default Probability", f"{default_prob:.2%}")
            
            with col2:
                # Style the risk category based on level
                if risk_category == "Very Low Risk":
                    st.markdown(f'<p class="risk-very-low">Risk Category: {risk_category}</p>', unsafe_allow_html=True)
                elif risk_category == "Low Risk":
                    st.markdown(f'<p class="risk-low">Risk Category: {risk_category}</p>', unsafe_allow_html=True)
                elif risk_category == "Moderate Risk":
                    st.markdown(f'<p class="risk-moderate">Risk Category: {risk_category}</p>', unsafe_allow_html=True)
                elif risk_category == "High Risk":
                    st.markdown(f'<p class="risk-high">Risk Category: {risk_category}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="risk-very-high">Risk Category: {risk_category}</p>', unsafe_allow_html=True)
            
            # Determine color based on risk
            if risk_category == "Very Low Risk":
                color = '#10B981'  # green
            elif risk_category == "Low Risk":
                color = '#84CC16'  # light green
            elif risk_category == "Moderate Risk":
                color = '#F59E0B'  # amber
            elif risk_category == "High Risk":
                color = '#EF4444'  # red
            else:
                color = '#7F1D1D'  # dark red
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Default Risk'], [default_prob], color=color)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.05, 0.15, 0.30, 0.50, 1.0])
            ax.set_xticklabels(['0%', '5%', '15%', '30%', '50%', '100%'])
            
            # Add threshold lines
            ax.axvline(x=0.05, color='#10B981', linestyle='--')
            ax.axvline(x=0.15, color='#84CC16', linestyle='--')
            ax.axvline(x=0.30, color='#F59E0B', linestyle='--')
            ax.axvline(x=0.50, color='#EF4444', linestyle='--')
            
            st.pyplot(fig)
            
            # Explain important factors
            st.markdown("### Key Risk Factors")
            
            # This would be based on SHAP values in a production app
            # For this example, we'll use some business logic
            risk_factors = []
            
            if revolving_utilization > 0.5:
                risk_factors.append("High credit utilization (>50%)")
            if debt_ratio > 0.43:
                risk_factors.append("High debt-to-income ratio (>43%)")
            if num_30_59_days_late > 0 or num_60_89_days_late > 0 or num_90_days_late > 0:
                risk_factors.append("History of delinquent payments")
            if monthly_income < 3000:
                risk_factors.append("Lower monthly income (<$3,000)")
            if age < 25:
                risk_factors.append("Young borrower (higher statistical risk)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No significant risk factors identified.")
                
            # Recommendation
            st.markdown("### Recommendation")
            if risk_category in ["Very Low Risk", "Low Risk"]:
                st.success("✅ This applicant is eligible for approval with standard terms.")
            elif risk_category == "Moderate Risk":
                st.warning("⚠️ This applicant may be approved with adjusted terms (higher interest rate or lower loan amount).")
            else:
                st.error("❌ This applicant presents a high default risk and should be declined or require additional guarantees.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please make sure you've saved your trained models in the 'models' directory.")