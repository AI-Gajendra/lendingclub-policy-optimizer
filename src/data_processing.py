import pandas as pd
import numpy as np

def format_emp_length(emp_length_str):
    
    if pd.isna(emp_length_str):
        return np.nan
    if emp_length_str == '< 1 year':
        return 0
    if emp_length_str == '10+ years':
        return 10
    try:
        return int(''.join(filter(str.isdigit, emp_length_str)))
    except (ValueError, TypeError):
        return np.nan

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # 1. Define Target and Filter for Concluded Loans
    concluded_statuses = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(concluded_statuses)].copy()
    
    if df.empty:
        print("Warning: No loans with 'Fully Paid' or 'Charged Off' status found.")
        return df
        
    df['target'] = (df['loan_status'] == 'Charged Off').astype(int)
    
    # 2. Drop Leaky, Redundant, or Uninformative Columns
    # This list is now much more comprehensive based on EDA.
    cols_to_drop = [
        # --- Identifiers and Redundant ---
        'id', 'member_id', 'sub_grade', 'emp_title', 'title', 'zip_code',
        'url', 'desc', 
        
        # --- Post-Issue and Leakage Columns ---
        'loan_status', 'funded_amnt', 'funded_amnt_inv', 'out_prncp', 
        'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 
        'total_rec_int', 'total_rec_late_fee', 'recoveries', 
        'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 
        'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 
        'last_fico_range_low', 'debt_settlement_flag',
        
        # --- Columns with >40% Missing Values or Low Predictive Power ---
        'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
        'annual_inc_joint', 'dti_joint', 'verification_status_joint',
        'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 
        'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 
        'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 
        'inq_last_12m', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
        'mths_since_recent_revol_delinq', 'revol_bal_joint', 'sec_app_fico_range_low',
        'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',
        'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
        'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
        'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog',
        
        # --- All Hardship, Settlement, Deferral Columns ---
        *[col for col in df.columns if 'hardship' in col],
        *[col for col in df.columns if 'settlement' in col],
        *[col for col in df.columns if 'deferral' in col],
        'payment_plan_start_date', 'orig_projected_additional_accrued_interest',
        'disbursement_method'
    ]
    df = df.drop(columns=list(set(cols_to_drop)), errors='ignore')

    # 3. Clean and Engineer Features
    df['term'] = df['term'].str.extract('(\d+)').astype(int)
    df['emp_length'] = df['emp_length'].apply(format_emp_length)
    
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['credit_history_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
    df = df.drop(columns=['earliest_cr_line', 'issue_d'])

    # 4. Handle Remaining Missing Values
    
    # Drop rows where critical financial info is missing
    df.dropna(subset=['dti', 'revol_util', 'loan_amnt', 'int_rate', 'installment'], inplace=True)


    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            

    df.dropna(inplace=True)
    
    # 5. Encode Categorical Variables
    if df.empty:
        print("Warning: DataFrame is empty after handling missing values.")
        return df

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    return df