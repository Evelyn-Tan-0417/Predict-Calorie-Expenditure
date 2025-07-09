"""
Utility Functions for Calorie Prediction Project

This module contains helper functions used throughout the project,
including evaluation metrics, file I/O, and submission generation.

Author: Evelyn
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error.
    
    This is the function used in the original notebook for evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    rmsle_score : float
        RMSLE score
    """
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def evaluation(model, X_test, y_test):
    """
    Evaluate model performance and create visualization.
    
    This is the main evaluation function from the original notebook.
    Calculates RMSE, RMSLE and shows scatter plot.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
        
    Returns:
    --------
    metrics : dict
        Dictionary with RMSE and RMSLE scores
    """
    y_pred = model.predict(X_test)
    
    # Ensure predictions are non-negative (calories can't be negative)
    y_pred = np.abs(y_pred)
    
    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred)
    rmsle_val = rmsle(y_test, y_pred)
    
    print(f"RMSE on the test set: {rmse}")
    print(f"RMSLE on the test set: {rmsle_val}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Calories")
    plt.ylabel("Predicted Calories")
    plt.title("Actual vs. Predicted Calories (Test Set)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()
    
    return {"RMSE": rmse, "RMSLE": rmsle_val}


def save_submission_file(model, X_test, sample_submission_path, output_path=None):
    """
    Generate Kaggle submission file from trained model.
    
    This replicates the save_submission_file function from the original notebook.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pandas.DataFrame
        Test features (properly aligned with training features)
    sample_submission_path : str
        Path to sample submission CSV file
    output_path : str, optional
        Output path for submission file. If None, overwrites sample submission.
        
    Returns:
    --------
    submission_df : pandas.DataFrame
        Submission dataframe
    """
    # Load the sample submission file
    submission_df = pd.read_csv(sample_submission_path)
    
    # Make predictions on the test data
    test_predictions = model.predict(X_test)
    
    # Ensure predictions are non-negative and convert to integers
    test_predictions = np.abs(test_predictions)
    test_predictions = test_predictions.round().astype(int)
    
    # Replace the 'Calories' column with our predictions
    submission_df['Calories'] = test_predictions
    
    # Display first few rows
    print(submission_df.head())
    
    # Save to file
    if output_path is None:
        output_path = sample_submission_path
    
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return submission_df


def download_and_extract_data(file_id, download_path='/content/downloaded_file.zip', 
                             extract_path='/content/unzipped_content'):
    """
    Download and extract competition data from Google Drive.
    
    This replicates the data downloading process from the original notebook.
    
    Parameters:
    -----------
    file_id : str
        Google Drive file ID
    download_path : str
        Path where to download the zip file
    extract_path : str
        Path where to extract the contents
        
    Returns:
    --------
    extract_path : str
        Path to extracted files
    """
    import os
    
    # Download using gdown
    os.system(f'gdown --id {file_id} -O {download_path}')
    
    # Create extraction directory
    os.system(f'mkdir -p {extract_path}')
    
    # Extract files
    os.system(f'unzip {download_path} -d {extract_path}')
    
    # List contents
    os.system(f'ls {extract_path}')
    
    return extract_path


def setup_plotting_style():
    """
    Set up matplotlib plotting style used in the original notebook.
    """
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)


def print_data_info(df, name="DataFrame"):
    """
    Print comprehensive information about a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    name : str
        Name to display for the dataframe
    """
    print(f"\n=== {name} Information ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types and missing values:")
    print(df.info())
    print(f"\nDescriptive statistics:")
    print(df.describe())


def check_data_quality(df):
    """
    Check data quality and report any issues.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
        
    Returns:
    --------
    issues : list
        List of data quality issues found
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Duplicate rows found: {duplicates}")
    
    # Check for negative values in columns that shouldn't have them
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Calories']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Negative values in {col}: {negative_count}")
    
    if not issues:
        print("✅ No data quality issues found!")
    else:
        print("⚠️ Data quality issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return issues