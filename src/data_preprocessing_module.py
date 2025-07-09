"""
Data Preprocessing Functions

This module contains functions for loading, cleaning, and preparing
the competition data for modeling.

Author: Evelyn
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_competition_data(data_dir):
    """
    Load training and test data from the competition files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the CSV files
        
    Returns:
    --------
    train_df : pandas.DataFrame
        Training data
    test_df : pandas.DataFrame
        Test data  
    sample_submission : pandas.DataFrame
        Sample submission format
    """
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    sample_submission = pd.read_csv(f'{data_dir}/sample_submission.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")
    
    return train_df, test_df, sample_submission


def encode_categorical_variables(train_df, test_df=None):
    """
    Encode categorical variables (Sex) for both training and test sets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    test_df : pandas.DataFrame, optional
        Test data
        
    Returns:
    --------
    train_encoded : pandas.DataFrame
        Training data with encoded categorical variables
    test_encoded : pandas.DataFrame or None
        Test data with encoded categorical variables (if provided)
    encoder : OrdinalEncoder
        Fitted encoder for future use
    """
    # Initialize encoder
    encoder = OrdinalEncoder()
    
    # Encode training data
    train_encoded = train_df.copy()
    sex_cat_train = train_df[["Sex"]]
    sex_encoded_train = encoder.fit_transform(sex_cat_train)
    train_encoded["Sex"] = sex_encoded_train
    
    print("Training data - Sex encoding:")
    print(f"Original values: {train_df['Sex'].unique()}")
    print(f"Encoded values: {np.unique(sex_encoded_train)}")
    print(f"First 10 encoded values: {sex_encoded_train[:10].flatten()}")
    
    # Encode test data if provided
    test_encoded = None
    if test_df is not None:
        test_encoded = test_df.copy()
        sex_cat_test = test_df[["Sex"]]
        sex_encoded_test = encoder.transform(sex_cat_test)
        test_encoded["Sex"] = sex_encoded_test
        
        print("\nTest data - Sex encoding:")
        print(f"First 10 encoded values: {sex_encoded_test[:10].flatten()}")
    
    return train_encoded, test_encoded, encoder


def create_train_test_split(df, target_col='Calories', test_size=0.2, random_state=42):
    """
    Create train/test split from the training data.
    
    This replicates the train_test_split used in the original notebook.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Training dataframe
    target_col : str
        Name of target column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    train_set : pandas.DataFrame
        Training subset
    test_set : pandas.DataFrame
        Testing subset
    """
    # Set random seed for reproducibility (like in original notebook)
    np.random.seed(random_state)
    
    train_set, test_set = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Train set shape: {train_set.shape}")
    print(f"Test set shape: {test_set.shape}")
    
    return train_set, test_set


def prepare_features_and_target(train_set, test_set, target_col='Calories'):
    """
    Separate features and target variables from train/test sets.
    
    Parameters:
    -----------
    train_set : pandas.DataFrame
        Training data
    test_set : pandas.DataFrame
        Test data
    target_col : str
        Name of target column
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training targets
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test targets
    """
    # Split training set
    X_train = train_set.drop(target_col, axis=1)
    y_train = train_set[target_col].copy()
    
    # Split test set
    X_test = test_set.drop(target_col, axis=1)
    y_test = test_set[target_col].copy()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def align_test_features(test_df, training_columns):
    """
    Align test dataframe features with training features.
    
    This ensures test data has same columns as training data,
    which is important for the submission process.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test dataframe
    training_columns : list or Index
        Column names from training data
        
    Returns:
    --------
    test_aligned : pandas.DataFrame
        Test data with aligned features
    """
    # Remove target column if it exists in training_columns
    feature_columns = [col for col in training_columns if col != 'Calories']
    
    # Select only the features that exist in training
    test_aligned = test_df[feature_columns]
    
    print(f"Test data aligned to {len(feature_columns)} features")
    print(f"Feature columns: {feature_columns}")
    
    return test_aligned


def get_basic_data_stats(df, name="Data"):
    """
    Print basic statistics about the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    name : str
        Name for the dataset
    """
    print(f"\n=== {name} Statistics ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"Categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            print(f"  {col} values: {df[col].value_counts().to_dict()}")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"Numeric columns: {list(numeric_cols)}")
        print(f"Numeric summary:")
        print(df[numeric_cols].describe())


def validate_data_integrity(train_df, test_df):
    """
    Validate that train and test data have compatible structure.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    test_df : pandas.DataFrame
        Test data
        
    Returns:
    --------
    is_valid : bool
        Whether data passes validation
    issues : list
        List of validation issues found
    """
    issues = []
    
    # Check that test data has all required features (except target)
    train_features = set(train_df.columns) - {'Calories'}
    test_features = set(test_df.columns)
    
    missing_features = train_features - test_features
    if missing_features:
        issues.append(f"Test data missing features: {missing_features}")
    
    extra_features = test_features - train_features - {'id'}  # 'id' is expected in test
    if extra_features:
        issues.append(f"Test data has unexpected features: {extra_features}")
    
    # Check data types match
    common_features = train_features & test_features
    for feature in common_features:
        if train_df[feature].dtype != test_df[feature].dtype:
            issues.append(f"Data type mismatch for {feature}: train={train_df[feature].dtype}, test={test_df[feature].dtype}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("✅ Data validation passed!")
    else:
        print("❌ Data validation failed:")
        for issue in issues:
            print(f"  - {issue}")
    
    return is_valid, issues