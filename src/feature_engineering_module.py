"""
Feature Engineering Functions for Calorie Prediction

This module contains all the feature engineering functions used in the
Kaggle Playground Series S5E5 competition.

Author: Evelyn
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def encode_categorical_features(df, fit_encoder=None):
    """
    Encode categorical features (Sex) using OrdinalEncoder.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the 'Sex' column
    fit_encoder : OrdinalEncoder, optional
        Pre-fitted encoder. If None, creates and fits new encoder.
        
    Returns:
    --------
    df_encoded : pandas.DataFrame
        DataFrame with encoded categorical features
    encoder : OrdinalEncoder
        The fitted encoder object
    """
    df_encoded = df.copy()
    
    if fit_encoder is None:
        encoder = OrdinalEncoder()
        df_encoded["Sex"] = encoder.fit_transform(df[["Sex"]])
    else:
        encoder = fit_encoder
        df_encoded["Sex"] = encoder.transform(df[["Sex"]])
    
    return df_encoded, encoder


def create_bmi_feature(df):
    """
    Create BMI (Body Mass Index) feature from height and weight.
    
    BMI = weight (kg) / height (m)²
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Weight' and 'Height' columns
        
    Returns:
    --------
    df_with_bmi : pandas.DataFrame
        DataFrame with added 'BMI' column
    """
    df_with_bmi = df.copy()
    # Convert height from cm to meters and calculate BMI
    df_with_bmi["BMI"] = df_with_bmi["Weight"] / (df_with_bmi["Height"] / 100)**2
    return df_with_bmi


def create_heart_rate_features(df):
    """
    Create heart rate based features.
    
    Features created:
    - Heart_Rate_Duration_Ratio: Heart rate per minute of exercise
    - Heart_Rate_Duration_Squared: Non-linear interaction term
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Heart_Rate' and 'Duration' columns
        
    Returns:
    --------
    df_with_hr_features : pandas.DataFrame
        DataFrame with added heart rate features
    """
    df_with_hr_features = df.copy()
    
    # Heart rate intensity (HR per minute)
    df_with_hr_features['Heart_Rate_Duration_Ratio'] = (
        df_with_hr_features['Heart_Rate'] / df_with_hr_features['Duration']
    )
    
    # Non-linear interaction term
    df_with_hr_features['Heart_Rate_Duration_Squared'] = (
        (df_with_hr_features['Heart_Rate'] * df_with_hr_features['Duration'])**2
    )
    
    return df_with_hr_features


def create_body_temp_features(df):
    """
    Create body temperature based features.
    
    Features created:
    - Body_Temp_Heart_Rate_Ratio: Body temp relative to heart rate
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Body_Temp' and 'Heart_Rate' columns
        
    Returns:
    --------
    df_with_temp_features : pandas.DataFrame
        DataFrame with added temperature features
    """
    df_with_temp_features = df.copy()
    
    # Body temperature relative to heart rate
    df_with_temp_features['Body_Temp_Heart_Rate_Ratio'] = (
        df_with_temp_features['Body_Temp'] / df_with_temp_features['Heart_Rate']
    )
    
    return df_with_temp_features


def engineer_all_features(df, fit_encoder=None):
    """
    Apply all feature engineering steps to the DataFrame.
    
    This is the main function that applies all feature engineering
    transformations in the correct order.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw DataFrame with original features
    fit_encoder : OrdinalEncoder, optional
        Pre-fitted encoder for categorical features
        
    Returns:
    --------
    df_engineered : pandas.DataFrame
        DataFrame with all engineered features
    encoder : OrdinalEncoder
        The fitted encoder object
    """
    # 1. Encode categorical features
    df_engineered, encoder = encode_categorical_features(df, fit_encoder)
    
    # 2. Create BMI feature
    df_engineered = create_bmi_feature(df_engineered)
    
    # 3. Create heart rate features
    df_engineered = create_heart_rate_features(df_engineered)
    
    # 4. Create body temperature features
    df_engineered = create_body_temp_features(df_engineered)
    
    return df_engineered, encoder


def get_feature_names():
    """
    Get list of all feature names after engineering.
    
    Returns:
    --------
    feature_names : list
        List of all feature column names
    """
    base_features = ['Age', 'Sex', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    engineered_features = [
        'BMI', 
        'Heart_Rate_Duration_Ratio', 
        'Heart_Rate_Duration_Squared',
        'Body_Temp_Heart_Rate_Ratio'
    ]
    
    return base_features + engineered_features


def prepare_features_for_modeling(df, target_col='Calories', fit_encoder=None):
    """
    Prepare features for modeling by applying all engineering and splitting X/y.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw DataFrame
    target_col : str
        Name of target column (default: 'Calories')
    fit_encoder : OrdinalEncoder, optional
        Pre-fitted encoder
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series or None
        Target vector (None if target_col not in df)
    encoder : OrdinalEncoder
        Fitted encoder
    """
    # Apply feature engineering
    df_engineered, encoder = engineer_all_features(df, fit_encoder)
    
    # Split features and target
    if target_col in df_engineered.columns:
        X = df_engineered.drop(target_col, axis=1)
        y = df_engineered[target_col].copy()
    else:
        X = df_engineered
        y = None
    
    return X, y, encoder