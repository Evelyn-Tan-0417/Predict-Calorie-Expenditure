"""
Visualization Functions for Calorie Prediction Project

This module contains all plotting and visualization functions
used in the data exploration and model evaluation.

Author: Evelyn
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """
    Set up the plotting style used throughout the project.
    
    This replicates the plotting configuration from the original notebook.
    """
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)


def plot_data_histograms(df, title="Feature Distributions", figsize=(20, 15)):
    """
    Create histogram plots for all numeric features.
    
    This replicates the histogram plotting from the original notebook.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features to plot
    title : str
        Title for the overall plot
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    df.hist(bins=50, figsize=figsize)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, title="Feature Correlation Matrix", figsize=(10, 8)):
    """
    Create correlation heatmap for numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    # Calculate correlation matrix for numeric columns only
    corr_matrix = df.corr(numeric_only=True)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.3f')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def plot_target_correlations(df, target_col='Calories', figsize=(8, 6)):
    """
    Plot correlations with the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and target
    target_col : str
        Name of target column
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    # Calculate correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    
    # Get correlations with target
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    print("Correlations with target (sorted by strength):")
    print(target_corr)
    
    # Plot correlations (exclude self-correlation)
    plt.figure(figsize=figsize)
    target_corr[:-1].plot(kind='barh')  # Exclude self-correlation
    plt.title(f'Feature Correlations with {target_col}')
    plt.xlabel('Correlation Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return target_corr


def plot_feature_relationships(df, target_col='Calories', features=None, figsize=(15, 5)):
    """
    Plot relationships between key features and target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and target
    target_col : str
        Name of target column
    features : list, optional
        List of features to plot. If None, uses top correlated features.
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    if features is None:
        # Use features from original notebook analysis
        features = ['Heart_Rate', 'Weight', 'Duration']
    
    n_features = len(features)
    plt.figure(figsize=(figsize[0], figsize[1]))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(1, n_features, i)
        plt.scatter(df[feature], df[target_col], alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f'{feature} vs {target_col}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_scatter(y_true, y_pred, title="Actual vs Predicted", figsize=(10, 6)):
    """
    Create scatter plot of actual vs predicted values.
    
    This replicates the prediction visualization from the evaluation function.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Calories")
    plt.ylabel("Predicted Calories")
    plt.title(title)
    
    # Add diagonal line for perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, figsize=(12, 4)):
    """
    Plot residual analysis for model predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importance information")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe for easy sorting
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    top_features = feature_imp_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return feature_imp_df


def plot_model_comparison(results_df, metric='Test_RMSE', figsize=(10, 6)):
    """
    Plot comparison of different models.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with model comparison results
    metric : str
        Metric to plot
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    plt.figure(figsize=figsize)
    plt.barh(results_df['Model'], results_df[metric])
    plt.xlabel(metric)
    plt.title(f'Model Comparison - {metric}')
    plt.gca().invert_yaxis()  # Best model at top
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(results_df[metric]):
        plt.text(v, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_scores, val_scores, train_sizes, title="Learning Curve", figsize=(10, 6)):
    """
    Plot learning curves for model performance.
    
    Parameters:
    -----------
    train_scores : array-like
        Training scores
    val_scores : array-like
        Validation scores
    train_sizes : array-like
        Training set sizes
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()