"""
Model Training and Evaluation Functions

This module contains all the machine learning models and evaluation
functions used in the Calorie Prediction competition.

Author: Evelyn
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error.
    
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


def evaluation(model, X_test, y_test, show_plot=True):
    """
    Evaluate model performance with RMSE, RMSLE and visualization.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    show_plot : bool
        Whether to show scatter plot of predictions
        
    Returns:
    --------
    metrics : dict
        Dictionary with RMSE and RMSLE scores
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are non-negative (calories can't be negative)
    y_pred = np.abs(y_pred)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmsle_val = rmsle(y_test, y_pred)
    
    print(f"RMSE on the test set: {rmse:.5f}")
    print(f"RMSLE on the test set: {rmsle_val:.5f}")
    
    # Visualization
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Calories")
        plt.ylabel("Predicted Calories")
        plt.title("Actual vs. Predicted Calories (Test Set)")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.show()
    
    return {"RMSE": rmse, "RMSLE": rmsle_val}


def train_linear_regression(X_train, y_train, use_scaling=False):
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    use_scaling : bool
        Whether to use StandardScaler in pipeline
        
    Returns:
    --------
    model : sklearn estimator
        Trained model
    """
    if use_scaling:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree Regressor.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    model : DecisionTreeRegressor
        Trained model
    """
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    model : RandomForestRegressor
        Trained model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_with_gridsearch(X_train, y_train, cv=3, random_state=42):
    """
    Train Random Forest with Grid Search hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    cv : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    best_model : RandomForestRegressor
        Best model from grid search
    grid_search : GridSearchCV
        Grid search object with results
    """
    # Create pipeline
    pipeline = Pipeline([
        ("random_forest", RandomForestRegressor(random_state=random_state)),
    ])
    
    # Define parameter grid
    param_grid = [{
        'random_forest__max_features': [2, 3, 4, 5, 6]
    }]
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.5f}")
    
    return grid_search.best_estimator_, grid_search


def train_ensemble_model(X_train, y_train, random_state=42):
    """
    Train an ensemble model using VotingRegressor.
    
    Combines Ridge, Random Forest, and SVR models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    model : VotingRegressor
        Trained ensemble model
    """
    # Individual regressors
    reg1 = Ridge(alpha=1.0)
    reg2 = RandomForestRegressor(n_estimators=100, random_state=random_state)
    reg3 = SVR(kernel='rbf', C=100)
    
    # Create ensemble
    model = VotingRegressor(
        estimators=[('ridge', reg1), ('rf', reg2), ('svr', reg3)]
    )
    
    model.fit(X_train, y_train)
    return model


def cross_validate_model(model, X, y, cv=10, scoring='neg_root_mean_squared_error'):
    """
    Perform cross-validation on a model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to cross-validate
    X : array-like
        Features
    y : array-like
        Targets
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
        
    Returns:
    --------
    cv_scores : array
        Cross-validation scores
    stats : dict
        Statistics of CV scores
    """
    cv_scores = -cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    stats = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'min': cv_scores.min(),
        'max': cv_scores.max()
    }
    
    print(f"Cross-validation {scoring} scores:")
    print(f"Mean: {stats['mean']:.5f} (+/- {stats['std']*2:.5f})")
    print(f"Range: [{stats['min']:.5f}, {stats['max']:.5f}]")
    
    return cv_scores, stats


def save_submission_file(model, X_test, test_ids, filename='submission.csv'):
    """
    Generate submission file for Kaggle competition.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    test_ids : array-like
        Test set IDs
    filename : str
        Output filename
        
    Returns:
    --------
    submission_df : pandas.DataFrame
        Submission dataframe
    """
    # Make predictions
    test_predictions = model.predict(X_test)
    
    # Ensure predictions are non-negative and integer
    test_predictions = np.abs(test_predictions).round().astype(int)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Calories': test_predictions
    })
    
    # Save to file
    submission_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    print(f"Submission shape: {submission_df.shape}")
    print("\nFirst few rows:")
    print(submission_df.head())
    
    return submission_df


def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Compare multiple models and return performance summary.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: trained_model}
    X_train : array-like
        Training features  
    y_train : array-like
        Training targets
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Comparison results
    """
    results = []
    
    for name, model in models_dict.items():
        # Test set performance
        test_metrics = evaluation(model, X_test, y_test, show_plot=False)
        
        # Cross-validation performance
        cv_scores, cv_stats = cross_validate_model(model, X_train, y_train, cv=5)
        
        results.append({
            'Model': name,
            'Test_RMSE': test_metrics['RMSE'],
            'Test_RMSLE': test_metrics['RMSLE'],
            'CV_RMSE_Mean': cv_stats['mean'],
            'CV_RMSE_Std': cv_stats['std']
        })
    
    results_df = pd.DataFrame(results).sort_values('Test_RMSE')
    return results_df