"""
Calorie Prediction Project Source Package

This package contains all the source code modules for the
Kaggle Playground Series S5E5 calorie expenditure prediction project.

Author: Evelyn
Date: 2025

Modules:
--------
- data_preprocessing: Functions for loading and preparing data
- feature_engineering: Feature creation and transformation functions  
- models: Model training, evaluation, and comparison functions
- utils: Utility functions for evaluation, file I/O, and helpers
- visualization: Plotting and visualization functions

Usage:
------
from src.data_preprocessing import load_competition_data
from src.feature_engineering import engineer_all_features
from src.models import train_random_forest
from src.utils import evaluation
from src.visualization import plot_correlation_matrix
"""

# Import key functions for easy access
from .data_preprocessing import (
    load_competition_data,
    encode_categorical_variables,
    create_train_test_split,
    prepare_features_and_target
)

from .feature_engineering import (
    engineer_all_features,
    create_bmi_feature,
    create_heart_rate_features,
    prepare_features_for_modeling
)

from .models import (
    train_random_forest,
    train_linear_regression,
    train_ensemble_model,
    evaluation as model_evaluation,
    cross_validate_model
)

from .utils import (
    rmsle,
    evaluation,
    save_submission_file,
    setup_plotting_style
)

from .visualization import (
    plot_correlation_matrix,
    plot_target_correlations,
    plot_prediction_scatter,
    plot_feature_importance
)

__version__ = "1.0.0"
__author__ = "Evelyn"

# List of all available modules
__all__ = [
    'data_preprocessing',
    'feature_engineering', 
    'models',
    'utils',
    'visualization'
]