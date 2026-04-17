"""
Configuration file for Streamlit UI
Contains paths, model parameters, and styling settings
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR

# Data file paths
PATHS = {
    # Without label dataset files
    'train_features': DATA_DIR / 'without_label' / 'Train_Features.csv',
    'test_features': DATA_DIR / 'without_label' / 'Test_Features.csv',
    'train_features_selected': DATA_DIR / 'without_label' / 'train_features_selected.csv',
    'train_recommended_features': DATA_DIR / 'without_label' / 'train_recommended_features.txt',
    'train_importance_report': DATA_DIR / 'without_label' / 'train_features_importance_report.csv',
    'test_predictions': DATA_DIR / 'without_label' / 'test_defect_predictions.csv',
    'defect_summary': DATA_DIR / 'without_label' / 'defect_prediction_summary.txt',
    'preprocessed_data': DATA_DIR / 'preprocessed_data.csv',

    # CM1 labeled dataset files
    'cm1_data': DATA_DIR / 'C_withlabel' / 'cm1.csv',
    'cm1_preprocessed': DATA_DIR / 'C_withlabel' / 'cm1_preprocessed.csv',
    'cm1_recommended_features': DATA_DIR / 'C_withlabel' / 'cm1_recommended_features.txt',
    'smoteenn_resampled': DATA_DIR / 'C_withlabel' / 'smoteenn_resampled_data.csv',
}

# Feature configuration
FEATURE_CONFIG = {
    'total_features': 21,
    'selected_features': 10,
    'top_features': [
        'HALSTEAD_LEVEL',
        'ESSENTIAL_COMPLEXITY',
        'HALSTEAD_ERROR_EST',
        'LOC_BLANK',
        'LOC_COMMENTS',
        'LOC_EXECUTABLE',
        'NUM_UNIQUE_OPERATORS',
        'HALSTEAD_EFFORT',
        'HALSTEAD_VOLUME',
        'NUM_OPERATORS'
    ]
}

# Model performance metrics (from project results)
MODEL_METRICS = {
    'Neural Network': {
        'accuracy': 0.9970,
        'f1_score': 0.9940,
        'precision': 0.9940,
        'recall': 0.9940,
        'auc_roc': 1.0,
        'architecture': '2 Hidden Layers (100, 50 neurons)',
        'test_defects': 109
    },
    'Random Forest': {
        'accuracy': 0.9940,
        'f1_score': 0.9880,
        'precision': 0.9880,
        'recall': 0.9880,
        'auc_roc': 0.9998,
    },
    'Gradient Boosting': {
        'accuracy': 0.9911,
        'f1_score': 0.9820,
        'precision': 0.9820,
        'recall': 0.9820,
        'auc_roc': 0.9995,
    }
}

# SMOTEENN configuration
SMOTEENN_CONFIG = {
    'before': {
        'total_samples': 498,
        'non_defective': 449,
        'defective': 49,
        'non_defective_pct': 90.2,
        'defective_pct': 9.8
    },
    'after': {
        'total_samples': 723,
        'non_defective': 297,
        'defective': 426,
        'non_defective_pct': 41.1,
        'defective_pct': 58.9
    }
}

# Dataset configuration
DATASET_CONFIG = {
    'train_samples': 1676,
    'test_samples': 419,
    'non_defective_train': 1257,
    'defective_train': 419,
    'defect_rate_train': 0.25,
    'predicted_defects_test': 109,
    'predicted_defect_rate_test': 0.26
}

# Styling configuration
COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'success': '#2ca02c',  # Green
    'danger': '#d62728',  # Red
    'background': '#f0f2f6',
    'defective': '#ff6b6b',
    'non_defective': '#4ecdc4',
    'aerospace_blue': '#003f5c',
    'aerospace_gray': '#58508d',
    'accent': '#ffa600'
}

# Page configuration
PAGE_CONFIG = {
    'page_title': 'Aerospace Defect Prediction',
    'page_icon': '✈️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #003f5c;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #003f5c 0%, #58508d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #003f5c;
        margin: 0.5rem 0;
    }

    .success-box {
        background-color: #000000;
        border-left: 5px solid #2ca02c;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000000;
    }

    .info-box * {
        color: #000000;
    }

    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffa600;
        color: #000000;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stButton>button {
        background-color: #003f5c;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #58508d;
    }
</style>
"""
