"""
Data Loading Utilities
Handles loading and preprocessing of all project data files
"""

import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS, FEATURE_CONFIG


@st.cache_data
def load_train_features():
    """Load training features dataset"""
    try:
        df = pd.read_csv(PATHS['train_features'])
        return df
    except Exception as e:
        st.error(f"Error loading train features: {e}")
        return None


@st.cache_data
def load_test_features():
    """Load test features dataset"""
    try:
        df = pd.read_csv(PATHS['test_features'])
        return df
    except Exception as e:
        st.error(f"Error loading test features: {e}")
        return None


@st.cache_data
def load_selected_features():
    """Load the 10 selected features with statistics"""
    try:
        df = pd.read_csv(PATHS['train_features_selected'])
        return df
    except Exception as e:
        st.error(f"Error loading selected features: {e}")
        return None


@st.cache_data
def load_recommended_features():
    """Load recommended features list with importance scores"""
    try:
        features_dict = {}
        with open(PATHS['train_recommended_features'], 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    feature, score = line.split(':')
                    features_dict[feature.strip()] = float(score.strip())
        return features_dict
    except Exception as e:
        st.error(f"Error loading recommended features: {e}")
        return {}


@st.cache_data
def load_importance_report():
    """Load detailed feature importance report"""
    try:
        df = pd.read_csv(PATHS['train_importance_report'])
        return df
    except Exception as e:
        st.error(f"Error loading importance report: {e}")
        return None


@st.cache_data
def load_test_predictions():
    """Load test set predictions"""
    try:
        df = pd.read_csv(PATHS['test_predictions'])
        return df
    except Exception as e:
        st.error(f"Error loading test predictions: {e}")
        return None


@st.cache_data
def load_defect_summary():
    """Load defect prediction summary text"""
    try:
        with open(PATHS['defect_summary'], 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading defect summary: {e}")
        return ""


@st.cache_data
def load_preprocessed_data():
    """Load preprocessed (standardized) data"""
    try:
        df = pd.read_csv(PATHS['preprocessed_data'])
        return df
    except Exception as e:
        st.error(f"Error loading preprocessed data: {e}")
        return None


@st.cache_data
def load_cm1_data():
    """Load CM1 original dataset"""
    try:
        df = pd.read_csv(PATHS['cm1_data'])
        return df
    except Exception as e:
        st.error(f"Error loading CM1 data: {e}")
        return None


@st.cache_data
def load_cm1_preprocessed():
    """Load CM1 preprocessed dataset"""
    try:
        df = pd.read_csv(PATHS['cm1_preprocessed'])
        return df
    except Exception as e:
        st.error(f"Error loading CM1 preprocessed data: {e}")
        return None


@st.cache_data
def load_smoteenn_data():
    """Load SMOTEENN resampled data"""
    try:
        df = pd.read_csv(PATHS['smoteenn_resampled'])
        return df
    except Exception as e:
        st.error(f"Error loading SMOTEENN data: {e}")
        return None


@st.cache_data
def load_cm1_recommended_features():
    """Load CM1 recommended features"""
    try:
        features_dict = {}
        with open(PATHS['cm1_recommended_features'], 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    feature, score = line.split(':')
                    features_dict[feature.strip()] = float(score.strip())
        return features_dict
    except Exception as e:
        st.error(f"Error loading CM1 recommended features: {e}")
        return {}


def get_feature_stats(df, feature_list=None):
    """
    Calculate statistics for features

    Args:
        df: DataFrame with features
        feature_list: List of features to analyze (default: all numeric columns)

    Returns:
        DataFrame with statistics
    """
    if feature_list is None:
        feature_list = df.select_dtypes(include=['number']).columns.tolist()

    stats = df[feature_list].describe().T
    stats['feature'] = stats.index
    return stats[['feature', 'mean', 'std', 'min', 'max']]


def get_defective_samples(predictions_df, threshold=0.5):
    """
    Get defective samples based on prediction threshold

    Args:
        predictions_df: DataFrame with predictions
        threshold: Probability threshold for defect classification

    Returns:
        DataFrame with defective samples
    """
    if 'defect_probability' in predictions_df.columns:
        return predictions_df[predictions_df['defect_probability'] >= threshold]
    elif 'prediction' in predictions_df.columns:
        return predictions_df[predictions_df['prediction'] == 1]
    else:
        return predictions_df
