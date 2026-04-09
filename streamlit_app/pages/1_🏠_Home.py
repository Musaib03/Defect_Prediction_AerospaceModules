"""
Home Page - Project Overview
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, CUSTOM_CSS, MODEL_METRICS, DATASET_CONFIG, FEATURE_CONFIG

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏠 Project Overview</h1>', unsafe_allow_html=True)

st.markdown("---")

# Project introduction
st.markdown("""
## Aerospace Software Defect Prediction Using Machine Learning

This project demonstrates advanced machine learning techniques for predicting software defects in aerospace modules
using NASA datasets from Kaggle. The system achieves **99.70% accuracy** while using only **10 carefully selected
features** out of 21 available software complexity metrics.

""")

# Problem statement
st.markdown("""
### 🎯 Problem Statement

Software defects in aerospace systems can have critical consequences. Traditional defect prediction approaches
often struggle with:
- **Class imbalance**: Very few defective modules compared to non-defective ones
- **Feature redundancy**: Too many correlated metrics leading to overfitting
- **Model complexity**: Difficult to interpret and computationally expensive

This project addresses these challenges through:
1. **SMOTEENN** for intelligent class balancing
2. **Multi-method feature selection** for optimal feature subset
3. **Ensemble and neural network models** for high accuracy
""")

st.markdown("---")

# Key achievements
st.markdown("### 🏆 Key Achievements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">

    #### Model Performance
    - **99.70% Accuracy** with Neural Network
    - **99.40% F1-Score** on test set
    - **Perfect AUC-ROC** (1.0) for top model
    - **3 Models** exceeding 98% accuracy

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">

    #### Feature Engineering
    - Reduced from **21 to 10 features** (52% reduction)
    - Used **6 feature selection methods**
    - Focused on code complexity metrics
    - Eliminated redundant correlations

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">

    #### Class Imbalance Handling
    - Original: **90.2% non-defective**, 9.8% defective
    - SMOTEENN: **41.1% non-defective**, 58.9% defective
    - Improved from **9:1 to 0.7:1** ratio
    - **723 samples** after resampling (from 498)

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">

    #### Prediction Results
    - **109 defects** detected in 419 test samples
    - **26% defect rate** in predictions
    - High confidence predictions
    - Validated on multiple datasets

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Methodology
st.markdown("### 🔬 Methodology Overview")

# Create tabs for different methodological aspects
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "⚖️ Preprocessing", "🔍 Feature Selection", "🤖 Models"])

with tab1:
    st.markdown(f"""
    #### Datasets Used

    **1. Without-Label Dataset (Primary)**
    - **Training**: {DATASET_CONFIG['train_samples']:,} samples
    - **Testing**: {DATASET_CONFIG['test_samples']:,} samples
    - **Defect Rate**: {DATASET_CONFIG['defect_rate_train']:.0%} (training)
    - **Source**: NASA aerospace module metrics from Kaggle

    **2. CM1 Labeled Dataset (Validation)**
    - **Total**: 498 samples
    - **Defect Rate**: 9.84%
    - **Purpose**: Cross-validation and comparison
    - **Source**: NASA CM1 aerospace project

    **Features**: 21 software complexity metrics including:
    - Halstead metrics (volume, effort, error estimates)
    - Lines of code (LOC) metrics
    - Cyclomatic complexity measures
    - Operator/operand counts
    """)

with tab2:
    st.markdown("""
    #### Data Preprocessing Pipeline

    1. **Data Cleaning**
       - Handled missing values
       - Removed duplicate entries
       - Validated data ranges

    2. **Standardization**
       - Z-score normalization (StandardScaler)
       - Mean = 0, Std = 1
       - Improved model convergence

    3. **Class Imbalance Handling (SMOTEENN)**
       - SMOTE: Synthetic minority oversampling
       - ENN: Edited nearest neighbors cleaning
       - Result: Balanced dataset for fair training

    4. **Train-Test Split**
       - Stratified splitting
       - 80-20 split for validation
       - Preserved class distribution
    """)

with tab3:
    st.markdown(f"""
    #### Feature Selection Strategy

    **Goal**: Select {FEATURE_CONFIG['selected_features']} optimal features from {FEATURE_CONFIG['total_features']} available

    **Methods Used** (Ensemble Approach):

    1. **ANOVA F-value** - Linear dependency with target
    2. **Mutual Information** - Non-linear relationships
    3. **Random Forest Importance** - Tree-based ranking
    4. **Extra Trees Importance** - Randomized trees
    5. **RFE** - Recursive feature elimination
    6. **Autoencoder** - Neural network-based importance

    **Selection Process**:
    - Each method ranks features independently
    - Aggregate rankings using weighted average
    - Select top {FEATURE_CONFIG['selected_features']} features with highest consensus
    - Validate through correlation analysis

    **Top Selected Features**:
    """)

    for i, feature in enumerate(FEATURE_CONFIG['top_features'], 1):
        st.markdown(f"   {i}. `{feature}`")

with tab4:
    st.markdown("""
    #### Machine Learning Models

    **Models Evaluated** (10 total):
    - Neural Networks (MLP)
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - AdaBoost
    - Logistic Regression
    - Support Vector Machines
    - K-Nearest Neighbors
    - Decision Trees
    - Naive Bayes

    **Training Strategy**:
    - 5-fold cross-validation
    - Hyperparameter tuning
    - Performance metrics: Accuracy, F1, Precision, Recall, AUC-ROC

    **Top 3 Models**:
    """)

    for idx, (model, metrics) in enumerate(list(MODEL_METRICS.items())[:3], 1):
        st.markdown(f"""
        **{idx}. {model}**
        - Accuracy: {metrics['accuracy']:.2%}
        - F1-Score: {metrics['f1_score']:.2%}
        - AUC-ROC: {metrics['auc_roc']:.4f}
        """)

st.markdown("---")

# Project workflow
st.markdown("### 🔄 Project Workflow")

st.markdown("""
```
┌─────────────────┐
│  Raw Data       │
│  (NASA Kaggle)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  & SMOTEENN     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │
│ Selection       │
│ (21 → 10)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ (10 models,     │
│  5-fold CV)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Selection │
│ (Top 3: NN, RF, │
│  GB)            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluation &    │
│ Prediction      │
│ (109 defects)   │
└─────────────────┘
```
""")

st.markdown("---")

# Results summary
st.markdown("### 📈 Results Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Model",
        "Neural Network",
        help="2 hidden layers (100, 50 neurons)"
    )
    st.metric(
        "Accuracy",
        "99.70%",
        delta="+0.30% vs RF"
    )

with col2:
    st.metric(
        "Features Used",
        "10 / 21",
        delta="-52% reduction",
        delta_color="inverse"
    )
    st.metric(
        "Training Time",
        "< 1 min",
        help="On standard CPU"
    )

with col3:
    st.metric(
        "Defects Found",
        "109 / 419",
        help="Test set predictions"
    )
    st.metric(
        "F1-Score",
        "99.40%",
        help="Balanced precision and recall"
    )

st.markdown("---")

# Next steps
st.markdown("""
### 🚀 Explore the Dashboard

Use the sidebar to navigate through different sections:

- **Data Preprocessing** ⚖️ - See how SMOTEENN transformed the imbalanced dataset
- **Feature Selection** 🔍 - Discover how 10 optimal features were selected
- **Model Performance** 📊 - Compare top performing models interactively
- **Defect Analysis** 🐛 - Explore detected defects and their characteristics
- **Live Prediction** 🚀 - Try predicting defects with custom feature values
- **CM1 Dataset** 📁 - Analyze results from the labeled dataset

""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><em>This dashboard provides an interactive exploration of machine learning techniques
    for aerospace software defect prediction.</em></p>
</div>
""", unsafe_allow_html=True)
