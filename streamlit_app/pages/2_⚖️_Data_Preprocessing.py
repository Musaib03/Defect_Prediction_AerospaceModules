"""
Data Preprocessing Page - SMOTEENN Class Imbalance Handling
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, CUSTOM_CSS, SMOTEENN_CONFIG, DATASET_CONFIG
from utils.smoteenn_handler import (
    create_smoteenn_comparison,
    create_sample_flow_diagram,
    get_smoteenn_explanation
)
from utils.data_loader import load_cm1_data, load_smoteenn_data, load_preprocessed_data

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ Data Preprocessing & Class Imbalance</h1>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.markdown("""
## Handling Class Imbalance with SMOTEENN

Class imbalance is a critical challenge in defect prediction. Our dataset had a severe imbalance with
**90.2% non-defective** and only **9.8% defective** modules. Without addressing this, models would be biased
toward predicting "no defect" and miss critical software issues.

We used **SMOTEENN** - a combination of SMOTE (oversampling) and ENN (cleaning) to create a balanced dataset
for effective model training.
""")

st.markdown("---")

# SMOTEENN Explanation
explanation = get_smoteenn_explanation()

with st.expander("📚 What is SMOTEENN? (Click to expand)", expanded=True):
    st.markdown(f"### {explanation['title']}")
    st.markdown(explanation['description'])

    st.markdown("#### ✅ Key Benefits")
    for benefit in explanation['benefits']:
        st.markdown(f"- {benefit}")

    st.markdown("#### 📊 Impact Metrics")
    cols = st.columns(4)
    for idx, (metric, value) in enumerate(explanation['metrics'].items()):
        with cols[idx]:
            st.metric(metric, value)

st.markdown("---")

# Interactive visualization
st.markdown("### 📊 SMOTEENN Transformation Visualization")

# Create tabs for different views
viz_tab1, viz_tab2 = st.tabs(["📈 Comprehensive Analysis", "🔄 Sample Flow Diagram"])

with viz_tab1:
    st.markdown("""
    This comprehensive view shows the transformation in multiple aspects:
    - Class distribution before and after SMOTEENN
    - Sample count changes
    - Imbalance ratio improvement
    """)

    fig_comparison = create_smoteenn_comparison()
    st.plotly_chart(fig_comparison, use_container_width=True)

with viz_tab2:
    st.markdown("""
    This Sankey diagram illustrates how samples flow through the SMOTEENN process:
    - Original dataset splits into defective and non-defective
    - SMOTE creates synthetic defective samples
    - ENN cleans noisy non-defective samples
    - Final balanced dataset emerges
    """)

    fig_flow = create_sample_flow_diagram()
    st.plotly_chart(fig_flow, use_container_width=True)

st.markdown("---")

# Detailed statistics
st.markdown("### 📈 Detailed Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Before SMOTEENN")
    st.markdown(f"""
    <div class="warning-box">

    - **Total Samples**: {SMOTEENN_CONFIG['before']['total_samples']}
    - **Non-Defective**: {SMOTEENN_CONFIG['before']['non_defective']} ({SMOTEENN_CONFIG['before']['non_defective_pct']}%)
    - **Defective**: {SMOTEENN_CONFIG['before']['defective']} ({SMOTEENN_CONFIG['before']['defective_pct']}%)
    - **Imbalance Ratio**: {SMOTEENN_CONFIG['before']['non_defective'] / SMOTEENN_CONFIG['before']['defective']:.2f}:1
    - **Problem**: Severe imbalance - models would ignore minority class

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### After SMOTEENN")
    st.markdown(f"""
    <div class="success-box">

    - **Total Samples**: {SMOTEENN_CONFIG['after']['total_samples']} (↑ {SMOTEENN_CONFIG['after']['total_samples'] - SMOTEENN_CONFIG['before']['total_samples']})
    - **Non-Defective**: {SMOTEENN_CONFIG['after']['non_defective']} ({SMOTEENN_CONFIG['after']['non_defective_pct']}%)
    - **Defective**: {SMOTEENN_CONFIG['after']['defective']} ({SMOTEENN_CONFIG['after']['defective_pct']}%)
    - **Imbalance Ratio**: {SMOTEENN_CONFIG['after']['defective'] / SMOTEENN_CONFIG['after']['non_defective']:.2f}:1
    - **Solution**: Balanced dataset - fair representation for both classes

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Interactive slider demonstration
st.markdown("### 🎚️ Interactive Class Balance Threshold")

st.markdown("""
Adjust the slider below to see how different class balance ratios would affect the dataset composition.
The green zone represents our achieved balance with SMOTEENN.
""")

balance_ratio = st.slider(
    "Minority Class Percentage",
    min_value=0.0,
    max_value=100.0,
    value=SMOTEENN_CONFIG['after']['defective_pct'],
    step=0.1,
    format="%.1f%%",
    help="Percentage of minority (defective) class in the dataset"
)

# Calculate what the sample counts would be
total_samples = SMOTEENN_CONFIG['after']['total_samples']
minority_count = int(total_samples * (balance_ratio / 100))
majority_count = total_samples - minority_count

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Minority Class (Defective)",
        f"{minority_count}",
        f"{balance_ratio:.1f}%"
    )

with col2:
    st.metric(
        "Majority Class (Non-Defective)",
        f"{majority_count}",
        f"{100 - balance_ratio:.1f}%"
    )

with col3:
    if majority_count > 0:
        ratio = minority_count / majority_count if minority_count > majority_count else majority_count / minority_count
        st.metric(
            "Imbalance Ratio",
            f"{ratio:.2f}:1",
            "Balanced" if 0.4 <= (balance_ratio / 100) <= 0.6 else "Imbalanced",
            delta_color="inverse" if 0.4 <= (balance_ratio / 100) <= 0.6 else "normal"
        )

# Feedback on selected ratio
if balance_ratio < 40:
    st.warning("⚠️ Still imbalanced - minority class underrepresented. Models may be biased toward majority class.")
elif balance_ratio > 60:
    st.warning("⚠️ Overbalanced - minority class overrepresented. May lead to overfitting on synthetic samples.")
else:
    st.success("✅ Good balance! This ratio allows models to learn patterns from both classes effectively.")

st.markdown("---")

# Additional preprocessing steps
st.markdown("### 🔧 Other Preprocessing Steps")

proc_tab1, proc_tab2, proc_tab3 = st.tabs(["🧹 Data Cleaning", "📏 Standardization", "✂️ Train-Test Split"])

with proc_tab1:
    st.markdown("""
    #### Data Cleaning Pipeline

    1. **Missing Value Handling**
       - Checked all 21 features for missing values
       - NASA datasets are generally complete - no missing values found
       - Would use median imputation for numeric features if needed

    2. **Duplicate Removal**
       - Identified and removed any duplicate module entries
       - Ensured unique sample identification

    3. **Outlier Detection**
       - Analyzed distributions for extreme values
       - Retained outliers as they may represent legitimate high-complexity modules
       - Used robust scaling methods instead of removal

    4. **Data Type Validation**
       - Verified all features are numeric
       - Checked value ranges for consistency
       - Validated target labels (0 = no defect, 1 = defect)
    """)

with proc_tab2:
    st.markdown(f"""
    #### Feature Standardization (Z-Score Normalization)

    **Method**: StandardScaler from scikit-learn

    **Formula**: `z = (x - μ) / σ`
    - `x`: Original value
    - `μ`: Feature mean
    - `σ`: Feature standard deviation

    **Why Standardization?**
    - Different features have vastly different scales
       * LOC_EXECUTABLE: 1 - 500+ lines
       * HALSTEAD_LEVEL: 0.001 - 0.1
    - Neural networks and distance-based models require scaled features
    - Improves convergence speed and model stability

    **Result**: All features transformed to have:
    - Mean = 0
    - Standard deviation = 1
    - Preserved original distributions

    **Dataset Size**: {DATASET_CONFIG['train_samples']:,} training samples standardized
    """)

    # Show example if data is available
    try:
        preprocessed = load_preprocessed_data()
        if preprocessed is not None:
            st.markdown("#### Sample Standardized Data (First 5 rows)")
            st.dataframe(preprocessed.head(), use_container_width=True)
    except:
        pass

with proc_tab3:
    st.markdown("""
    #### Stratified Train-Test Split

    **Strategy**: Stratified splitting to maintain class distribution

    **Configuration**:
    - **Training Set**: 80% of data
    - **Validation Set**: 20% of data (used during cross-validation)
    - **Test Set**: Separate held-out set (419 samples)
    - **Random State**: Fixed (42) for reproducibility

    **Why Stratified?**
    - Ensures both sets have same proportion of defective/non-defective modules
    - Critical when dealing with imbalanced data
    - Provides representative evaluation

    **Cross-Validation**:
    - 5-fold stratified cross-validation during training
    - Each fold maintains class distribution
    - Robust performance estimation

    **Final Setup**:
    - Training: ~1,340 samples (80%)
    - Validation: ~336 samples (20%)
    - Test: 419 samples (separate)
    """)

st.markdown("---")

# Impact on model performance
st.markdown("### 🎯 Impact on Model Performance")

st.markdown("""
<div class="info-box">

#### How SMOTEENN Improved Our Results

**Without Class Balancing** (Hypothetical):
- Model would predict "no defect" for most cases (>90% accuracy by always predicting majority)
- Very low recall for defective modules (would miss most defects)
- Poor F1-score due to imbalanced precision/recall
- Not useful for real-world defect detection

**With SMOTEENN**:
- ✅ Balanced representation forces model to learn defect patterns
- ✅ High recall (99.40%) - catches most defects
- ✅ High precision (99.40%) - few false alarms
- ✅ Excellent F1-score (99.40%) - balanced performance
- ✅ Real-world applicable - reliable defect detection

**Key Achievement**: 99.70% accuracy with 99.40% F1-score shows the model performs well on BOTH classes!

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Data samples preview
st.markdown("### 📋 Data Samples Preview")

with st.expander("View SMOTEENN Resampled Data Sample", expanded=False):
    try:
        smoteenn_data = load_smoteenn_data()
        if smoteenn_data is not None:
            st.markdown(f"**Total Samples**: {len(smoteenn_data)}")
            st.markdown(f"**Features**: {smoteenn_data.shape[1]}")

            # Show class distribution
            if 'defects' in smoteenn_data.columns or 'Defective' in smoteenn_data.columns:
                defect_col = 'defects' if 'defects' in smoteenn_data.columns else 'Defective'
                class_dist = smoteenn_data[defect_col].value_counts()
                st.markdown(f"**Class Distribution**:")
                st.write(class_dist)

            st.markdown("**Sample Data (First 10 rows)**:")
            st.dataframe(smoteenn_data.head(10), use_container_width=True)
        else:
            st.info("SMOTEENN data file not found. This is expected if running on a different dataset.")
    except Exception as e:
        st.info(f"Could not load SMOTEENN data: {e}")

st.markdown("---")

# Summary
st.markdown("### 📌 Summary")

st.markdown("""
**Key Takeaways**:

1. **Class imbalance** was a critical challenge (90:10 ratio)
2. **SMOTEENN** effectively balanced the dataset (41:59 ratio)
3. **Synthetic samples** created realistic defect examples
4. **Data cleaning** removed noisy borderline cases
5. **Standardization** prepared features for ML models
6. **Result**: High-quality, balanced dataset for training

**Next Step**: Explore how we selected the optimal 10 features from 21 in the **Feature Selection** page! →
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><em>SMOTEENN: Combining the power of oversampling and cleaning for optimal class balance</em></p>
</div>
""", unsafe_allow_html=True)
