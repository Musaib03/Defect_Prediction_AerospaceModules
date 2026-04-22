"""
Aerospace Software Defect Prediction - Streamlit UI
Main application entry point
"""

import streamlit as st
from config import PAGE_CONFIG, CUSTOM_CSS

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">✈️ Aerospace Software Defect Prediction</h1>', unsafe_allow_html=True)

st.markdown("---")

# Welcome section
st.markdown("""
## Welcome to the Aerospace Software Defect Prediction Dashboard

This interactive dashboard presents a comprehensive analysis of software defect prediction in aerospace modules
using machine learning techniques on NASA datasets.

### 🎯 Project Highlights

<div class="info-box">

**Key Achievements:**
- ✅ **99.70% Accuracy** achieved with Neural Network model
- ✅ **10 Features** selected from 21 using advanced feature selection (52% reduction)
- ✅ **SMOTEENN** successfully balanced highly imbalanced dataset (9:1 → 0.7:1)
- ✅ **109 Defects** detected in 419 test samples with high confidence
- ✅ **Multiple Approaches** - Both labeled (CM1) and unlabeled datasets analyzed

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Navigation guide
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Data & Preprocessing
    - **Home**: Project overview
    - **Data Preprocessing**: SMOTEENN class balancing
    - **Feature Selection**: 21 → 10 features reduction
    """)

with col2:
    st.markdown("""
    ### 🤖 Models & Analysis
    - **Model Performance**: Top 3 models comparison
    - **Defect Analysis**: Detected defects exploration
    - **Live Prediction**: Real-time defect prediction
    """)

with col3:
    st.markdown("""
    ### 📁 Additional Resources
    - **CM1 Dataset**: Labeled dataset analysis
    - Interactive visualizations
    - Downloadable results
    """)

st.markdown("---")

# Quick stats
st.markdown("### 📈 Quick Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Best Model Accuracy",
        value="99.70%",
        delta="Neural Network"
    )

with col2:
    st.metric(
        label="Features Selected",
        value="10/21",
        delta="-52% reduction"
    )

with col3:
    st.metric(
        label="Training Samples",
        value="1,676",
        delta="After preprocessing"
    )

with col4:
    st.metric(
        label="Test Defects Found",
        value="109/419",
        delta="26% defect rate"
    )

st.markdown("---")

# Getting started
st.markdown("""
### 🚀 Getting Started

Use the sidebar navigation to explore different aspects of the project:

1. **Start with Home** 🏠 - Get an overview of the project objectives and methodology
2. **Data Preprocessing** ⚖️ - Understand how SMOTEENN handled class imbalance
3. **Feature Selection** 🔍 - See how 10 optimal features were selected from 21
4. **Model Performance** 📊 - Compare the top 3 performing models
5. **Defect Analysis** 🐛 - Explore detected defects and their characteristics
6. **Live Prediction** 🚀 - Try real-time defect prediction with custom inputs
7. **CM1 Dataset** 📁 - Analyze results from the labeled NASA CM1 dataset

---

### 💡 Key Features of This Dashboard

- **Interactive Visualizations**: All charts are interactive - hover, zoom, and explore
- **Real-time Predictions**: Test the model with custom feature values
- **Comprehensive Analysis**: From data preprocessing to model evaluation
- **Downloadable Results**: Export predictions and visualizations
- **Educational**: Detailed explanations of techniques used

""")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Aerospace Software Defect Prediction Project</strong></p>
    <p>Using NASA Aerospace Module Datasets from Kaggle</p>
    <p>Built with Streamlit • Machine Learning • Feature Engineering</p>
</div>
""", unsafe_allow_html=True)
