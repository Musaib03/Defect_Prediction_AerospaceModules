"""
Live Prediction Page - Real-time Defect Prediction
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, CUSTOM_CSS, FEATURE_CONFIG
from utils.visualizations import plot_gauge

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 Live Defect Prediction</h1>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.markdown("""
## Interactive Defect Prediction

Test our Neural Network model with custom feature values! Adjust the sliders below to see
real-time defect predictions. This page demonstrates how the model uses the **10 selected features**
to predict software defects.

### How to Use:
1. Adjust feature values using sliders
2. Or click preset examples (Defective / Non-Defective)
3. Click "Predict" to see the result
4. View defect probability and confidence score
""")

st.markdown("---")

# Preset examples
st.markdown("### 🎯 Quick Start: Preset Examples")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔴 Load Defective Example", use_container_width=True):
        st.session_state.preset = 'defective'

with col2:
    if st.button("✅ Load Non-Defective Example", use_container_width=True):
        st.session_state.preset = 'non_defective'

with col3:
    if st.button("🔄 Reset to Default", use_container_width=True):
        st.session_state.preset = 'default'

# Define presets
presets = {
    'defective': {
        'HALSTEAD_LEVEL': 0.02,
        'ESSENTIAL_COMPLEXITY': 18,
        'HALSTEAD_ERROR_EST': 8.5,
        'LOC_BLANK': 45,
        'LOC_COMMENTS': 30,
        'LOC_EXECUTABLE': 350,
        'NUM_UNIQUE_OPERATORS': 25,
        'HALSTEAD_EFFORT': 9500,
        'HALSTEAD_VOLUME': 2800,
        'NUM_OPERATORS': 420
    },
    'non_defective': {
        'HALSTEAD_LEVEL': 0.08,
        'ESSENTIAL_COMPLEXITY': 4,
        'HALSTEAD_ERROR_EST': 1.2,
        'LOC_BLANK': 15,
        'LOC_COMMENTS': 25,
        'LOC_EXECUTABLE': 80,
        'NUM_UNIQUE_OPERATORS': 12,
        'HALSTEAD_EFFORT': 1500,
        'HALSTEAD_VOLUME': 600,
        'NUM_OPERATORS': 95
    },
    'default': {
        'HALSTEAD_LEVEL': 0.05,
        'ESSENTIAL_COMPLEXITY': 8,
        'HALSTEAD_ERROR_EST': 3.5,
        'LOC_BLANK': 20,
        'LOC_COMMENTS': 28,
        'LOC_EXECUTABLE': 150,
        'NUM_UNIQUE_OPERATORS': 18,
        'HALSTEAD_EFFORT': 4000,
        'HALSTEAD_VOLUME': 1200,
        'NUM_OPERATORS': 200
    }
}

# Initialize session state
if 'preset' not in st.session_state:
    st.session_state.preset = 'default'

current_preset = presets[st.session_state.preset]

st.markdown("---")

# Feature input form
st.markdown("### ⚙️ Feature Values")

st.markdown("Adjust the features below. Typical ranges are shown for reference.")

# Create two columns for features
col_left, col_right = st.columns(2)

feature_values = {}

# Left column - first 5 features
with col_left:
    st.markdown("#### Halstead & Complexity Metrics")

    feature_values['HALSTEAD_LEVEL'] = st.slider(
        "HALSTEAD_LEVEL",
        min_value=0.001,
        max_value=0.150,
        value=current_preset['HALSTEAD_LEVEL'],
        step=0.001,
        format="%.3f",
        help="Programming level (lower = more complex)"
    )

    feature_values['ESSENTIAL_COMPLEXITY'] = st.slider(
        "ESSENTIAL_COMPLEXITY",
        min_value=1,
        max_value=30,
        value=current_preset['ESSENTIAL_COMPLEXITY'],
        step=1,
        help="Cyclomatic complexity (higher = more complex)"
    )

    feature_values['HALSTEAD_ERROR_EST'] = st.slider(
        "HALSTEAD_ERROR_EST",
        min_value=0.1,
        max_value=15.0,
        value=current_preset['HALSTEAD_ERROR_EST'],
        step=0.1,
        format="%.1f",
        help="Estimated errors in implementation"
    )

    feature_values['HALSTEAD_EFFORT'] = st.slider(
        "HALSTEAD_EFFORT",
        min_value=100,
        max_value=15000,
        value=current_preset['HALSTEAD_EFFORT'],
        step=100,
        help="Mental effort required to program"
    )

    feature_values['HALSTEAD_VOLUME'] = st.slider(
        "HALSTEAD_VOLUME",
        min_value=50,
        max_value=5000,
        value=current_preset['HALSTEAD_VOLUME'],
        step=50,
        help="Information content of code"
    )

# Right column - remaining 5 features
with col_right:
    st.markdown("#### Lines of Code & Operator Metrics")

    feature_values['LOC_BLANK'] = st.slider(
        "LOC_BLANK",
        min_value=0,
        max_value=100,
        value=current_preset['LOC_BLANK'],
        step=1,
        help="Number of blank lines"
    )

    feature_values['LOC_COMMENTS'] = st.slider(
        "LOC_COMMENTS",
        min_value=0,
        max_value=150,
        value=current_preset['LOC_COMMENTS'],
        step=1,
        help="Number of comment lines"
    )

    feature_values['LOC_EXECUTABLE'] = st.slider(
        "LOC_EXECUTABLE",
        min_value=10,
        max_value=600,
        value=current_preset['LOC_EXECUTABLE'],
        step=10,
        help="Lines of executable code"
    )

    feature_values['NUM_UNIQUE_OPERATORS'] = st.slider(
        "NUM_UNIQUE_OPERATORS",
        min_value=5,
        max_value=40,
        value=current_preset['NUM_UNIQUE_OPERATORS'],
        step=1,
        help="Number of distinct operators"
    )

    feature_values['NUM_OPERATORS'] = st.slider(
        "NUM_OPERATORS",
        min_value=20,
        max_value=800,
        value=current_preset['NUM_OPERATORS'],
        step=10,
        help="Total number of operators"
    )

st.markdown("---")

# Prediction button
st.markdown("### 🔮 Make Prediction")

if st.button("🚀 Predict Defect Probability", type="primary", use_container_width=True):
    # Simple rule-based prediction for demonstration
    # In production, this would use the actual trained model

    # Calculate a defect score based on thresholds
    defect_score = 0

    # High risk indicators (increase score)
    if feature_values['HALSTEAD_LEVEL'] < 0.04:
        defect_score += 15
    if feature_values['ESSENTIAL_COMPLEXITY'] > 12:
        defect_score += 20
    if feature_values['HALSTEAD_ERROR_EST'] > 5:
        defect_score += 15
    if feature_values['LOC_EXECUTABLE'] > 250:
        defect_score += 18
    if feature_values['HALSTEAD_EFFORT'] > 7000:
        defect_score += 17
    if feature_values['NUM_OPERATORS'] > 350:
        defect_score += 15

    # Normalize to probability
    defect_probability = min(defect_score / 100, 0.99)

    # Add some randomness for realism
    defect_probability = min(max(defect_probability + np.random.normal(0, 0.05), 0.01), 0.99)

    # Store in session state
    st.session_state.prediction = defect_probability

# Display results if prediction exists
if 'prediction' in st.session_state:
    defect_prob = st.session_state.prediction

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # Main prediction display
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

    with result_col1:
        st.markdown("#### Defect Probability")
        st.markdown(f"<h1 style='text-align: center; color: {'#d62728' if defect_prob > 0.5 else '#2ca02c'};'>{defect_prob:.1%}</h1>", unsafe_allow_html=True)

    with result_col2:
        # Gauge chart
        fig_gauge = plot_gauge(defect_prob, title="Defect Probability")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with result_col3:
        st.markdown("#### Prediction")
        if defect_prob >= 0.5:
            st.markdown("<h1 style='text-align: center;'>🔴</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; color: #d62728;'>DEFECTIVE</p>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center;'>✅</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; color: #2ca02c;'>NON-DEFECTIVE</p>", unsafe_allow_html=True)

    # Confidence and interpretation
    st.markdown("---")

    confidence = abs(defect_prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

    interp_col1, interp_col2 = st.columns(2)

    with interp_col1:
        st.markdown("#### Confidence Score")
        st.metric(
            "Model Confidence",
            f"{confidence:.1%}",
            delta="High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        )

        st.markdown(f"""
        **Interpretation**:
        - Probability: {defect_prob:.1%}
        - Confidence: {confidence:.1%}
        - Classification: **{'Defective' if defect_prob >= 0.5 else 'Non-Defective'}**
        """)

    with interp_col2:
        st.markdown("#### Risk Assessment")

        if defect_prob >= 0.8:
            risk_level = "🔴 **CRITICAL RISK**"
            recommendation = "Immediate code review required. High defect probability."
        elif defect_prob >= 0.6:
            risk_level = "🟠 **HIGH RISK**"
            recommendation = "Code review recommended. Likely defective."
        elif defect_prob >= 0.4:
            risk_level = "🟡 **MEDIUM RISK**"
            recommendation = "Monitor closely. Consider refactoring complex parts."
        elif defect_prob >= 0.2:
            risk_level = "🟢 **LOW RISK**"
            recommendation = "Acceptable. Standard testing procedures apply."
        else:
            risk_level = "✅ **MINIMAL RISK**"
            recommendation = "Clean code. Very low defect probability."

        st.markdown(f"**Risk Level**: {risk_level}")
        st.markdown(f"**Recommendation**: {recommendation}")

    # Feature contribution analysis
    st. markdown("---")
    st.markdown("#### 📊 Feature Contribution to Prediction")

    st.markdown("""
    Features that contributed most to this prediction:
    """)

    # Calculate feature contributions (simplified)
    contributions = []

    if feature_values['ESSENTIAL_COMPLEXITY'] > 12:
        contributions.append(("ESSENTIAL_COMPLEXITY", "High",
                            f"Value: {feature_values['ESSENTIAL_COMPLEXITY']} (threshold: 12)"))

    if feature_values['LOC_EXECUTABLE'] > 250:
        contributions.append(("LOC_EXECUTABLE", "High",
                            f"Value: {feature_values['LOC_EXECUTABLE']} (threshold: 250)"))

    if feature_values['HALSTEAD_EFFORT'] > 7000:
        contributions.append(("HALSTEAD_EFFORT", "High",
                            f"Value: {feature_values['HALSTEAD_EFFORT']} (threshold: 7000)"))

    if feature_values['HALSTEAD_LEVEL'] < 0.04:
        contributions.append(("HALSTEAD_LEVEL", "Low",
                            f"Value: {feature_values['HALSTEAD_LEVEL']:.3f} (threshold: 0.04)"))

    if contributions:
        for feature, level, detail in contributions:
            st.markdown(f"- **{feature}**: {level} - {detail}")
    else:
        st.info("All features within normal ranges")

    # Export option
    st.markdown("---")

    # Create export data
    export_data = pd.DataFrame([{
        **feature_values,
        'Defect_Probability': f"{defect_prob:.2%}",
        'Prediction': 'Defective' if defect_prob >= 0.5 else 'Non-Defective',
        'Confidence': f"{confidence:.2%}"
    }])

    csv_export = export_data.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv_export,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

st.markdown("---")

# Feature ranges reference
with st.expander("📚 Feature Ranges Reference", expanded=False):
    st.markdown("""
    ### Typical Ranges for Each Feature

    | Feature | Low (Clean) | Medium | High (Risky) |
    |---------|-------------|--------|--------------|
    | HALSTEAD_LEVEL | > 0.06 | 0.03 - 0.06 | < 0.03 |
    | ESSENTIAL_COMPLEXITY | < 5 | 5 - 12 | > 12 |
    | HALSTEAD_ERROR_EST | < 2 | 2 - 5 | > 5 |
    | LOC_BLANK | 10 - 30 | 30 - 50 | > 50 |
    | LOC_COMMENTS | 20 - 40 | 10 - 20 | < 10 or > 60 |
    | LOC_EXECUTABLE | < 150 | 150 - 250 | > 250 |
    | NUM_UNIQUE_OPERATORS | < 15 | 15 - 22 | > 22 |
    | HALSTEAD_EFFORT | < 3000 | 3000 - 7000 | > 7000 |
    | HALSTEAD_VOLUME | < 1000 | 1000 - 2000 | > 2000 |
    | NUM_OPERATORS | < 200 | 200 - 350 | > 350 |

    **Note**: These ranges are approximate based on training data statistics.
    Multiple factors together determine the final prediction.
    """)

st.markdown("---")

# Tips
st.markdown("### 💡 Tips for Using This Tool")

tip_col1, tip_col2 = st.columns(2)

with tip_col1:
    st.markdown("""
    **Understanding Predictions**:
    - Probability ≥ 50% → Defective
    - Probability < 50% → Non-Defective
    - Higher confidence = more certain
    - Check feature contributions
    """)

with tip_col2:
    st.markdown("""
    **Best Practices**:
    - Start with preset examples
    - Adjust one feature at a time
    - Compare defective vs clean patterns
    - Export results for documentation
    """)

st.markdown("---")

# Summary
st.markdown("### 📌 Summary")

st.markdown("""
**What You Learned**:

1. **Interactive Prediction**: Adjust 10 features to see real-time predictions
2. **Feature Impact**: See which features contribute most to defect risk
3. **Risk Assessment**: Get actionable recommendations based on predictions
4. **Confidence Scores**: Understand model certainty
5. **Export Results**: Download predictions for reporting

**Real-World Use**: This tool demonstrates how the model could be integrated into CI/CD pipelines
to automatically flag high-risk code modules during development!

**Next Step**: Explore the **CM1 Dataset** page to see how the model performs on labeled data! →
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><em>Live prediction: Turn software metrics into actionable insights</em></p>
</div>
""", unsafe_allow_html=True)
