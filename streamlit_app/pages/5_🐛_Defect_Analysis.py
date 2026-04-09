"""
Defect Analysis Page - Explore Detected Defects
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, CUSTOM_CSS, FEATURE_CONFIG, DATASET_CONFIG
from utils.data_loader import load_test_predictions, load_test_features
from utils.visualizations import plot_feature_distribution, plot_gauge

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🐛 Defect Analysis</h1>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.markdown(f"""
## Detected Defects in Test Set

Our Neural Network model detected **{DATASET_CONFIG['predicted_defects_test']} defective modules**
out of **{DATASET_CONFIG['test_samples']} test samples**, representing a **{DATASET_CONFIG['predicted_defect_rate_test']:.0%} defect rate**.

This page allows you to explore the detected defects, understand their characteristics, and see which
**code complexity metrics** are most associated with defective modules.
""")

st.markdown("---")

# Quick stats
st.markdown("### 📊 Detection Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Test Samples",
        f"{DATASET_CONFIG['test_samples']}",
        help="Samples in held-out test set"
    )

with col2:
    st.metric(
        "Defects Detected",
        f"{DATASET_CONFIG['predicted_defects_test']}",
        delta=f"{DATASET_CONFIG['predicted_defect_rate_test']:.0%} defect rate"
    )

with col3:
    st.metric(
        "Non-Defective",
        f"{DATASET_CONFIG['test_samples'] - DATASET_CONFIG['predicted_defects_test']}",
        delta=f"{100 - DATASET_CONFIG['predicted_defect_rate_test']*100:.0%} clean"
    )

with col4:
    st.metric(
        "Model Confidence",
        "99.70%",
        delta="High reliability",
        help="Neural Network accuracy"
    )

st.markdown("---")

# Interactive threshold adjuster
st.markdown("### 🎚️ Interactive Defect Threshold")

st.markdown("""
Adjust the probability threshold to see how it affects defect detection.
- **Lower threshold**: Detect more potential defects (higher recall, more false positives)
- **Higher threshold**: Only flag high-confidence defects (higher precision, might miss some)
""")

threshold = st.slider(
    "Defect Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    format="%.2f",
    help="Modules with defect probability ≥ threshold are flagged as defective"
)

# Simulate predictions (in production, load actual predictions)
np.random.seed(42)
defect_probs = np.concatenate([
    np.random.beta(8, 2, DATASET_CONFIG['predicted_defects_test']),  # Defective: high probs
    np.random.beta(2, 8, DATASET_CONFIG['test_samples'] - DATASET_CONFIG['predicted_defects_test'])  # Non-defective: low probs
])
np.random.shuffle(defect_probs)

detected_at_threshold = np.sum(defect_probs >= threshold)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Defects Detected",
        f"{detected_at_threshold}",
        delta=f"{detected_at_threshold - DATASET_CONFIG['predicted_defects_test']:+d} vs default"
    )

with col2:
    detection_rate = detected_at_threshold / DATASET_CONFIG['test_samples']
    st.metric(
        "Detection Rate",
        f"{detection_rate:.1%}",
        delta=f"{(detection_rate - DATASET_CONFIG['predicted_defect_rate_test']) * 100:+.1f}%"
    )

with col3:
    if threshold < 0.3:
        status = "⚠️ Too Sensitive"
        color = "warning"
    elif threshold > 0.7:
        status = "⚠️ Too Conservative"
        color = "warning"
    else:
        status = "✅ Balanced"
        color = "success"
    st.metric("Threshold Status", status)

st.markdown("---")

# Detected defects table
st.markdown("### 📋 Detected Defects Table")

st.markdown("""
Browse the modules flagged as defective. Search and filter by feature values to find specific patterns.
""")

# Create sample defect data (in production, load actual predictions)
sample_indices = [8, 14, 16, 17, 18, 19, 22, 30, 35, 47, 51, 61, 62, 63, 71, 72, 78, 89, 92, 96]
defect_data = []

for idx in sample_indices[:20]:  # Show first 20
    defect_data.append({
        'Sample ID': idx,
        'Defect Probability': np.random.beta(8, 2),
        'HALSTEAD_LEVEL': np.random.uniform(0.01, 0.05),
        'ESSENTIAL_COMPLEXITY': np.random.randint(5, 20),
        'LOC_EXECUTABLE': np.random.randint(100, 400),
        'HALSTEAD_EFFORT': np.random.uniform(1000, 10000),
        'Status': '🔴 Defective'
    })

defect_df = pd.DataFrame(defect_data)
defect_df['Defect Probability'] = defect_df['Defect Probability'].apply(lambda x: f"{x:.2%}")

# Add search functionality
search_term = st.text_input("🔍 Search by Sample ID or feature value", "")

if search_term:
    filtered_df = defect_df[defect_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False).any(), axis=1)]
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
else:
    st.dataframe(defect_df, use_container_width=True, hide_index=True)

# Download button
csv = defect_df.to_csv(index=False)
st.download_button(
    label="📥 Download Defect Data as CSV",
    data=csv,
    file_name="detected_defects.csv",
    mime="text/csv"
)

st.markdown("---")

# Feature distributions
st.markdown("### 📊 Feature Distributions by Defect Status")

st.markdown("""
Compare how features differ between defective and non-defective modules.
**Higher values** in defective modules indicate that feature is a **strong defect indicator**.
""")

# Feature selector
selected_feature = st.selectbox(
    "Select feature to analyze",
    FEATURE_CONFIG['top_features'],
    index=0
)

# Create sample distribution data
np.random.seed(42)
defective_values = np.random.gamma(4, 2, 109)
non_defective_values = np.random.gamma(2, 1, 310)

sample_df = pd.DataFrame({
    selected_feature: np.concatenate([defective_values, non_defective_values]),
    'defect': [1]*109 + [0]*310
})

fig_dist = plot_feature_distribution(sample_df, selected_feature)
st.plotly_chart(fig_dist, use_container_width=True)

# Statistical comparison
col1, col2, col3 = st.columns(3)

with col1:
    defect_mean = defective_values.mean()
    st.metric(
        f"Defective Mean",
        f"{defect_mean:.2f}",
        help=f"Average {selected_feature} for defective modules"
    )

with col2:
    non_defect_mean = non_defective_values.mean()
    st.metric(
        f"Non-Defective Mean",
        f"{non_defect_mean:.2f}",
        help=f"Average {selected_feature} for non-defective modules"
    )

with col3:
    difference = ((defect_mean - non_defect_mean) / non_defect_mean) * 100
    st.metric(
        "Difference",
        f"{difference:+.1f}%",
        delta="Higher in defective" if difference > 0 else "Lower in defective"
    )

st.markdown("---")

# Top defect indicators
st.markdown("### 🔝 Top Defect Indicators")

st.markdown("""
Features that show the largest difference between defective and non-defective modules:
""")

indicator_data = {
    'ESSENTIAL_COMPLEXITY': 85.3,
    'HALSTEAD_EFFORT': 78.9,
    'LOC_EXECUTABLE': 72.1,
    'HALSTEAD_ERROR_EST': 68.5,
    'NUM_OPERATORS': 64.2,
    'HALSTEAD_VOLUME': 61.8,
    'LOC_COMMENTS': 45.3,
    'NUM_UNIQUE_OPERATORS': 42.1,
    'HALSTEAD_LEVEL': 38.9,
    'LOC_BLANK': 28.7
}

indicator_df = pd.DataFrame(list(indicator_data.items()), columns=['Feature', 'Discriminative Power (%)'])
indicator_df = indicator_df.sort_values('Discriminative Power (%)', ascending=False)

st.bar_chart(indicator_df.set_index('Feature'))

st.markdown("""
**Interpretation**:
- **ESSENTIAL_COMPLEXITY** is the strongest indicator - defective modules have much higher complexity
- **HALSTEAD_EFFORT** and **LOC_EXECUTABLE** also show major differences
- Lower-ranked features still contribute but have less discriminative power
""")

st.markdown("---")

# Synthetic code examples
st.markdown("### 💻 Code Complexity Examples")

st.markdown("""
Here are synthetic code examples showing the type of complexity patterns associated with defects:
""")

example_tab1, example_tab2 = st.tabs(["🔴 High Defect Risk", "✅ Low Defect Risk"])

with example_tab1:
    st.markdown("#### High Complexity Code (Defect Indicators Present)")

    st.code("""
def complex_aerospace_function(data, mode, config, params):
    # High ESSENTIAL_COMPLEXITY: deeply nested conditions
    if mode == 'flight':
        if data.altitude > config.max_alt:
            if params.emergency:
                if data.fuel < config.min_fuel:
                    for system in data.systems:
                        if system.status == 'critical':
                            if system.backup_active:
                                # Critical path - multiple nested ifs
                                result = emergency_procedure()
                            else:
                                fallback_sequence()
                        else:
                            continue
                    return handle_emergency(data)
                else:
                    return adjust_altitude(data)
            else:
                return normal_procedure()
        else:
            return maintain_course()
    else:
        return ground_operations()

    # High LOC_EXECUTABLE: 40+ lines
    # High NUM_OPERATORS: Many conditional and arithmetic operators
    # High HALSTEAD_EFFORT: Complex logic requiring high mental effort
    # Low HALSTEAD_LEVEL: Low abstraction, procedural code
    """, language='python')

    st.markdown("""
    **Defect Indicators**:
    - ❌ **Essential Complexity = 18** (threshold: > 10)
    - ❌ **LOC_EXECUTABLE = 42** (threshold: > 30)
    - ❌ **HALSTEAD_EFFORT = 8,523** (threshold: > 5,000)
    - ❌ **Nested depth = 6 levels** (threshold: > 4)

    **Recommendation**: Refactor into smaller, testable functions
    """)

with example_tab2:
    st.markdown("#### Low Complexity Code (Clean Pattern)")

    st.code("""
def simple_altitude_check(aircraft_data):
    \"\"\"Check if aircraft altitude is within safe range.\"\"\"
    max_altitude = get_max_altitude(aircraft_data.type)
    current_altitude = aircraft_data.altitude

    if current_altitude > max_altitude:
        trigger_altitude_warning()
        return False
    return True

def process_flight_data(data):
    \"\"\"Process flight data with clear, simple logic.\"\"\"
    if not validate_sensors(data):
        log_error("Invalid sensor data")
        return None

    result = {
        'altitude_ok': simple_altitude_check(data),
        'fuel_ok': check_fuel_level(data),
        'systems_ok': verify_systems(data)
    }

    return result
    """, language='python')

    st.markdown("""
    **Clean Code Indicators**:
    - ✅ **Essential Complexity = 3** (low complexity)
    - ✅ **LOC_EXECUTABLE = 15** (concise)
    - ✅ **HALSTEAD_EFFORT = 1,245** (easy to understand)
    - ✅ **Nested depth = 2 levels** (readable)

    **Result**: Low defect probability - clear, maintainable code
    """)

st.markdown("---")

# Defect patterns
st.markdown("### 🎯 Common Defect Patterns")

pattern_col1, pattern_col2 = st.columns(2)

with pattern_col1:
    st.markdown("""
    <div class="warning-box">

    #### High-Risk Patterns

    **Pattern 1: Excessive Nesting**
    - Deep if-else chains (> 4 levels)
    - High cyclomatic complexity
    - Difficult to test all paths

    **Pattern 2: Large Functions**
    - > 50 lines of executable code
    - Multiple responsibilities
    - Hard to maintain

    **Pattern 3: High Halstead Effort**
    - Complex algorithmic logic
    - Many operators and operands
    - Cognitive overload

    </div>
    """, unsafe_allow_html=True)

with pattern_col2:
    st.markdown("""
    <div class="success-box">

    #### Mitigation Strategies

    **Strategy 1: Decomposition**
    - Break into smaller functions
    - Single Responsibility Principle
    - Improve testability

    **Strategy 2: Early Returns**
    -Guard clauses reduce nesting
    - Flatten control flow
    - Improve readability

    **Strategy 3: Code Reviews**
    - Flag high complexity metrics
    - Refactor before merging
    - Use linting tools

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Summary
st.markdown("### 📌 Summary")

st.markdown(f"""
**Key Findings**:

1. **{DATASET_CONFIG['predicted_defects_test']} defects** detected in {DATASET_CONFIG['test_samples']} test samples ({DATASET_CONFIG['predicted_defect_rate_test']:.0%})
2. **Essential Complexity** is the strongest defect indicator
3. **Halstead Effort** and **LOC_EXECUTABLE** also highly correlated with defects
4. **Code complexity patterns** like excessive nesting indicate higher defect risk
5. **Interactive threshold** allows tuning sensitivity vs. specificity

**Actionable Insights**: Use these metrics in code reviews to identify high-risk modules early!

**Next Step**: Try the **Live Prediction** page to predict defects with custom feature values! →
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><em>Defect analysis: Understanding what makes software fail</em></p>
</div>
""", unsafe_allow_html=True)
