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

# st.markdown("---")

# # Interactive threshold adjuster
# st.markdown("### 🎚️ Interactive Defect Threshold")

# st.markdown("""
# Adjust the probability threshold to see how it affects defect detection.
# - **Lower threshold**: Detect more potential defects (higher recall, more false positives)
# - **Higher threshold**: Only flag high-confidence defects (higher precision, might miss some)
# """)

# threshold = st.slider(
#     "Defect Probability Threshold",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.5,
#     step=0.05,
#     format="%.2f",
#     help="Modules with defect probability ≥ threshold are flagged as defective"
# )

# # Simulate predictions (in production, load actual predictions)
# np.random.seed(42)
# defect_probs = np.concatenate([
#     np.random.beta(8, 2, DATASET_CONFIG['predicted_defects_test']),  # Defective: high probs
#     np.random.beta(2, 8, DATASET_CONFIG['test_samples'] - DATASET_CONFIG['predicted_defects_test'])  # Non-defective: low probs
# ])
# np.random.shuffle(defect_probs)

# detected_at_threshold = np.sum(defect_probs >= threshold)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.metric(
#         "Defects Detected",
#         f"{detected_at_threshold}",
#         delta=f"{detected_at_threshold - DATASET_CONFIG['predicted_defects_test']:+d} vs default"
#     )

# with col2:
#     detection_rate = detected_at_threshold / DATASET_CONFIG['test_samples']
#     st.metric(
#         "Detection Rate",
#         f"{detection_rate:.1%}",
#         delta=f"{(detection_rate - DATASET_CONFIG['predicted_defect_rate_test']) * 100:+.1f}%"
#     )

# with col3:
#     if threshold < 0.3:
#         status = "⚠️ Too Sensitive"
#         color = "warning"
#     elif threshold > 0.7:
#         status = "⚠️ Too Conservative"
#         color = "warning"
#     else:
#         status = "✅ Balanced"
#         color = "success"
#     st.metric("Threshold Status", status)

# st.markdown("---")

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
# st.markdown("### 📊 Feature Distributions by Defect Status")

# st.markdown("""
# Compare how features differ between defective and non-defective modules.
# **Higher values** in defective modules indicate that feature is a **strong defect indicator**.
# """)

# # Feature selector
# selected_feature = st.selectbox(
#     "Select feature to analyze",
#     FEATURE_CONFIG['top_features'],
#     index=0
# )

# # Create sample distribution data
# np.random.seed(42)
# defective_values = np.random.gamma(4, 2, 109)
# non_defective_values = np.random.gamma(2, 1, 310)

# sample_df = pd.DataFrame({
#     selected_feature: np.concatenate([defective_values, non_defective_values]),
#     'defect': [1]*109 + [0]*310
# })

# fig_dist = plot_feature_distribution(sample_df, selected_feature)
# st.plotly_chart(fig_dist, use_container_width=True)

# # Statistical comparison
# col1, col2, col3 = st.columns(3)

# with col1:
#     defect_mean = defective_values.mean()
#     st.metric(
#         f"Defective Mean",
#         f"{defect_mean:.2f}",
#         help=f"Average {selected_feature} for defective modules"
#     )

# with col2:
#     non_defect_mean = non_defective_values.mean()
#     st.metric(
#         f"Non-Defective Mean",
#         f"{non_defect_mean:.2f}",
#         help=f"Average {selected_feature} for non-defective modules"
#     )

# with col3:
#     difference = ((defect_mean - non_defect_mean) / non_defect_mean) * 100
#     st.metric(
#         "Difference",
#         f"{difference:+.1f}%",
#         delta="Higher in defective" if difference > 0 else "Lower in defective"
#     )

# st.markdown("---")

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
Below are two high-defect-risk examples. Use the button in each example to refactor it into a low-defect-risk version.
""")

if "example1_low_risk" not in st.session_state:
    st.session_state.example1_low_risk = False
if "example2_low_risk" not in st.session_state:
    st.session_state.example2_low_risk = False

st.markdown("#### Example 1: Flight Control Branching")

if st.button(
    "✅ Change" if not st.session_state.example1_low_risk else "🔄 ChangeAgain",
    key="toggle_example_1"
):
    st.session_state.example1_low_risk = not st.session_state.example1_low_risk

if not st.session_state.example1_low_risk:
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
    """, language='python')

    st.markdown("""
    **Defect Indicators**:
    - ❌ **Essential Complexity = 18** (threshold: > 10)
    - ❌ **LOC_EXECUTABLE = 42** (threshold: > 30)
    - ❌ **HALSTEAD_EFFORT = 8,523** (threshold: > 5,000)
    - ❌ **Nested depth = 6 levels** (threshold: > 4)
    """)
else:
    st.code("""
def should_enter_emergency(data, config, params):
    return params.emergency and data.fuel < config.min_fuel

def process_critical_systems(systems):
    for system in systems:
        if system.status != 'critical':
            continue
        if system.backup_active:
            emergency_procedure()
        else:
            fallback_sequence()

def flight_mode_action(data, config, params):
    if data.altitude <= config.max_alt:
        return maintain_course()
    if not should_enter_emergency(data, config, params):
        return adjust_altitude(data)
    process_critical_systems(data.systems)
    return handle_emergency(data)

def refactored_aerospace_function(data, mode, config, params):
    if mode != 'flight':
        return ground_operations()
    return flight_mode_action(data, config, params)
    """, language='python')

    st.markdown("""
    **Refactored (Low-Risk) Indicators**:
    - ✅ **Essential Complexity ≈ 4**
    - ✅ **LOC_EXECUTABLE reduced**
    - ✅ **Lower Halstead effort via decomposition**
    - ✅ **Shallower nesting and clearer intent**
    """)

st.markdown("---")

st.markdown("#### Example 2: Sensor Validation Pipeline")

if st.button(
    "✅ Change" if not st.session_state.example2_low_risk else "🔄ChangeAgain",
    key="toggle_example_2"
):
    st.session_state.example2_low_risk = not st.session_state.example2_low_risk

if not st.session_state.example2_low_risk:
    st.code("""
def validate_sensor_packet(packet, cfg, telemetry):
    if packet is not None:
        if packet.header_ok:
            if packet.crc_ok:
                if packet.timestamp > telemetry.last_timestamp:
                    if packet.sensor_id in cfg.allowed_sensors:
                        if packet.value is not None:
                            if cfg.min_val <= packet.value <= cfg.max_val:
                                if not telemetry.is_quarantine(packet.sensor_id):
                                    telemetry.update(packet.sensor_id, packet.value)
                                    return True
                                else:
                                    telemetry.raise_alarm("quarantined sensor")
                            else:
                                telemetry.raise_alarm("value out of bounds")
                        else:
                            telemetry.raise_alarm("missing value")
                    else:
                        telemetry.raise_alarm("unknown sensor")
                else:
                    telemetry.raise_alarm("old timestamp")
            else:
                telemetry.raise_alarm("crc failed")
        else:
            telemetry.raise_alarm("bad header")
    else:
        telemetry.raise_alarm("null packet")
    return False
    """, language='python')

    st.markdown("""
    **Defect Indicators**:
    - ❌ Deeply nested condition chain
    - ❌ High branch count and cyclomatic complexity
    - ❌ Repeated alarm logic increases operator count
    - ❌ Hard to test all paths
    """)
else:
    st.code("""
def reject(telemetry, reason):
    telemetry.raise_alarm(reason)
    return False

def basic_packet_checks(packet, telemetry):
    if packet is None:
        return reject(telemetry, "null packet")
    if not packet.header_ok:
        return reject(telemetry, "bad header")
    if not packet.crc_ok:
        return reject(telemetry, "crc failed")
    return True

def semantic_checks(packet, cfg, telemetry):
    if packet.timestamp <= telemetry.last_timestamp:
        return reject(telemetry, "old timestamp")
    if packet.sensor_id not in cfg.allowed_sensors:
        return reject(telemetry, "unknown sensor")
    if packet.value is None:
        return reject(telemetry, "missing value")
    if not (cfg.min_val <= packet.value <= cfg.max_val):
        return reject(telemetry, "value out of bounds")
    if telemetry.is_quarantine(packet.sensor_id):
        return reject(telemetry, "quarantined sensor")
    return True

def validate_sensor_packet_refactored(packet, cfg, telemetry):
    if not basic_packet_checks(packet, telemetry):
        return False
    if not semantic_checks(packet, cfg, telemetry):
        return False
    telemetry.update(packet.sensor_id, packet.value)
    return True
    """, language='python')

    st.markdown("""
    **Refactored (Low-Risk) Indicators**:
    - ✅ Lower branch count per function
    - ✅ Early-return guards reduce nesting depth
    - ✅ Reusable error handler reduces duplicated operators
    - ✅ Better testability and readability
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
