"""
SMOTEENN Handler
Utilities for visualizing and explaining SMOTEENN class imbalance handling
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import SMOTEENN_CONFIG, COLORS


def get_smoteenn_stats():
    """Get SMOTEENN before/after statistics"""
    return SMOTEENN_CONFIG


def create_smoteenn_comparison():
    """Create comprehensive SMOTEENN before/after comparison visualization"""
    config = SMOTEENN_CONFIG

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Before SMOTEENN - Class Distribution',
            'After SMOTEENN - Class Distribution',
            'Sample Count Comparison',
            'Class Balance Ratio'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Before - Pie chart
    fig.add_trace(go.Pie(
        labels=['Non-Defective', 'Defective'],
        values=[config['before']['non_defective'], config['before']['defective']],
        marker_colors=[COLORS['non_defective'], COLORS['defective']],
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='label+percent'
    ), row=1, col=1)

    # After - Pie chart
    fig.add_trace(go.Pie(
        labels=['Non-Defective', 'Defective'],
        values=[config['after']['non_defective'], config['after']['defective']],
        marker_colors=[COLORS['non_defective'], COLORS['defective']],
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='label+percent'
    ), row=1, col=2)

    # Sample count comparison
    categories = ['Total Samples', 'Non-Defective', 'Defective']
    before_counts = [
        config['before']['total_samples'],
        config['before']['non_defective'],
        config['before']['defective']
    ]
    after_counts = [
        config['after']['total_samples'],
        config['after']['non_defective'],
        config['after']['defective']
    ]

    fig.add_trace(go.Bar(
        name='Before SMOTEENN',
        x=categories,
        y=before_counts,
        marker_color=COLORS['aerospace_blue'],
        text=before_counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        name='After SMOTEENN',
        x=categories,
        y=after_counts,
        marker_color=COLORS['accent'],
        text=after_counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ), row=2, col=1)

    # Class balance ratio
    imbalance_before = config['before']['non_defective'] / config['before']['defective']
    imbalance_after = config['after']['defective'] / config['after']['non_defective']

    fig.add_trace(go.Bar(
        x=['Before SMOTEENN', 'After SMOTEENN'],
        y=[imbalance_before, imbalance_after],
        marker_color=[COLORS['danger'], COLORS['success']],
        text=[f'{imbalance_before:.2f}:1', f'{imbalance_after:.2f}:1'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Ratio: %{y:.2f}:1<extra></extra>',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        title_text='SMOTEENN Class Imbalance Handling - Comprehensive Analysis',
        showlegend=True,
        legend=dict(x=0.35, y=-0.05, orientation='h')
    )

    fig.update_xaxes(title_text='Category', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_xaxes(title_text='Stage', row=2, col=2)
    fig.update_yaxes(title_text='Imbalance Ratio', row=2, col=2)

    return fig


def create_sample_flow_diagram():
    """Create a Sankey diagram showing sample flow through SMOTEENN"""
    config = SMOTEENN_CONFIG

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[
                f"Original Dataset<br>({config['before']['total_samples']} samples)",
                f"Non-Defective<br>({config['before']['non_defective']})",
                f"Defective<br>({config['before']['defective']})",
                "SMOTE<br>(Oversampling)",
                "ENN<br>(Cleaning)",
                f"Balanced Dataset<br>({config['after']['total_samples']} samples)",
                f"Non-Defective<br>({config['after']['non_defective']})",
                f"Defective<br>({config['after']['defective']})"
            ],
            color=[
                COLORS['aerospace_blue'],
                COLORS['non_defective'],
                COLORS['defective'],
                COLORS['accent'],
                COLORS['secondary'],
                COLORS['success'],
                COLORS['non_defective'],
                COLORS['defective']
            ]
        ),
        link=dict(
            source=[0, 0, 1, 2, 3, 4, 4],
            target=[1, 2, 4, 3, 4, 6, 7],
            value=[
                config['before']['non_defective'],
                config['before']['defective'],
                config['before']['non_defective'],
                config['before']['defective'],
                config['after']['total_samples'],
                config['after']['non_defective'],
                config['after']['defective']
            ],
            color=[
                'rgba(78, 205, 196, 0.4)',
                'rgba(255, 107, 107, 0.4)',
                'rgba(78, 205, 196, 0.3)',
                'rgba(255, 107, 107, 0.3)',
                'rgba(255, 166, 0, 0.3)',
                'rgba(78, 205, 196, 0.4)',
                'rgba(255, 107, 107, 0.4)'
            ]
        )
    )])

    fig.update_layout(
        title_text='SMOTEENN Sample Flow: From Imbalanced to Balanced Dataset',
        font_size=12,
        height=500
    )

    return fig


def get_smoteenn_explanation():
    """Get detailed SMOTEENN explanation text"""
    return {
        'title': 'SMOTEENN: Combined Oversampling and Cleaning',
        'description': '''
SMOTEENN combines two powerful techniques to handle class imbalance:

**1. SMOTE (Synthetic Minority Over-sampling Technique)**
- Creates synthetic samples of the minority class (defective modules)
- Uses k-nearest neighbors to generate realistic new examples
- Increases representation of defective cases from 9.8% to majority

**2. ENN (Edited Nearest Neighbors)**
- Cleans the dataset by removing noisy or ambiguous samples
- Removes samples that are misclassified by their k-nearest neighbors
- Improves the quality of decision boundaries

**Results:**
- Original: 498 samples (90.2% non-defective, 9.8% defective) - highly imbalanced
- After SMOTEENN: 723 samples (41.1% non-defective, 58.9% defective) - balanced
- Imbalance ratio improved from 9.16:1 to 0.70:1
- Better model training with fair representation of both classes
        ''',
        'benefits': [
            'Prevents model bias toward majority class',
            'Improves minority class (defect) detection',
            'Creates realistic synthetic samples',
            'Removes noisy borderline cases',
            'Enhances model generalization'
        ],
        'metrics': {
            'Samples Added': config['after']['defective'] - config['before']['defective'],
            'Samples Removed': config['before']['non_defective'] - config['after']['non_defective'],
            'Net Increase': config['after']['total_samples'] - config['before']['total_samples'],
            'Final Balance': f"{config['after']['defective_pct']:.1f}% defective"
        }
    }


config = SMOTEENN_CONFIG
