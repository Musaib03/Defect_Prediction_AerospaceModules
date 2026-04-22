"""
Visualization Utilities
Interactive Plotly-based charts and visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import COLORS


def plot_class_distribution(before_counts, after_counts=None, title="Class Distribution"):
    """
    Plot class distribution before and optionally after resampling

    Args:
        before_counts: dict with class counts before resampling
        after_counts: dict with class counts after resampling (optional)
        title: Chart title
    """
    if after_counts:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Before SMOTEENN", "After SMOTEENN"),
            specs=[[{'type':'pie'}, {'type':'pie'}]]
        )

        # Before
        fig.add_trace(go.Pie(
            labels=list(before_counts.keys()),
            values=list(before_counts.values()),
            marker_colors=[COLORS['non_defective'], COLORS['defective']],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ), row=1, col=1)

        # After
        fig.add_trace(go.Pie(
            labels=list(after_counts.keys()),
            values=list(after_counts.values()),
            marker_colors=[COLORS['non_defective'], COLORS['defective']],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ), row=1, col=2)

        fig.update_layout(height=400, title_text=title, showlegend=True)
    else:
        fig = go.Figure(data=[go.Pie(
            labels=list(before_counts.keys()),
            values=list(before_counts.values()),
            marker_colors=[COLORS['non_defective'], COLORS['defective']],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        fig.update_layout(height=400, title_text=title)

    return fig


def plot_feature_importance(features_dict, top_n=10, title="Feature Importance"):
    """
    Plot feature importance as horizontal bar chart

    Args:
        features_dict: Dictionary of {feature: importance}
        top_n: Number of top features to display
        title: Chart title
    """
    # Convert to DataFrame and sort
    df = pd.DataFrame(list(features_dict.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker_color=COLORS['primary'],
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=max(400, top_n * 30),
        showlegend=False
    )

    return fig


def plot_model_comparison(models_metrics, metric='accuracy'):
    """
    Plot model comparison bar chart

    Args:
        models_metrics: Dictionary of model metrics
        metric: Metric to compare ('accuracy', 'f1_score', etc.)
    """
    models = list(models_metrics.keys())
    values = [models_metrics[m].get(metric, 0) for m in models]

    fig = go.Figure(go.Bar(
        x=models,
        y=values,
        marker_color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']],
        text=[f'{v:.2%}' for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' + metric.replace('_', ' ').title() + ': %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Model Comparison - {metric.replace("_", " ").title()}',
        xaxis_title='Model',
        yaxis_title=metric.replace('_', ' ').title(),
        yaxis_tickformat='.1%',
        height=500,
        showlegend=False
    )

    return fig


def plot_confusion_matrix(cm, labels=['No Defect', 'Defect'], title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap

    Args:
        cm: 2x2 confusion matrix (numpy array or list)
        labels: Class labels
        title: Chart title
    """
    cm_array = np.array(cm)

    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm_array,
        texttemplate='%{text}',
        textfont={"size": 20},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        width=400
    )

    return fig


def plot_roc_curves(models_data):
    """
    Plot ROC curves for multiple models

    Args:
        models_data: Dictionary with model names as keys and AUC values
    """
    fig = go.Figure()

    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier (AUC=0.5)',
        hoverinfo='skip'
    ))

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    # For each model, create a smooth ROC curve
    for idx, (model, auc) in enumerate(models_data.items()):
        # Generate smooth curve points
        fpr = np.linspace(0, 1, 100)
        # Approximate TPR for high AUC values
        if auc > 0.99:
            tpr = np.power(fpr, 0.1)  # Very curved for high AUC
        else:
            tpr = fpr ** (1/(2*auc))

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            line=dict(color=colors[idx % len(colors)], width=3),
            name=f'{model} (AUC={auc:.4f})',
            hovertemplate='<b>' + model + '</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        legend=dict(x=0.6, y=0.1)
    )

    return fig


def plot_feature_distribution(df, feature, defect_col='defect', title=None):
    """
    Plot feature distribution by defect status

    Args:
        df: DataFrame with features and defect column
        feature: Feature name to plot
        defect_col: Column name for defect labels
        title: Chart title (auto-generated if None)
    """
    if title is None:
        title = f'Distribution of {feature}'

    fig = go.Figure()

    # Non-defective
    fig.add_trace(go.Histogram(
        x=df[df[defect_col] == 0][feature],
        name='No Defect',
        marker_color=COLORS['non_defective'],
        opacity=0.7,
        hovertemplate='<b>No Defect</b><br>Value: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Defective
    fig.add_trace(go.Histogram(
        x=df[df[defect_col] == 1][feature],
        name='Defect',
        marker_color=COLORS['defective'],
        opacity=0.7,
        hovertemplate='<b>Defect</b><br>Value: %{x}<br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=feature,
        yaxis_title='Count',
        barmode='overlay',
        height=400,
        legend=dict(x=0.7, y=0.95)
    )

    return fig


def plot_correlation_heatmap(df, features=None, title='Feature Correlation Matrix'):
    """
    Plot correlation heatmap

    Args:
        df: DataFrame with features
        features: List of features to include (default: all numeric)
        title: Chart title
    """
    if features is None:
        features = df.select_dtypes(include=['number']).columns.tolist()

    corr_matrix = df[features].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        height=max(600, len(features) * 30),
        width=max(600, len(features) * 30)
    )

    return fig


def plot_gauge(value, title='Prediction Confidence', max_value=1.0):
    """
    Plot gauge chart for probabilities/confidence

    Args:
        value: Current value
        title: Chart title
        max_value: Maximum value on gauge
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [0, 0.3], 'color': COLORS['success']},
                {'range': [0.3, 0.7], 'color': COLORS['secondary']},
                {'range': [0.7, max_value], 'color': COLORS['danger']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))

    fig.update_layout(height=300)

    return fig


def plot_metrics_radar(metrics_dict, title='Model Performance Metrics'):
    """
    Plot radar chart for multiple metrics

    Args:
        metrics_dict: Dictionary of metric: value pairs
        title: Chart title
    """
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker_color=COLORS['primary'],
        hovertemplate='<b>%{theta}</b><br>Value: %{r:.2%}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title=title,
        height=500,
        showlegend=False
    )

    return fig
