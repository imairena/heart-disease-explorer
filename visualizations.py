import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_correlation_heatmap(df):
    """
    Create an interactive correlation heatmap using Plotly.
    """
    if len(df) < 2:
        return go.Figure().add_annotation(text="Not enough data for correlation heatmap", showarrow=False)

    corr = df.corr()
    
    # Hide the upper triangle by setting it to NaN
    # np.triu_indices_from gets the indices for the upper triangle
    z_vals = corr.values.astype(float).copy()
    z_vals[np.triu_indices_from(z_vals, k=1)] = np.nan
    
    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',  # red positive, blue negative
        zmin=-1, zmax=1,
        text=np.round(z_vals, 2),
        texttemplate='%{text}',
        hovertext=[[f"X: {x}<br>Y: {y}<br>Corr: {z:.2f}" if not np.isnan(z) else "" 
                    for x, z in zip(corr.columns, row)] 
                   for y, row in zip(corr.index, z_vals)],
        hoverinfo='text',
        showscale=True
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        title_font=dict(size=16),
        height=600,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(tickangle=-45),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_feature_distributions(df):
    """
    Plot interactive distributions of numeric features using Plotly.
    """
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=numeric_cols,
                        vertical_spacing=0.15, horizontal_spacing=0.1)
    
    colors = {0: '#2ecc71', 1: '#e74c3c'}
    names = {0: 'No Disease', 1: 'Disease'}
    
    for i, col in enumerate(numeric_cols):
        row = (i // 3) + 1
        col_pos = (i % 3) + 1
        
        for target_val in [0, 1]:
            subset = df[df['num'] == target_val][col]
            fig.add_trace(go.Histogram(
                x=subset,
                name=names[target_val],
                marker_color=colors[target_val],
                opacity=0.7,
                histnorm='probability density',
                showlegend=(i == 0) # Only show legend once on the first subplot
            ), row=row, col=col_pos)
            
    fig.update_layout(
        title='Feature Distributions by Heart Disease Status',
        title_font=dict(size=16),
        barmode='overlay',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_target_breakdown(df):
    """
    Create interactive visualizations showing target variable breakdown and age-based analysis.
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=['Heart Disease Prevalence', 'Disease Rate by Age Group (%)'])
    
    counts = df['num'].value_counts()
    
    # Left plot: Pie chart
    fig.add_trace(go.Pie(
        labels=['No Disease', 'Disease'],
        values=[counts.get(0, 0), counts.get(1, 0)],
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        hole=0.4,
        hoverinfo="label+percent+value",
        textinfo="percent"
    ), row=1, col=1)
    
    # Right plot: Bar chart
    # Use observed=False to avoid future warnings
    age_group = pd.cut(
        df['age'], 
        bins=[0, 40, 50, 60, 70, 100], 
        labels=['<40', '40-50', '50-60', '60-70', '70+']
    )
    age_rates = df.groupby(age_group, observed=False)['num'].mean() * 100
    
    fig.add_trace(go.Bar(
        x=age_rates.index.astype(str),
        y=age_rates.values,
        marker_color='#e74c3c',
        showlegend=False,
        text=np.round(age_rates.values, 1),
        textposition='auto',
        hoverinfo="x+y"
    ), row=1, col=2)
    
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_yaxes(title_text="Disease Rate %", row=1, col=2)
    
    return fig

def plot_risk_factors(df):
    """
    Create interactive box plots comparing key risk factors between groups.
    """
    risk_pairs = [
        ('thalach', 'Max Heart Rate'),
        ('chol', 'Cholesterol (mg/dl)'),
        ('trestbps', 'Resting BP (mm Hg)'),
        ('oldpeak', 'ST Depression')
    ]
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[title for _, title in risk_pairs],
                        vertical_spacing=0.15, horizontal_spacing=0.1)
    
    colors = {0: '#2ecc71', 1: '#e74c3c'}
    names = {0: 'No Disease', 1: 'Disease'}
    
    for i, (col, title) in enumerate(risk_pairs):
        row = (i // 2) + 1
        col_pos = (i % 2) + 1
        
        for target_val in [0, 1]:
            subset = df[df['num'] == target_val][col]
            fig.add_trace(go.Box(
                y=subset,
                name=names[target_val],
                marker_color=colors[target_val],
                showlegend=(i == 0),
                boxpoints='outliers' # only show outliers
            ), row=row, col=col_pos)
            
    fig.update_layout(
        title='Key Risk Factors: Disease vs No Disease',
        title_font=dict(size=16),
        boxmode='group',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
