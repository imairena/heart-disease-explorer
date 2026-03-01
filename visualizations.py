import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    """
    Create a correlation heatmap showing relationships between all features.
    
    This visualization helps identify which features are strongly correlated with
    each other and with the target variable. High correlation can indicate:
    - Redundant features (high correlation between features)
    - Important predictors (high correlation with target)
    
    Args:
        df: DataFrame with all features and target
        
    Returns:
        Matplotlib figure object to display in Streamlit
    """
    # Create a figure with specific size (width=12, height=10 inches)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate correlation matrix (Pearson correlation coefficient)
    # Values range from -1 (perfect negative correlation) to +1 (perfect positive)
    corr = df.corr()
    
    # Create a mask to show only the upper triangle of the heatmap
    # This avoids redundancy (correlation of A with B is same as B with A)
    # k=1 means we exclude the diagonal (correlation of feature with itself = 1.0)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    # Create a diverging color palette (blue for negative, red for positive)
    # This makes it easy to see positive vs negative correlations
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center='light', as_cmap=True)
    
    # Create the heatmap
    sns.heatmap(
        corr,                    # Correlation matrix to plot
        mask=mask,               # Hide lower triangle
        annot=True,              # Show correlation values in each cell
        fmt='.2f',               # Format numbers to 2 decimal places
        cmap=cmap,               # Color scheme
        center=0,                # Center colormap at 0 (neutral correlation)
        square=True,             # Make cells square-shaped
        linewidths=0.5,          # Width of lines between cells
        ax=ax,                   # Plot on our axis
        annot_kws={'size': 9}    # Font size for correlation values
    )
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()  # Adjust spacing to prevent label cutoff
    return fig


def plot_feature_distributions(df):
    """
    Plot distributions of numeric features, comparing disease vs no-disease groups.
    
    This visualization shows how the distribution of each numeric feature differs
    between patients with heart disease and those without. This helps identify
    which features are most predictive of disease status.
    
    Args:
        df: DataFrame with features and target variable
        
    Returns:
        Matplotlib figure with multiple subplots
    """
    # List of numeric features to visualize
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Create a 2x3 grid of subplots (6 total, but we only need 5)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()  # Convert 2D array to 1D for easier iteration
    
    # Plot each numeric feature
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        
        # Plot histogram for each target class (disease vs no disease)
        for target_val, label in [(0, 'No Disease'), (1, 'Disease')]:
            # Get subset of data for this target class and feature
            subset = df[df['num'] == target_val][col]
            
            # Create histogram with density=True (shows probability density, not counts)
            # This makes it easier to compare distributions even if class sizes differ
            ax.hist(subset, bins=20, alpha=0.6, label=label, density=True)
        
        # Customize the subplot
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel(col)
        ax.legend()  # Show legend with 'No Disease' and 'Disease' labels
        ax.grid(True, alpha=0.3)  # Add light grid for easier reading
    
    # Hide the last (6th) subplot since we only have 5 features
    axes[-1].axis('off')
    
    # Add overall title to the figure
    fig.suptitle('Feature Distributions by Heart Disease Status', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()  # Adjust spacing
    return fig


def plot_target_breakdown(df):
    """
    Create visualizations showing target variable breakdown and age-based analysis.
    
    This function creates two visualizations:
    1. Pie chart showing overall disease prevalence
    2. Bar chart showing disease rate by age group
    
    Args:
        df: DataFrame with features and target variable
        
    Returns:
        Matplotlib figure with two subplots
    """
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Pie chart showing disease prevalence
    # Count how many cases have disease (1) vs no disease (0)
    counts = df['num'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for no disease, red for disease
    
    axes[0].pie(
        counts,                                    # Values to plot
        labels=['No Disease', 'Disease'],          # Labels for each slice
        autopct='%1.1f%%',                        # Show percentage with 1 decimal place
        colors=colors,                            # Color scheme
        explode=(0, 0.05),                        # Slightly separate the disease slice (emphasis)
        startangle=90                             # Start angle (top of pie)
    )
    axes[0].set_title('Heart Disease Prevalence', fontweight='bold')
    
    # Right plot: Bar chart showing disease rate by age group
    # Create age groups using pandas cut function
    # Bins define the boundaries: 0-40, 40-50, 50-60, 60-70, 70-100
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 40, 50, 60, 70, 100], 
        labels=['<40', '40-50', '50-60', '60-70', '70+']
    )
    
    # Calculate disease rate (percentage) for each age group
    # Group by age_group, take mean of 'num' (0 or 1), multiply by 100 for percentage
    age_rates = df.groupby('age_group')['num'].mean() * 100
    
    # Create bar chart
    age_rates.plot(kind='bar', ax=axes[1], color='#e74c3c', edgecolor='black')
    axes[1].set_title('Disease Rate by Age Group (%)', fontweight='bold')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Disease Rate %')
    axes[1].tick_params(axis='x', rotation=0)  # Keep x-axis labels horizontal
    
    plt.tight_layout()
    return fig


def plot_risk_factors(df):
    """
    Create box plots comparing key risk factors between disease and no-disease groups.
    
    Box plots show the distribution of each risk factor, including:
    - Median (line in middle of box)
    - Quartiles (box edges)
    - Outliers (points beyond whiskers)
    
    This helps identify which risk factors show the most difference between groups.
    
    Args:
        df: DataFrame with features and target variable
        
    Returns:
        Matplotlib figure with 2x2 grid of box plots
    """
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define which features to plot and their display titles
    risk_pairs = [
        ('thalach', 'Max Heart Rate'),           # Maximum heart rate achieved
        ('chol', 'Cholesterol (mg/dl)'),         # Serum cholesterol
        ('trestbps', 'Resting BP (mm Hg)'),      # Resting blood pressure
        ('oldpeak', 'ST Depression')             # ST depression (ECG measure)
    ]
    
    # Create a box plot for each risk factor
    for ax, (col, title) in zip(axes.flatten(), risk_pairs):
        # Box plot grouped by disease status (x='num')
        # Shows distribution of the risk factor (y=col) for each group
        sns.boxplot(
            data=df, 
            x='num',                              # Group by disease status (0 or 1)
            y=col,                                # Risk factor to compare
            ax=ax,                                # Plot on this subplot
            palette=['#2ecc71', '#e74c3c']       # Green for no disease, red for disease
        )
        ax.set_title(title)
        ax.set_xlabel('Disease Status (0=No, 1=Yes)')
    
    # Add overall title
    fig.suptitle('Key Risk Factors: Disease vs No Disease', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
