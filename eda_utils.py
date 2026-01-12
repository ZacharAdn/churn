"""
EDA Utility Functions for Telco Customer Churn Analysis
========================================================
Helper functions for data loading, preprocessing, and chart generation.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load and perform initial preprocessing on the churn dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(filepath)
    
    # Convert TotalCharges to numeric (handle empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0 (new customers with tenure=0)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Create binary churn column for analysis
    df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
    
    return df


def get_feature_categories():
    """
    Return dictionary of feature categories with descriptions.
    """
    return {
        'demographics': {
            'columns': ['gender', 'SeniorCitizen', 'Partner', 'Dependents'],
            'description': 'Customer personal information and family status'
        },
        'services': {
            'columns': ['tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies'],
            'description': 'Services subscribed by the customer'
        },
        'billing': {
            'columns': ['Contract', 'PaperlessBilling', 'PaymentMethod',
                       'MonthlyCharges', 'TotalCharges'],
            'description': 'Contract and payment information'
        }
    }


def get_feature_explanations():
    """
    Return dictionary with simple explanations for each feature.
    """
    return {
        'customerID': 'Unique identifier for each customer',
        'gender': 'Customer gender (Male/Female)',
        'SeniorCitizen': 'Is the customer 65 years or older? (1=Yes, 0=No)',
        'Partner': 'Does the customer have a partner/spouse?',
        'Dependents': 'Does the customer have dependents (children, elderly, etc.)?',
        'tenure': 'Number of months the customer has been with the company',
        'PhoneService': 'Does the customer have phone service?',
        'MultipleLines': 'Does the customer have multiple phone lines?',
        'InternetService': 'Type of internet service (DSL, Fiber optic, or None)',
        'OnlineSecurity': 'Does the customer have online security add-on?',
        'OnlineBackup': 'Does the customer have online backup add-on?',
        'DeviceProtection': 'Does the customer have device protection add-on?',
        'TechSupport': 'Does the customer have tech support add-on?',
        'StreamingTV': 'Does the customer have streaming TV add-on?',
        'StreamingMovies': 'Does the customer have streaming movies add-on?',
        'Contract': 'Contract type (Month-to-month, One year, Two year)',
        'PaperlessBilling': 'Does the customer use paperless billing?',
        'PaymentMethod': 'How does the customer pay?',
        'MonthlyCharges': 'Amount charged monthly (in dollars)',
        'TotalCharges': 'Total amount charged over entire tenure (in dollars)',
        'Churn': 'Did the customer leave the company? (Target variable)'
    }


# =============================================================================
# STATISTICAL CALCULATIONS
# =============================================================================

def calculate_churn_rate(df, column):
    """
    Calculate churn rate for each category in a column.
    
    Returns:
        pd.DataFrame with category, count, and churn rate
    """
    grouped = df.groupby(column).agg(
        Total=('Churn', 'count'),
        Churned=('Churn_Binary', 'sum')
    ).reset_index()
    grouped['Churn_Rate'] = (grouped['Churned'] / grouped['Total'] * 100).round(2)
    grouped['Retained'] = grouped['Total'] - grouped['Churned']
    return grouped


def get_correlation_matrix(df):
    """
    Create correlation matrix for numerical and encoded categorical features.
    """
    # Select and encode features
    df_encoded = df.copy()
    
    # Binary encode Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
    
    # Encode categorical columns
    cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaymentMethod']
    
    for col in cat_cols:
        if col in df_encoded.columns:
            df_encoded[col] = pd.factorize(df_encoded[col])[0]
    
    # Select numeric columns for correlation
    numeric_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                    'PhoneService', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'Contract', 'PaperlessBilling', 'MonthlyCharges', 
                    'TotalCharges', 'Churn']
    
    available_cols = [col for col in numeric_cols if col in df_encoded.columns]
    
    return df_encoded[available_cols].corr()


def get_top_churn_factors(df):
    """
    Identify top factors correlated with churn.
    """
    corr_matrix = get_correlation_matrix(df)
    churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
    return churn_corr


# =============================================================================
# CHART GENERATION FUNCTIONS
# =============================================================================

# Color scheme
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Purple
    'success': '#22c55e',      # Green
    'danger': '#ef4444',       # Red
    'warning': '#f59e0b',      # Amber
    'info': '#06b6d4',         # Cyan
    'churn_yes': '#ef4444',    # Red for churned
    'churn_no': '#22c55e',     # Green for retained
    'neutral': '#64748b'       # Slate
}


def create_churn_pie_chart(df):
    """Create pie chart showing churn distribution."""
    churn_counts = df['Churn'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Retained', 'Churned'],
        values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
        hole=0.4,
        marker_colors=[COLORS['churn_no'], COLORS['churn_yes']],
        textinfo='percent+label',
        textfont_size=14,
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Customer Churn Distribution',
        showlegend=True,
        height=400
    )
    
    return fig


def create_churn_bar_chart(df):
    """Create bar chart showing churn counts."""
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn Status', 'Count']
    churn_counts['Churn Status'] = churn_counts['Churn Status'].map({'No': 'Retained', 'Yes': 'Churned'})
    
    fig = px.bar(
        churn_counts,
        x='Churn Status',
        y='Count',
        color='Churn Status',
        color_discrete_map={'Retained': COLORS['churn_no'], 'Churned': COLORS['churn_yes']},
        text='Count'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title='Customer Count by Churn Status',
        showlegend=False,
        height=400,
        xaxis_title='',
        yaxis_title='Number of Customers'
    )
    
    return fig


def create_categorical_churn_chart(df, column, title=None):
    """Create grouped bar chart showing churn distribution for a categorical variable."""
    churn_data = calculate_churn_rate(df, column)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Retained',
        x=churn_data[column],
        y=churn_data['Retained'],
        marker_color=COLORS['churn_no'],
        text=churn_data['Retained'],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Churned',
        x=churn_data[column],
        y=churn_data['Churned'],
        marker_color=COLORS['churn_yes'],
        text=churn_data['Churned'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title or f'Churn Distribution by {column}',
        barmode='group',
        xaxis_title=column,
        yaxis_title='Number of Customers',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_churn_rate_chart(df, column, title=None):
    """Create bar chart showing churn rate for each category."""
    churn_data = calculate_churn_rate(df, column)
    
    # Sort by churn rate
    churn_data = churn_data.sort_values('Churn_Rate', ascending=True)
    
    fig = px.bar(
        churn_data,
        x='Churn_Rate',
        y=column,
        orientation='h',
        color='Churn_Rate',
        color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
        text=churn_data['Churn_Rate'].apply(lambda x: f'{x:.1f}%')
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title=title or f'Churn Rate by {column}',
        xaxis_title='Churn Rate (%)',
        yaxis_title='',
        height=400,
        coloraxis_showscale=False
    )
    
    return fig


def create_tenure_histogram(df):
    """Create histogram of tenure by churn status."""
    fig = px.histogram(
        df,
        x='tenure',
        color='Churn',
        nbins=30,
        barmode='overlay',
        color_discrete_map={'No': COLORS['churn_no'], 'Yes': COLORS['churn_yes']},
        opacity=0.7
    )
    
    fig.update_layout(
        title='Customer Tenure Distribution by Churn Status',
        xaxis_title='Tenure (Months)',
        yaxis_title='Number of Customers',
        height=400,
        legend_title='Churn Status'
    )
    
    return fig


def create_tenure_boxplot(df):
    """Create box plot of tenure by churn status."""
    fig = px.box(
        df,
        x='Churn',
        y='tenure',
        color='Churn',
        color_discrete_map={'No': COLORS['churn_no'], 'Yes': COLORS['churn_yes']}
    )
    
    fig.update_layout(
        title='Tenure Distribution: Churned vs Retained',
        xaxis_title='Churn Status',
        yaxis_title='Tenure (Months)',
        height=400,
        showlegend=False
    )
    
    return fig


def create_monthly_charges_distribution(df):
    """Create histogram of monthly charges by churn status."""
    fig = px.histogram(
        df,
        x='MonthlyCharges',
        color='Churn',
        nbins=30,
        barmode='overlay',
        color_discrete_map={'No': COLORS['churn_no'], 'Yes': COLORS['churn_yes']},
        opacity=0.7
    )
    
    fig.update_layout(
        title='Monthly Charges Distribution by Churn Status',
        xaxis_title='Monthly Charges ($)',
        yaxis_title='Number of Customers',
        height=400
    )
    
    return fig


def create_monthly_charges_boxplot(df):
    """Create box plot of monthly charges by churn status."""
    fig = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map={'No': COLORS['churn_no'], 'Yes': COLORS['churn_yes']}
    )
    
    fig.update_layout(
        title='Monthly Charges: Churned vs Retained',
        xaxis_title='Churn Status',
        yaxis_title='Monthly Charges ($)',
        height=400,
        showlegend=False
    )
    
    return fig


def create_scatter_charges(df):
    """Create scatter plot of monthly vs total charges."""
    fig = px.scatter(
        df,
        x='tenure',
        y='MonthlyCharges',
        color='Churn',
        size='TotalCharges',
        color_discrete_map={'No': COLORS['churn_no'], 'Yes': COLORS['churn_yes']},
        opacity=0.6,
        hover_data=['Contract', 'InternetService']
    )
    
    fig.update_layout(
        title='Tenure vs Monthly Charges (bubble size = Total Charges)',
        xaxis_title='Tenure (Months)',
        yaxis_title='Monthly Charges ($)',
        height=500
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    corr_matrix = get_correlation_matrix(df)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig


def create_churn_factors_chart(df):
    """Create horizontal bar chart of top churn factors."""
    churn_corr = get_top_churn_factors(df)
    
    # Get top 10 factors
    top_factors = churn_corr.head(10)
    
    colors = [COLORS['danger'] if x > 0 else COLORS['success'] for x in top_factors.values]
    
    fig = go.Figure(go.Bar(
        x=top_factors.values,
        y=top_factors.index,
        orientation='h',
        marker_color=colors,
        text=[f'{x:.3f}' for x in top_factors.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Top 10 Features Correlated with Churn',
        xaxis_title='Correlation with Churn',
        yaxis_title='',
        height=450,
        xaxis=dict(range=[-0.5, 0.5])
    )
    
    return fig


def create_services_heatmap(df):
    """Create heatmap showing churn rate by service combinations."""
    # Calculate churn rate for each internet service and add-on combination
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Filter customers with internet service
    df_internet = df[df['InternetService'] != 'No'].copy()
    
    # Calculate churn rates
    churn_rates = {}
    for service in services:
        for value in ['Yes', 'No']:
            subset = df_internet[df_internet[service] == value]
            if len(subset) > 0:
                rate = subset['Churn_Binary'].mean() * 100
                churn_rates[(service, value)] = rate
    
    # Create matrix
    matrix_data = []
    for service in services:
        row = [
            churn_rates.get((service, 'No'), 0),
            churn_rates.get((service, 'Yes'), 0)
        ]
        matrix_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=['Without Service', 'With Service'],
        y=services,
        colorscale='RdYlGn_r',
        text=[[f'{val:.1f}%' for val in row] for row in matrix_data],
        texttemplate='%{text}',
        hovertemplate='%{y}<br>%{x}<br>Churn Rate: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Churn Rate by Service Add-ons (Internet Customers Only)',
        height=400,
        xaxis_title='Service Status',
        yaxis_title='Service Type'
    )
    
    return fig


def create_contract_payment_heatmap(df):
    """Create heatmap of churn rate by contract and payment method."""
    pivot = df.pivot_table(
        values='Churn_Binary',
        index='Contract',
        columns='PaymentMethod',
        aggfunc='mean'
    ) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=[[f'{val:.1f}%' for val in row] for row in pivot.values],
        texttemplate='%{text}',
        hovertemplate='Contract: %{y}<br>Payment: %{x}<br>Churn Rate: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Churn Rate by Contract Type and Payment Method',
        height=350,
        xaxis_title='Payment Method',
        yaxis_title='Contract Type',
        xaxis_tickangle=-30
    )
    
    return fig
