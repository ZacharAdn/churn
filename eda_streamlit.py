"""
Telco Customer Churn - Interactive EDA Dashboard
=================================================
A comprehensive Exploratory Data Analysis dashboard built with Streamlit.
Designed to be understandable for both technical and non-technical users.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from eda_utils import (
    load_data, get_feature_categories, get_feature_explanations,
    calculate_churn_rate, get_correlation_matrix, get_top_churn_factors,
    create_churn_pie_chart, create_churn_bar_chart,
    create_categorical_churn_chart, create_churn_rate_chart,
    create_tenure_histogram, create_tenure_boxplot,
    create_monthly_charges_distribution, create_monthly_charges_boxplot,
    create_scatter_charges, create_correlation_heatmap,
    create_churn_factors_chart, create_services_heatmap,
    create_contract_payment_heatmap, COLORS
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Telco Churn EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
    .insight-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def get_data():
    return load_data()

df = get_data()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
st.sidebar.title("📊 EDA Navigation")

pages = [
    "🏠 Dataset Overview",
    "🎯 Target Variable (Churn)",
    "👥 Customer Demographics",
    "📡 Service Analysis",
    "💰 Billing & Contracts",
    "🔗 Correlation Analysis",
    "💡 Key Insights",
    "🔧 Data Preparation",
    "🤖 Baseline Models",
    "🔮 Predict Churn"
]

selected_page = st.sidebar.radio("Select a Section:", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Total Customers", f"{len(df):,}")
st.sidebar.metric("Churn Rate", f"{df['Churn_Binary'].mean()*100:.1f}%")
st.sidebar.metric("Avg Tenure", f"{df['tenure'].mean():.0f} months")


# =============================================================================
# PAGE 1: DATASET OVERVIEW
# =============================================================================

if selected_page == "🏠 Dataset Overview":
    st.markdown('<p class="main-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understanding the structure and contents of our data</p>', unsafe_allow_html=True)
    
    # What is this section?
    with st.expander("ℹ️ What will I learn in this section?", expanded=True):
        st.markdown("""
        This section provides a **bird's eye view** of the dataset:
        - How many customers and features do we have?
        - What type of information is available?
        - Are there any missing values we need to handle?
        
        **Why is this important?** Before analyzing data, we need to understand its structure. 
        This helps us plan our analysis and identify potential data quality issues.
        """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Records",
            value=f"{df.shape[0]:,}",
            help="Number of customers in the dataset"
        )
    
    with col2:
        st.metric(
            label="📊 Total Features",
            value=f"{df.shape[1]}",
            help="Number of columns/variables"
        )
    
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric(
            label="💾 Memory Size",
            value=f"{memory_mb:.2f} MB",
            help="Dataset size in memory"
        )
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric(
            label="❓ Missing Values",
            value=f"{missing_pct:.2f}%",
            help="Percentage of missing data"
        )
    
    st.markdown("---")
    
    # Data Preview
    st.subheader("🔍 Data Preview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        num_rows = st.slider("Number of rows to display:", 5, 50, 10)
    with col2:
        show_all_cols = st.checkbox("Show all columns", value=True)
    
    if show_all_cols:
        st.dataframe(df.head(num_rows), use_container_width=True)
    else:
        selected_cols = st.multiselect("Select columns:", df.columns.tolist(), default=df.columns[:5].tolist())
        st.dataframe(df[selected_cols].head(num_rows), use_container_width=True)
    
    st.markdown("---")
    
    # Column Information
    st.subheader("📝 Feature Descriptions")
    
    feature_explanations = get_feature_explanations()
    feature_categories = get_feature_categories()
    
    tabs = st.tabs(["👥 Demographics", "📡 Services", "💰 Billing", "🎯 Target"])
    
    with tabs[0]:
        st.markdown("**Customer demographic information:**")
        for col in feature_categories['demographics']['columns']:
            with st.expander(f"**{col}** - {df[col].dtype}"):
                st.write(f"📖 {feature_explanations.get(col, 'No description')}")
                st.write(f"**Unique values:** {df[col].nunique()}")
                if df[col].dtype == 'object':
                    st.write(f"**Categories:** {', '.join(df[col].unique().astype(str))}")
    
    with tabs[1]:
        st.markdown("**Services subscribed by customers:**")
        for col in feature_categories['services']['columns']:
            with st.expander(f"**{col}** - {df[col].dtype}"):
                st.write(f"📖 {feature_explanations.get(col, 'No description')}")
                st.write(f"**Unique values:** {df[col].nunique()}")
                if df[col].dtype == 'object':
                    st.write(f"**Categories:** {', '.join(df[col].unique().astype(str))}")
                else:
                    st.write(f"**Range:** {df[col].min()} - {df[col].max()}")
    
    with tabs[2]:
        st.markdown("**Contract and payment information:**")
        for col in feature_categories['billing']['columns']:
            with st.expander(f"**{col}** - {df[col].dtype}"):
                st.write(f"📖 {feature_explanations.get(col, 'No description')}")
                if df[col].dtype == 'object':
                    st.write(f"**Categories:** {', '.join(df[col].unique().astype(str))}")
                else:
                    st.write(f"**Range:** ${df[col].min():.2f} - ${df[col].max():.2f}")
                    st.write(f"**Average:** ${df[col].mean():.2f}")
    
    with tabs[3]:
        st.markdown("**Target variable we want to predict:**")
        with st.expander("**Churn** - object", expanded=True):
            st.write(f"📖 {feature_explanations.get('Churn', 'No description')}")
            st.write("**Categories:** Yes (customer left), No (customer stayed)")
            st.info("💡 This is what we want to predict - which customers will leave!")


# =============================================================================
# PAGE 2: TARGET VARIABLE ANALYSIS
# =============================================================================

elif selected_page == "🎯 Target Variable (Churn)":
    st.markdown('<p class="main-header">🎯 Target Variable Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understanding customer churn - our prediction target</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What is Customer Churn?", expanded=True):
        st.markdown("""
        **Churn** means a customer has stopped using the company's services.
        
        - **"Yes"** = Customer churned (left the company)
        - **"No"** = Customer retained (stayed with the company)
        
        **Why does this matter?**
        - Acquiring new customers costs 5-25x more than retaining existing ones
        - Predicting churn allows proactive retention strategies
        - Understanding churn factors helps improve customer experience
        """)
    
    # Churn distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_churn_pie_chart(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_churn_bar_chart(df), use_container_width=True)
    
    # Class imbalance explanation
    st.markdown("---")
    st.subheader("⚖️ Class Imbalance Analysis")
    
    churn_yes = len(df[df['Churn'] == 'Yes'])
    churn_no = len(df[df['Churn'] == 'No'])
    imbalance_ratio = churn_no / churn_yes
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Retained Customers", f"{churn_no:,}", help="Customers who stayed")
    with col2:
        st.metric("Churned Customers", f"{churn_yes:,}", help="Customers who left")
    with col3:
        st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1", help="Ratio of retained to churned")
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Class Imbalance Detected!</strong><br>
        The dataset has approximately <strong>2.8x more retained customers</strong> than churned customers.
        This imbalance needs to be addressed during model training to avoid biased predictions.
        <br><br>
        <strong>Common solutions:</strong>
        <ul>
            <li>SMOTE (Synthetic Minority Over-sampling)</li>
            <li>Class weights adjustment</li>
            <li>Undersampling the majority class</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Business context
    st.markdown("---")
    st.subheader("💼 Business Impact")
    
    avg_monthly = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
    lost_revenue = df[df['Churn'] == 'Yes']['TotalCharges'].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Avg Monthly Revenue Lost per Churned Customer",
            f"${avg_monthly:.2f}",
            help="Average monthly charges of churned customers"
        )
    
    with col2:
        st.metric(
            "Total Revenue from Churned Customers",
            f"${lost_revenue:,.0f}",
            help="Total charges accumulated before these customers left"
        )
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Key Insight:</strong><br>
        The 26.5% churn rate represents significant revenue loss potential. 
        Even a small reduction in churn (e.g., 5%) could save substantial revenue.
        This is why predicting and preventing churn is valuable!
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: CUSTOMER DEMOGRAPHICS
# =============================================================================

elif selected_page == "👥 Customer Demographics":
    st.markdown('<p class="main-header">👥 Customer Demographics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How do customer characteristics relate to churn?</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What will I learn in this section?", expanded=True):
        st.markdown("""
        We'll analyze how different customer characteristics affect churn:
        - **Gender**: Does gender affect likelihood to churn?
        - **Senior Citizens**: Are older customers more/less likely to leave?
        - **Partner/Dependents**: Does family status matter?
        
        **Understanding these patterns helps target retention efforts to the right customer segments.**
        """)
    
    # Gender Analysis
    st.subheader("👫 Gender Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_categorical_churn_chart(df, 'gender', 'Churn by Gender'), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_churn_rate_chart(df, 'gender', 'Churn Rate by Gender'), use_container_width=True)
    
    gender_insight = calculate_churn_rate(df, 'gender')
    male_rate = gender_insight[gender_insight['gender'] == 'Male']['Churn_Rate'].values[0]
    female_rate = gender_insight[gender_insight['gender'] == 'Female']['Churn_Rate'].values[0]
    
    if abs(male_rate - female_rate) < 2:
        st.markdown("""
        <div class="success-box">
            <strong>✅ Finding:</strong> Gender has minimal impact on churn rates. 
            Both male and female customers churn at similar rates (~26-27%).
            <strong>Gender is likely NOT a significant predictor.</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Senior Citizen Analysis
    st.subheader("👴 Senior Citizen Analysis")
    
    # Convert for display
    df_display = df.copy()
    df_display['SeniorCitizen_Label'] = df_display['SeniorCitizen'].map({0: 'Not Senior (< 65)', 1: 'Senior (65+)'})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_categorical_churn_chart(df_display, 'SeniorCitizen_Label', 'Churn by Age Group'), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_churn_rate_chart(df_display, 'SeniorCitizen_Label', 'Churn Rate by Age Group'), use_container_width=True)
    
    senior_rate = calculate_churn_rate(df_display, 'SeniorCitizen_Label')
    senior_churn = senior_rate[senior_rate['SeniorCitizen_Label'] == 'Senior (65+)']['Churn_Rate'].values[0]
    non_senior_churn = senior_rate[senior_rate['SeniorCitizen_Label'] == 'Not Senior (< 65)']['Churn_Rate'].values[0]
    
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ Important Finding:</strong> Senior citizens have a significantly higher churn rate!
        <ul>
            <li>Senior Citizens: <strong>{senior_churn:.1f}%</strong> churn rate</li>
            <li>Non-Seniors: <strong>{non_senior_churn:.1f}%</strong> churn rate</li>
        </ul>
        Senior citizens are <strong>{senior_churn/non_senior_churn:.1f}x more likely</strong> to churn.
        This segment may need special attention and retention strategies.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Partner & Dependents
    st.subheader("👨‍👩‍👧‍👦 Partner & Dependents Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_categorical_churn_chart(df, 'Partner', 'Churn by Partner Status'), use_container_width=True)
        st.plotly_chart(create_churn_rate_chart(df, 'Partner', 'Churn Rate by Partner Status'), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_categorical_churn_chart(df, 'Dependents', 'Churn by Dependents Status'), use_container_width=True)
        st.plotly_chart(create_churn_rate_chart(df, 'Dependents', 'Churn Rate by Dependents Status'), use_container_width=True)
    
    partner_rate = calculate_churn_rate(df, 'Partner')
    dependents_rate = calculate_churn_rate(df, 'Dependents')
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Key Insights - Family Status:</strong>
        <ul>
            <li>Customers <strong>without partners</strong> have higher churn rates (~33%) vs. with partners (~20%)</li>
            <li>Customers <strong>without dependents</strong> have higher churn rates (~31%) vs. with dependents (~15%)</li>
        </ul>
        <strong>Interpretation:</strong> Family obligations may increase customer "stickiness" - 
        these customers might be less likely to switch providers due to the complexity involved.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: SERVICE ANALYSIS
# =============================================================================

elif selected_page == "📡 Service Analysis":
    st.markdown('<p class="main-header">📡 Service Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How do subscribed services affect customer retention?</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What will I learn in this section?", expanded=True):
        st.markdown("""
        We'll explore:
        - **Tenure**: How long customers stay before churning
        - **Internet Service Type**: DSL vs Fiber Optic impact
        - **Add-on Services**: Do security, backup, and support services reduce churn?
        
        **These insights help identify which services keep customers engaged!**
        """)
    
    # Tenure Analysis
    st.subheader("📅 Customer Tenure Analysis")
    
    st.markdown("""
    **Tenure** is how long a customer has been with the company (in months).
    This is often one of the strongest predictors of churn.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_tenure_histogram(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_tenure_boxplot(df), use_container_width=True)
    
    avg_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].mean()
    avg_tenure_retained = df[df['Churn'] == 'No']['tenure'].mean()
    
    st.markdown(f"""
    <div class="warning-box">
        <strong>🔥 Critical Finding - Tenure is KEY!</strong>
        <ul>
            <li>Churned customers average tenure: <strong>{avg_tenure_churned:.1f} months</strong></li>
            <li>Retained customers average tenure: <strong>{avg_tenure_retained:.1f} months</strong></li>
        </ul>
        <strong>Customers who churn typically leave within the first 1-2 years!</strong>
        The first year is the most critical period for retention efforts.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Internet Service
    st.subheader("🌐 Internet Service Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_categorical_churn_chart(df, 'InternetService', 'Churn by Internet Service'), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_churn_rate_chart(df, 'InternetService', 'Churn Rate by Internet Type'), use_container_width=True)
    
    internet_rates = calculate_churn_rate(df, 'InternetService')
    fiber_rate = internet_rates[internet_rates['InternetService'] == 'Fiber optic']['Churn_Rate'].values[0]
    dsl_rate = internet_rates[internet_rates['InternetService'] == 'DSL']['Churn_Rate'].values[0]
    
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ Surprising Finding - Fiber Optic Customers Churn More!</strong>
        <ul>
            <li>Fiber Optic: <strong>{fiber_rate:.1f}%</strong> churn rate</li>
            <li>DSL: <strong>{dsl_rate:.1f}%</strong> churn rate</li>
            <li>No Internet: Very low churn</li>
        </ul>
        <strong>Possible reasons:</strong> Higher prices, service quality issues, or more tech-savvy customers who shop around more.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add-on Services Heatmap
    st.subheader("🛡️ Add-on Services Impact")
    
    st.markdown("""
    Do additional services like Online Security, Backup, or Tech Support help retain customers?
    The heatmap below shows churn rates for customers with/without each service.
    """)
    
    st.plotly_chart(create_services_heatmap(df), use_container_width=True)
    
    st.markdown("""
    <div class="success-box">
        <strong>✅ Key Insight - Services Reduce Churn!</strong><br>
        Customers who subscribe to add-on services (especially <strong>Tech Support</strong> and <strong>Online Security</strong>) 
        have significantly lower churn rates. These services may:
        <ul>
            <li>Increase customer engagement</li>
            <li>Create additional value</li>
            <li>Make switching more costly (lock-in effect)</li>
        </ul>
        <strong>Recommendation:</strong> Encourage new customers to adopt these services!
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5: BILLING & CONTRACTS
# =============================================================================

elif selected_page == "💰 Billing & Contracts":
    st.markdown('<p class="main-header">💰 Billing & Contract Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How do pricing and contract terms affect churn?</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What will I learn in this section?", expanded=True):
        st.markdown("""
        Financial factors often have the biggest impact on churn:
        - **Contract Type**: Month-to-month vs. long-term contracts
        - **Payment Method**: How customers pay affects retention
        - **Monthly Charges**: Price sensitivity analysis
        
        **This is often where the most actionable insights are found!**
        """)
    
    # Contract Type - THE BIG ONE!
    st.subheader("📝 Contract Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_categorical_churn_chart(df, 'Contract', 'Churn by Contract Type'), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_churn_rate_chart(df, 'Contract', 'Churn Rate by Contract Type'), use_container_width=True)
    
    contract_rates = calculate_churn_rate(df, 'Contract')
    mtm_rate = contract_rates[contract_rates['Contract'] == 'Month-to-month']['Churn_Rate'].values[0]
    one_year_rate = contract_rates[contract_rates['Contract'] == 'One year']['Churn_Rate'].values[0]
    two_year_rate = contract_rates[contract_rates['Contract'] == 'Two year']['Churn_Rate'].values[0]
    
    st.markdown(f"""
    <div class="warning-box">
        <strong>🚨 CRITICAL FINDING - Contract Type is the #1 Predictor!</strong>
        <table style="width:100%; margin-top: 10px;">
            <tr><td>Month-to-month:</td><td><strong style="color: #ef4444;">{mtm_rate:.1f}%</strong> churn rate</td></tr>
            <tr><td>One year:</td><td><strong style="color: #f59e0b;">{one_year_rate:.1f}%</strong> churn rate</td></tr>
            <tr><td>Two year:</td><td><strong style="color: #22c55e;">{two_year_rate:.1f}%</strong> churn rate</td></tr>
        </table>
        <br>
        <strong>Month-to-month customers are {mtm_rate/two_year_rate:.0f}x more likely to churn!</strong>
        <br><br>
        <strong>Business Action:</strong> Incentivize longer contracts with discounts or perks.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Payment Method
    st.subheader("💳 Payment Method Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_categorical_churn_chart(df, 'PaymentMethod', 'Churn by Payment Method'), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_churn_rate_chart(df, 'PaymentMethod', 'Churn Rate by Payment Method'), use_container_width=True)
    
    payment_rates = calculate_churn_rate(df, 'PaymentMethod')
    echeck_rate = payment_rates[payment_rates['PaymentMethod'] == 'Electronic check']['Churn_Rate'].values[0]
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>💡 Insight - Electronic Check Customers Churn More!</strong><br>
        Electronic check users have <strong>{echeck_rate:.1f}%</strong> churn rate, significantly higher than automatic payment methods (~15-18%).
        <br><br>
        <strong>Possible reasons:</strong>
        <ul>
            <li>Less committed customers choose manual payment</li>
            <li>Automatic payments create psychological commitment</li>
            <li>Friction to cancel is lower with manual payments</li>
        </ul>
        <strong>Recommendation:</strong> Encourage automatic payment enrollment!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contract & Payment Heatmap
    st.subheader("🗺️ Contract + Payment Method Combination")
    
    st.plotly_chart(create_contract_payment_heatmap(df), use_container_width=True)
    
    st.markdown("---")
    
    # Monthly Charges
    st.subheader("💵 Monthly Charges Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_monthly_charges_distribution(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_monthly_charges_boxplot(df), use_container_width=True)
    
    avg_charges_churned = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
    avg_charges_retained = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>💡 Price Sensitivity Insight:</strong>
        <ul>
            <li>Churned customers average: <strong>${avg_charges_churned:.2f}/month</strong></li>
            <li>Retained customers average: <strong>${avg_charges_retained:.2f}/month</strong></li>
        </ul>
        Churned customers actually pay <strong>MORE</strong> on average! 
        This suggests high-value customers may be leaving due to perceived poor value-for-money,
        not because they can't afford the service.
    </div>
    """, unsafe_allow_html=True)
    
    # Scatter plot
    st.subheader("📊 Tenure vs. Charges Relationship")
    st.plotly_chart(create_scatter_charges(df), use_container_width=True)


# =============================================================================
# PAGE 6: CORRELATION ANALYSIS
# =============================================================================

elif selected_page == "🔗 Correlation Analysis":
    st.markdown('<p class="main-header">🔗 Correlation Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Statistical relationships between features and churn</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What is Correlation?", expanded=True):
        st.markdown("""
        **Correlation** measures how strongly two variables are related:
        - **+1**: Perfect positive relationship (as one increases, so does the other)
        - **0**: No relationship
        - **-1**: Perfect negative relationship (as one increases, the other decreases)
        
        **For churn prediction:**
        - **Positive correlation** with churn = feature associated with higher churn risk
        - **Negative correlation** with churn = feature associated with lower churn risk
        """)
    
    # Correlation Heatmap
    st.subheader("🗺️ Feature Correlation Heatmap")
    
    st.markdown("""
    This heatmap shows how all features relate to each other. 
    Look at the **last row/column (Churn)** to see which features most strongly predict churn.
    """)
    
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
    
    st.markdown("---")
    
    # Top Churn Factors
    st.subheader("🎯 Top Features Correlated with Churn")
    
    st.plotly_chart(create_churn_factors_chart(df), use_container_width=True)
    
    churn_factors = get_top_churn_factors(df)
    
    st.markdown("""
    <div class="insight-box">
        <strong>📊 How to Read This Chart:</strong>
        <ul>
            <li><span style="color: #ef4444;">Red bars (positive)</span> = Higher values increase churn risk</li>
            <li><span style="color: #22c55e;">Green bars (negative)</span> = Higher values decrease churn risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Increases Churn Risk")
        risk_factors = churn_factors[churn_factors > 0].head(5)
        for factor, value in risk_factors.items():
            st.markdown(f"- **{factor}**: {value:.3f}")
    
    with col2:
        st.markdown("### 🟢 Decreases Churn Risk")
        protective_factors = churn_factors[churn_factors < 0].head(5)
        for factor, value in protective_factors.items():
            st.markdown(f"- **{factor}**: {value:.3f}")
    
    st.markdown("""
    <div class="success-box">
        <strong>✅ Key Statistical Insights:</strong>
        <ul>
            <li><strong>Contract type</strong> shows the strongest negative correlation - longer contracts = less churn</li>
            <li><strong>Tenure</strong> is highly protective - the longer customers stay, the less likely they leave</li>
            <li><strong>Monthly charges</strong> show slight positive correlation - higher prices slightly increase churn risk</li>
            <li><strong>Online security/Tech support</strong> have protective effects</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 7: KEY INSIGHTS SUMMARY
# =============================================================================

elif selected_page == "💡 Key Insights":
    st.markdown('<p class="main-header">💡 Key Insights & Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Summary of findings and actionable recommendations</p>', unsafe_allow_html=True)
    
    # Top 5 Churn Predictors
    st.subheader("🏆 Top 5 Churn Predictors")
    
    predictors = [
        ("1️⃣", "Contract Type", "Month-to-month contracts have 42% churn vs 3% for two-year", "critical"),
        ("2️⃣", "Tenure", "New customers (< 12 months) churn the most", "high"),
        ("3️⃣", "Internet Service Type", "Fiber optic customers churn at 42% vs 19% for DSL", "high"),
        ("4️⃣", "Payment Method", "Electronic check users churn at 45%", "medium"),
        ("5️⃣", "Tech Support/Security", "Customers without these services churn more", "medium")
    ]
    
    for rank, name, description, importance in predictors:
        if importance == "critical":
            color = "#ef4444"
        elif importance == "high":
            color = "#f59e0b"
        else:
            color = "#3b82f6"
        
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 0 0.5rem 0.5rem 0;">
            <strong>{rank} {name}</strong><br>
            {description}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Business Recommendations
    st.subheader("💼 Business Recommendations")
    
    recommendations = [
        {
            "title": "Incentivize Long-Term Contracts",
            "description": "Offer significant discounts or perks for 1-2 year contracts. The data shows month-to-month customers are 14x more likely to churn.",
            "impact": "High",
            "effort": "Medium"
        },
        {
            "title": "Focus on First Year Retention",
            "description": "Implement a robust onboarding program and regular check-ins during the first 12 months. Most churn happens early.",
            "impact": "High",
            "effort": "Medium"
        },
        {
            "title": "Promote Automatic Payment",
            "description": "Offer a small discount for automatic payment enrollment. Electronic check users churn 2.5x more than auto-pay users.",
            "impact": "Medium",
            "effort": "Low"
        },
        {
            "title": "Bundle Security & Support Services",
            "description": "Create attractive bundles including Online Security and Tech Support. These services significantly reduce churn.",
            "impact": "Medium",
            "effort": "Low"
        },
        {
            "title": "Investigate Fiber Optic Issues",
            "description": "High churn in Fiber customers suggests service or value issues. Conduct customer surveys to identify problems.",
            "impact": "High",
            "effort": "High"
        },
        {
            "title": "Senior Citizen Outreach",
            "description": "Develop retention programs specifically for senior customers who have 41% churn rate vs 24% for non-seniors.",
            "impact": "Medium",
            "effort": "Medium"
        }
    ]
    
    for rec in recommendations:
        impact_color = "#22c55e" if rec["impact"] == "High" else "#f59e0b" if rec["impact"] == "Medium" else "#64748b"
        effort_color = "#22c55e" if rec["effort"] == "Low" else "#f59e0b" if rec["effort"] == "Medium" else "#ef4444"
        
        st.markdown(f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;">
            <strong>✅ {rec["title"]}</strong><br>
            {rec["description"]}<br>
            <span style="background-color: {impact_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-top: 8px; display: inline-block;">Impact: {rec["impact"]}</span>
            <span style="background-color: {effort_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-left: 4px;">Effort: {rec["effort"]}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Quality Summary
    st.subheader("📋 Data Quality Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Strengths
        - No significant missing values
        - Clean categorical variables
        - Good sample size (7,043 customers)
        - Balanced gender distribution
        - Rich feature set covering multiple aspects
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Considerations
        - 26.5% churn rate (imbalanced classes)
        - TotalCharges had 11 empty values (new customers)
        - Some features have high cardinality
        - Temporal information (when churn occurred) not available
        """)
    
    st.markdown("---")
    
    # Next Steps
    st.subheader("🚀 Next Steps for Modeling")
    
    st.markdown("""
    Based on this EDA, here are recommended next steps:
    
    1. **Data Preprocessing**
       - Encode categorical variables (One-Hot or Label Encoding)
       - Scale numerical features (StandardScaler)
       - Handle class imbalance (SMOTE, class weights)
    
    2. **Feature Engineering**
       - Create tenure buckets (new, medium, long-term)
       - Calculate average monthly service cost
       - Create interaction features (e.g., contract + tenure)
    
    3. **Model Selection**
       - Start with Logistic Regression (interpretable baseline)
       - Try Random Forest and Gradient Boosting
       - Consider XGBoost or LightGBM for best performance
    
    4. **Evaluation Focus**
       - Use F1-score or AUC-ROC (due to imbalance)
       - Focus on Recall to catch churning customers
       - Consider business cost of false positives vs negatives
    """)
    
    st.balloons()
    
    st.success("🎉 EDA Complete! You now have a comprehensive understanding of the Telco Customer Churn dataset.")


# =============================================================================
# PAGE 8: DATA PREPARATION
# =============================================================================

elif selected_page == "🔧 Data Preparation":
    st.markdown('<p class="main-header">🔧 Data Preparation Summary</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How the data was prepared for machine learning</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What happened in this step?", expanded=True):
        st.markdown("""
        After completing the exploratory data analysis, we prepared the data for machine learning:
        - **Cleaned** missing values
        - **Encoded** categorical variables
        - **Balanced** the target variable
        - **Scaled** numerical features
        - **Split** into training and test sets
        
        This page shows you exactly what changed and the final prepared dataset.
        """)
    
    # Check if prepared data exists
    try:
        X_train = pd.read_csv('X_train.csv')
        X_test = pd.read_csv('X_test.csv')
        y_train = pd.read_csv('y_train.csv')
        y_test = pd.read_csv('y_test.csv')
        data_prepared = True
    except FileNotFoundError:
        data_prepared = False
        st.error("⚠️ Prepared data files not found! Please run `data_preparation.py` first.")
    
    if data_prepared:
        # Before and After Comparison
        st.subheader("📊 Before vs After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Original Data")
            st.metric("Total Samples", "7,043")
            st.metric("Features", "20")
            st.metric("Class Balance", "73.5% / 26.5%")
            st.metric("Missing Values", "11")
        
        with col2:
            st.markdown("### 📤 Prepared Data")
            total_prepared = len(X_train) + len(X_test)
            st.metric("Total Samples", f"{total_prepared:,}", delta=f"-{7043-total_prepared:,}")
            st.metric("Features", "29", delta="+9")
            st.metric("Class Balance", "50% / 50%", delta="Balanced!")
            st.metric("Missing Values", "0", delta="-11")
        
        st.markdown("---")
        
        # Preparation Pipeline Visualization
        st.subheader("⚙️ Data Preparation Pipeline")
        
        # Create a visual flow
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; color: white; margin: 1rem 0;">
            <h3 style="margin-top: 0; margin-bottom: 0.5rem; color: white;">⚙️ Processing Steps</h3>
            <p style="margin: 0; opacity: 0.95;">Each data transformation step explained</p>
        </div>
        """, unsafe_allow_html=True)
        
        steps = [
            {
                "number": "1️⃣",
                "title": "Load Raw Data",
                "details": "Loaded 7,043 customers with 21 features (including customerID)",
                "color": "#f0f9ff"
            },
            {
                "number": "2️⃣",
                "title": "Handle Missing Values",
                "details": "• Found 11 missing values in TotalCharges<br>• Filled with 0 (new customers with tenure=0)<br>• Dropped customerID (not useful for prediction)",
                "color": "#fef3c7"
            },
            {
                "number": "3️⃣",
                "title": "Encode Categorical Variables",
                "details": "• Binary encoding: Partner, Dependents, PhoneService, PaperlessBilling<br>• Ordinal encoding: Contract (logical order)<br>• One-hot encoding: 9 nominal features",
                "color": "#e0e7ff"
            },
            {
                "number": "4️⃣",
                "title": "Balance Target Variable",
                "details": "• Used naive undersampling<br>• Majority class: 5,174 → 1,869<br>• Minority class: 1,869 (kept all)<br>• Result: Perfect 50/50 balance",
                "color": "#fce7f3"
            },
            {
                "number": "5️⃣",
                "title": "Train/Test Split",
                "details": "• Split ratio: 80/20<br>• Training: 2,990 samples<br>• Test: 748 samples<br>• Stratified split (maintains balance)",
                "color": "#d1fae5"
            },
            {
                "number": "6️⃣",
                "title": "Scale Numerical Features",
                "details": "• StandardScaler applied<br>• Features scaled: tenure, MonthlyCharges, TotalCharges<br>• Scaler saved for deployment",
                "color": "#dbeafe"
            }
        ]
        
        for step in steps:
            st.markdown(f"""
            <div style="background-color: {step['color']}; border-left: 4px solid #6366f1; padding: 1rem; margin: 0.5rem 0; border-radius: 0 0.5rem 0.5rem 0; color: #1e293b;">
                <strong>{step['number']} {step['title']}</strong><br>
                <div style="margin-top: 0.5rem; font-size: 0.9rem;">{step['details']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Class Balance Visualization
        st.subheader("⚖️ Class Balance Transformation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Original distribution
            fig_before = go.Figure(data=[go.Pie(
                labels=['Retained (No)', 'Churned (Yes)'],
                values=[5174, 1869],
                hole=0.4,
                marker_colors=[COLORS['churn_no'], COLORS['churn_yes']],
                textinfo='label+percent',
                textfont_size=12
            )])
            fig_before.update_layout(title='Before Balancing', height=350)
            st.plotly_chart(fig_before, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #fef3c7; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; color: #78350f;">
                <strong>⚠️ Imbalanced:</strong> 73.5% vs 26.5%<br>
                Models would be biased toward majority class
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # After balancing
            fig_after = go.Figure(data=[go.Pie(
                labels=['Retained (No)', 'Churned (Yes)'],
                values=[1869, 1869],
                hole=0.4,
                marker_colors=[COLORS['churn_no'], COLORS['churn_yes']],
                textinfo='label+percent',
                textfont_size=12
            )])
            fig_after.update_layout(title='After Balancing (Undersampling)', height=350)
            st.plotly_chart(fig_after, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #d1fae5; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; color: #065f46;">
                <strong>✅ Balanced:</strong> 50% vs 50%<br>
                Models will treat both classes equally
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature Engineering Summary
        st.subheader("🔨 Feature Engineering Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Original Features (20)**
            - CustomerID ❌
            - Gender
            - SeniorCitizen
            - Partner
            - Dependents
            - Tenure
            - PhoneService
            - MultipleLines
            - InternetService
            - OnlineSecurity
            - OnlineBackup
            - DeviceProtection
            - TechSupport
            - StreamingTV
            - StreamingMovies
            - Contract
            - PaperlessBilling
            - PaymentMethod
            - MonthlyCharges
            - TotalCharges
            """)
        
        with col2:
            st.markdown("""
            **Encoding Applied**
            - ❌ Dropped
            - Binary (0/1)
            - Keep as-is (0/1)
            - Binary (0/1)
            - Binary (0/1)
            - Scaled
            - Binary (0/1)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - One-Hot (2 cols)
            - Ordinal (0,1,2)
            - Binary (0/1)
            - One-Hot (3 cols)
            - Scaled
            - Scaled
            """)
        
        with col3:
            st.markdown("""
            **Result**
            - -
            - 1 feature
            - 1 feature
            - 1 feature
            - 1 feature
            - 1 feature
            - 1 feature
            - 2 features
            - 2 features
            - 2 features
            - 2 features
            - 2 features
            - 2 features
            - 2 features
            - 2 features
            - 1 feature
            - 1 feature
            - 3 features
            - 1 feature
            - 1 feature
            
            **Total: 29 features**
            """)
        
        st.markdown("---")
        
        # Preview Prepared Data
        st.subheader("👀 Preview of Prepared Data")
        
        tab1, tab2 = st.tabs(["Training Data", "Test Data"])
        
        with tab1:
            st.markdown(f"**Training Set:** {len(X_train):,} samples × {len(X_train.columns)} features")
            st.dataframe(X_train.head(10), use_container_width=True)
            
            st.markdown("**Training Labels Distribution:**")
            train_dist = y_train.value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Class 0 (No Churn)", f"{train_dist[0]:,}")
            with col2:
                st.metric("Class 1 (Churn)", f"{train_dist[1]:,}")
        
        with tab2:
            st.markdown(f"**Test Set:** {len(X_test):,} samples × {len(X_test.columns)} features")
            st.dataframe(X_test.head(10), use_container_width=True)
            
            st.markdown("**Test Labels Distribution:**")
            test_dist = y_test.value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Class 0 (No Churn)", f"{test_dist[0]:,}")
            with col2:
                st.metric("Class 1 (Churn)", f"{test_dist[1]:,}")
        
        st.markdown("---")
        
        # Files Created
        st.subheader("💾 Files Created")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Files:**
            - ✅ `X_train.csv` - Training features (2,990 × 29)
            - ✅ `X_test.csv` - Test features (748 × 29)
            - ✅ `y_train.csv` - Training labels
            - ✅ `y_test.csv` - Test labels
            """)
        
        with col2:
            st.markdown("""
            **Artifacts for Deployment:**
            - ✅ `scaler.pkl` - Fitted StandardScaler
            - ✅ `feature_names.pkl` - List of 29 feature names
            """)
        
        st.markdown("---")
        
        # Trade-offs and Considerations
        st.subheader("⚖️ Trade-offs & Considerations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Advantages
            
            **Perfect Class Balance**
            - Models won't be biased
            - Both classes weighted equally
            - Better recall on minority class
            
            **Clean Features**
            - No missing values
            - All categorical variables encoded
            - Numerical features scaled
            
            **Ready for Training**
            - Standard format (CSV)
            - Saved preprocessing artifacts
            - Easy to load and use
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Disadvantages
            
            **Data Loss**
            - Lost 3,305 samples (47% of data)
            - Removed valuable information
            - Smaller training set
            
            **Better Alternatives Exist**
            - SMOTE (synthetic oversampling)
            - Class weights in models
            - Ensemble methods
            
            **Why Naive Undersampling?**
            - Simple and fast
            - Good starting point
            - Can improve later if needed
            """)
        
        st.markdown("---")
        
        # Next Steps
        st.subheader("🚀 Next Steps")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; color: white;">
            <h4 style="margin-top: 0; color: white;">Ready for Model Training!</h4>
            The data is now fully prepared. Next steps:
            <ol>
                <li><strong>Train baseline models</strong> (Logistic Regression, Decision Tree)</li>
                <li><strong>Try ensemble methods</strong> (Random Forest, XGBoost, LightGBM)</li>
                <li><strong>Evaluate performance</strong> (F1-score, ROC-AUC, Confusion Matrix)</li>
                <li><strong>Tune hyperparameters</strong> (GridSearch, RandomSearch)</li>
                <li><strong>Select best model</strong> for deployment</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Data preparation complete! The dataset is ready for machine learning.")


# =============================================================================
# PAGE 9: BASELINE MODELS
# =============================================================================

elif selected_page == "🤖 Baseline Models":
    st.markdown('<p class="main-header">🤖 Baseline Models Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Performance of classical ML models</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ What are Baseline Models?", expanded=True):
        st.markdown("""
        **Baseline models** are simple, classical machine learning algorithms that serve as a 
        starting point for comparison. They help us understand:
        - What level of performance is achievable with simple methods
        - Whether more complex models are needed
        - What metrics we should aim to beat
        
        **Models trained:**
        - **Logistic Regression**: Linear model, fast and interpretable
        - **Decision Tree**: Non-linear model, provides feature importance
        """)
    
    # Check if results exist
    try:
        import json
        with open('baseline_results.json', 'r') as f:
            results = json.load(f)
        
        # Load test data for business impact analysis
        y_test = pd.read_csv('y_test.csv').values.ravel()
        
        results_exist = True
    except FileNotFoundError:
        results_exist = False
        st.error("⚠️ Model results not found! Please run `baseline_models.py` first.")
    
    if results_exist:
        lr_results = results['logistic_regression']
        dt_results = results['decision_tree']
        best_model = results['best_model']
        
        # Best Model Banner
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; color: white; margin: 1rem 0; text-align: center;">
            <h3 style="margin: 0; color: white;">🏆 Best Baseline Model: {best_model}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">Based on F1-Score performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Comparison Metrics
        st.subheader("📊 Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Logistic Regression")
            st.metric("Accuracy", f"{lr_results['accuracy']:.2%}")
            st.metric("Precision", f"{lr_results['precision']:.2%}")
            st.metric("Recall", f"{lr_results['recall']:.2%}")
            st.metric("F1-Score", f"{lr_results['f1_score']:.2%}")
            st.metric("ROC-AUC", f"{lr_results['roc_auc']:.2%}")
            st.metric("Training Time", f"{lr_results['training_time']:.3f}s")
        
        with col2:
            st.markdown("### 🌳 Decision Tree")
            st.metric("Accuracy", f"{dt_results['accuracy']:.2%}")
            st.metric("Precision", f"{dt_results['precision']:.2%}")
            st.metric("Recall", f"{dt_results['recall']:.2%}")
            st.metric("F1-Score", f"{dt_results['f1_score']:.2%}")
            st.metric("ROC-AUC", f"{dt_results['roc_auc']:.2%}")
            st.metric("Training Time", f"{dt_results['training_time']:.3f}s")
        
        st.markdown("---")
        
        # Metrics Explanation
        with st.expander("📖 Understanding the Metrics"):
            st.markdown("""
            **Accuracy**: Percentage of correct predictions (both churned and retained)
            - Good for balanced datasets
            - Can be misleading if classes are imbalanced
            
            **Precision**: Of all customers we predicted would churn, how many actually churned?
            - High precision = Few false alarms
            - Important when acting on predictions is costly
            
            **Recall**: Of all customers who actually churned, how many did we catch?
            - High recall = We catch most churners
            - Important when missing a churner is costly
            
            **F1-Score**: Harmonic mean of Precision and Recall
            - Balances precision and recall
            - Good for comparing models overall
            
            **ROC-AUC**: Area Under the Receiver Operating Characteristic curve
            - Measures model's ability to distinguish between classes
            - Higher is better (max = 1.0)
            """)
        
        st.markdown("---")
        
        # Confusion Matrices
        st.subheader("🎯 Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Logistic Regression")
            cm_lr = np.array(lr_results['confusion_matrix'])
            
            # Create confusion matrix heatmap
            fig_cm_lr = go.Figure(data=go.Heatmap(
                z=cm_lr,
                x=['Predicted: Retained', 'Predicted: Churn'],
                y=['Actual: Retained', 'Actual: Churn'],
                text=cm_lr,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=False
            ))
            fig_cm_lr.update_layout(
                title='Logistic Regression Confusion Matrix',
                height=400,
                xaxis_title='',
                yaxis_title=''
            )
            st.plotly_chart(fig_cm_lr, use_container_width=True)
            
            tn_lr, fp_lr = cm_lr[0]
            fn_lr, tp_lr = cm_lr[1]
            st.markdown(f"""
            <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 0.5rem; color: #1e293b;">
                <strong>True Negatives (TN):</strong> {tn_lr} - Correctly predicted retained<br>
                <strong>False Positives (FP):</strong> {fp_lr} - Incorrectly predicted churn<br>
                <strong>False Negatives (FN):</strong> {fn_lr} - Missed churners<br>
                <strong>True Positives (TP):</strong> {tp_lr} - Correctly predicted churn
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Decision Tree")
            cm_dt = np.array(dt_results['confusion_matrix'])
            
            fig_cm_dt = go.Figure(data=go.Heatmap(
                z=cm_dt,
                x=['Predicted: Retained', 'Predicted: Churn'],
                y=['Actual: Retained', 'Actual: Churn'],
                text=cm_dt,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Greens',
                showscale=False
            ))
            fig_cm_dt.update_layout(
                title='Decision Tree Confusion Matrix',
                height=400,
                xaxis_title='',
                yaxis_title=''
            )
            st.plotly_chart(fig_cm_dt, use_container_width=True)
            
            tn_dt, fp_dt = cm_dt[0]
            fn_dt, tp_dt = cm_dt[1]
            st.markdown(f"""
            <div style="background-color: #f0fdf4; padding: 1rem; border-radius: 0.5rem; color: #1e293b;">
                <strong>True Negatives (TN):</strong> {tn_dt} - Correctly predicted retained<br>
                <strong>False Positives (FP):</strong> {fp_dt} - Incorrectly predicted churn<br>
                <strong>False Negatives (FN):</strong> {fn_dt} - Missed churners<br>
                <strong>True Positives (TP):</strong> {tp_dt} - Correctly predicted churn
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performance Comparison Chart
        st.subheader("📈 Metrics Comparison")
        
        metrics_comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Logistic Regression': [
                lr_results['accuracy'],
                lr_results['precision'],
                lr_results['recall'],
                lr_results['f1_score'],
                lr_results['roc_auc']
            ],
            'Decision Tree': [
                dt_results['accuracy'],
                dt_results['precision'],
                dt_results['recall'],
                dt_results['f1_score'],
                dt_results['roc_auc']
            ]
        })
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Logistic Regression',
            x=metrics_comparison['Metric'],
            y=metrics_comparison['Logistic Regression'],
            marker_color='#667eea',
            text=metrics_comparison['Logistic Regression'].apply(lambda x: f'{x:.3f}'),
            textposition='outside'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Decision Tree',
            x=metrics_comparison['Metric'],
            y=metrics_comparison['Decision Tree'],
            marker_color='#22c55e',
            text=metrics_comparison['Decision Tree'].apply(lambda x: f'{x:.3f}'),
            textposition='outside'
        ))
        
        fig_comparison.update_layout(
            title='Performance Metrics Comparison',
            barmode='group',
            yaxis_title='Score',
            height=450,
            yaxis_range=[0, 1.1]
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance (Decision Tree)
        st.subheader("🔍 Feature Importance (Decision Tree)")
        
        try:
            feature_importance = pd.read_csv('feature_importance_dt.csv')
            
            top_features = feature_importance.head(15)
            
            fig_importance = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='#22c55e',
                text=top_features['importance'].apply(lambda x: f'{x:.4f}'),
                textposition='outside'
            ))
            
            fig_importance.update_layout(
                title='Top 15 Most Important Features',
                xaxis_title='Importance Score',
                yaxis_title='',
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #d1fae5; padding: 1rem; border-radius: 0.5rem; color: #065f46;">
                <strong>💡 Key Insight:</strong><br>
                <strong>Contract type</strong> is by far the most important feature (0.48), followed by 
                <strong>tenure</strong> (0.12) and <strong>TotalCharges</strong> (0.09).
                This confirms our EDA findings that contract type is the #1 predictor of churn!
            </div>
            """, unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.warning("Feature importance data not found.")
        
        st.markdown("---")
        
        # Model Insights
        st.subheader("💡 Key Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Strengths
            
            **Overall Performance**
            - Both models achieve ~75-78% accuracy
            - High ROC-AUC scores (0.83-0.87)
            - Balanced precision and recall
            
            **Logistic Regression**
            - Better overall performance (F1: 0.787)
            - Higher recall (81.3% of churners caught)
            - Very fast training (<0.01s)
            - Interpretable coefficients
            
            **Decision Tree**
            - Provides feature importance
            - Non-linear relationships captured
            - Good interpretability
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Areas for Improvement
            
            **Current Limitations**
            - ~21% of churners still missed (FN)
            - ~12-13% false alarms (FP)
            - Simple models may underfit
            
            **Next Steps**
            - Try ensemble methods (Random Forest)
            - Test gradient boosting (XGBoost, LightGBM)
            - Hyperparameter tuning
            - Feature engineering experiments
            - Cross-validation for robustness
            """)
        
        st.markdown("---")
        
        # Business Impact
        st.subheader("💼 Business Impact Analysis")
        
        total_customers = len(y_test)
        actual_churners = int(cm_lr[1,0] + cm_lr[1,1])
        caught_churners_lr = int(cm_lr[1,1])
        missed_churners_lr = int(cm_lr[1,0])
        false_alarms_lr = int(cm_lr[0,1])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Churners Caught",
                f"{caught_churners_lr}/{actual_churners}",
                delta=f"{(caught_churners_lr/actual_churners*100):.1f}%",
                help="Customers we correctly identified as likely to churn"
            )
        
        with col2:
            st.metric(
                "Churners Missed",
                missed_churners_lr,
                delta=f"-{(missed_churners_lr/actual_churners*100):.1f}%",
                delta_color="inverse",
                help="Churning customers we failed to identify"
            )
        
        with col3:
            st.metric(
                "False Alarms",
                false_alarms_lr,
                help="Retained customers we incorrectly flagged"
            )
        
        st.markdown("""
        <div style="background-color: #fef3c7; padding: 1rem; border-radius: 0.5rem; color: #78350f; margin-top: 1rem;">
            <strong>📊 Using Logistic Regression (Best Model):</strong><br>
            • We can identify and potentially save <strong>81.3% of churning customers</strong><br>
            • 18.7% of churners will still be missed - room for improvement<br>
            • 12.7% false positive rate - acceptable for proactive outreach
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Next Steps
        st.subheader("🚀 Next Steps")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; color: white;">
            <h4 style="margin-top: 0; color: white;">Moving Beyond Baselines</h4>
            Now that we have baseline performance, we can:
            <ol>
                <li><strong>Train ensemble models</strong> (Random Forest, Gradient Boosting)</li>
                <li><strong>Hyperparameter tuning</strong> using GridSearch or RandomSearch</li>
                <li><strong>Feature engineering</strong> to create better predictors</li>
                <li><strong>Cross-validation</strong> for more robust evaluation</li>
                <li><strong>Model deployment</strong> once we select the best performer</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Baseline models trained successfully! Ready to experiment with advanced algorithms.")


# =============================================================================
# PAGE 10: PREDICT CHURN
# =============================================================================

elif selected_page == "🔮 Predict Churn":
    st.markdown('<p class="main-header">🔮 Churn Prediction Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict churn probability for a single customer</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How to use this tool", expanded=True):
        st.markdown("""
        **This tool allows you to predict if a customer is likely to churn.**
        
        1. Enter the customer's information in the form below
        2. Click "Predict Churn Probability"
        3. See the prediction results and recommendations
        
        **The model uses:** Logistic Regression (78.65% F1-Score)
        """)
    
    # Load model and preprocessing artifacts
    try:
        import pickle
        
        with open('logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.error("⚠️ Model files not found! Please train the baseline models first by running `baseline_models.py`")
    
    if model_loaded:
        st.markdown("---")
        st.subheader("👤 Enter Customer Information")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📋 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen (65+)", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        with col2:
            st.markdown("### 📡 Services")
            tenure = st.slider("Tenure (months with company)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            
            if internet_service != "No":
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
        
        with col3:
            st.markdown("### 💰 Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Churn Probability", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'gender': 1 if gender == 'Male' else 0,
                'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
                'Partner': 1 if partner == 'Yes' else 0,
                'Dependents': 1 if dependents == 'Yes' else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone_service == 'Yes' else 0,
                'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract],
                'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
                'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
                'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
                'InternetService_No': 1 if internet_service == 'No' else 0,
                'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
                'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
                'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
                'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
                'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
                'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
                'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
                'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
                'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
                'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
                'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
                'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
                'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
                'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
            }
            
            # Create dataframe with correct feature order
            input_df = pd.DataFrame([input_data])[feature_names]
            
            # Scale numerical features
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            churn_probability = probability[1] * 100
            retain_probability = probability[0] * 100
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Big prediction display
            if churn_probability > 50:
                result_color = "#ef4444"
                result_icon = "⚠️"
                result_text = "HIGH RISK"
                result_message = "This customer is likely to churn!"
            elif churn_probability > 30:
                result_color = "#f59e0b"
                result_icon = "⚡"
                result_text = "MEDIUM RISK"
                result_message = "This customer shows some churn indicators."
            else:
                result_color = "#22c55e"
                result_icon = "✅"
                result_text = "LOW RISK"
                result_message = "This customer is likely to stay."
            
            st.markdown(f"""
            <div style="background: {result_color}; padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin: 1rem 0;">
                <h1 style="margin: 0; color: white; font-size: 3rem;">{result_icon}</h1>
                <h2 style="margin: 0.5rem 0; color: white;">{result_text}</h2>
                <p style="margin: 0; font-size: 1.2rem; opacity: 0.95;">{result_message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability bars
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🟢 Retention Probability",
                    f"{retain_probability:.1f}%",
                    help="Probability that the customer will stay"
                )
                st.progress(retain_probability / 100)
            
            with col2:
                st.metric(
                    "🔴 Churn Probability",
                    f"{churn_probability:.1f}%",
                    help="Probability that the customer will leave"
                )
                st.progress(churn_probability / 100)
            
            st.markdown("---")
            
            # Recommendations based on input
            st.subheader("💡 Recommendations")
            
            recommendations = []
            
            if contract == "Month-to-month":
                recommendations.append({
                    "icon": "📝",
                    "title": "Offer Long-Term Contract",
                    "description": "Month-to-month contracts have 14x higher churn. Offer incentives for 1-2 year contracts.",
                    "priority": "HIGH"
                })
            
            if tenure < 12:
                recommendations.append({
                    "icon": "🎯",
                    "title": "Early Customer Retention Program",
                    "description": "New customers (< 12 months) are most likely to churn. Increase engagement and support.",
                    "priority": "HIGH"
                })
            
            if payment_method == "Electronic check":
                recommendations.append({
                    "icon": "💳",
                    "title": "Encourage Automatic Payments",
                    "description": "Electronic check users churn 2.5x more. Offer discount for automatic payment enrollment.",
                    "priority": "MEDIUM"
                })
            
            if internet_service == "Fiber optic":
                if online_security == "No" or tech_support == "No":
                    recommendations.append({
                        "icon": "🛡️",
                        "title": "Bundle Premium Services",
                        "description": "Fiber customers without add-ons churn more. Offer bundled packages with security and support.",
                        "priority": "MEDIUM"
                    })
            
            if monthly_charges > 70 and tenure < 24:
                recommendations.append({
                    "icon": "💰",
                    "title": "Value Assessment",
                    "description": "High charges + low tenure = higher risk. Consider loyalty discounts or service review.",
                    "priority": "MEDIUM"
                })
            
            if not recommendations:
                recommendations.append({
                    "icon": "✅",
                    "title": "Maintain Current Engagement",
                    "description": "Customer profile looks stable. Continue regular check-ins and service quality.",
                    "priority": "LOW"
                })
            
            for rec in recommendations:
                priority_colors = {
                    "HIGH": "#ef4444",
                    "MEDIUM": "#f59e0b",
                    "LOW": "#22c55e"
                }
                color = priority_colors[rec['priority']]
                
                st.markdown(f"""
                <div style="background-color: #f8fafc; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 0 0.5rem 0.5rem 0;">
                    <strong>{rec['icon']} {rec['title']}</strong>
                    <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 8px;">{rec['priority']}</span>
                    <br>
                    <div style="margin-top: 0.5rem; color: #64748b;">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Risk factors analysis
            st.subheader("🔍 Risk Factors Analysis")
            
            risk_factors = []
            protective_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract (High Risk)")
            else:
                protective_factors.append(f"{contract} contract (Protective)")
            
            if tenure < 12:
                risk_factors.append(f"Low tenure ({tenure} months)")
            elif tenure > 36:
                protective_factors.append(f"High tenure ({tenure} months)")
            
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment")
            elif "automatic" in payment_method.lower():
                protective_factors.append("Automatic payment method")
            
            if internet_service == "Fiber optic":
                risk_factors.append("Fiber optic internet (historically higher churn)")
            
            if online_security == "Yes" or tech_support == "Yes":
                protective_factors.append("Has premium add-on services")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⚠️ Risk Factors")
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- 🔴 {factor}")
                else:
                    st.markdown("- ✅ No major risk factors identified")
            
            with col2:
                st.markdown("### 🛡️ Protective Factors")
                if protective_factors:
                    for factor in protective_factors:
                        st.markdown(f"- 🟢 {factor}")
                else:
                    st.markdown("- ⚠️ Few protective factors present")
            
            st.success("✅ Prediction complete! Use the recommendations above to improve customer retention.")


# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem;">
    Made with ❤️ using Streamlit<br>
    Telco Churn EDA Dashboard
</div>
""", unsafe_allow_html=True)
