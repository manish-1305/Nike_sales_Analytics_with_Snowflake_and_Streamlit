import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Nike Sales Analytics Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

# Function to load data from Snowflake using Snowflake Streamlit connection
@st.cache_data
def load_data_from_snowflake(query):
    """Load data from Snowflake using Streamlit's connection"""
    try:
        # Use Streamlit's connection method for Snowflake Streamlit apps
        conn = st.connection("snowflake")
        df = conn.query(query, ttl=600)  # Cache for 10 minutes
        
        # Ensure proper data types
        if 'INVOICE_DATE' in df.columns:
            df['INVOICE_DATE'] = pd.to_datetime(df['INVOICE_DATE'])
        elif 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to get table info from Snowflake
@st.cache_data
def get_table_info(table_name):
    """Get table structure information"""
    query = f"DESCRIBE TABLE {table_name}"
    try:
        conn = st.connection("snowflake")
        df = conn.query(query, ttl=3600)  # Cache for 1 hour
        return df
    except Exception as e:
        st.error(f"Error getting table info: {str(e)}")
        return None

# Advanced Analytics Functions
def calculate_rfm_analysis(df):
    """Calculate RFM (Recency, Frequency, Monetary) analysis"""
    try:
        # Get the most recent date in the dataset
        max_date = df['Invoice Date'].max()
        
        # Calculate RFM metrics by retailer
        rfm = df.groupby('Retailer').agg({
            'Invoice Date': lambda x: (max_date - x.max()).days,  # Recency
            'Total Sales': ['count', 'sum']  # Frequency and Monetary
        }).round(2)
        
        rfm.columns = ['Recency_Days', 'Frequency_Orders', 'Monetary_Value']
        rfm = rfm.reset_index()
        
        # Create RFM scores (1-5 scale) with better handling of duplicate edges
        try:
            # For Recency: Lower recency days = higher score (more recent = better)
            rfm['R_Score'] = pd.qcut(rfm['Recency_Days'], 5, labels=[5,4,3,2,1], duplicates='drop')
        except ValueError:
            # If qcut fails due to duplicate values, use rank-based approach
            rfm['R_Score'] = pd.cut(rfm['Recency_Days'].rank(method='first'), 5, labels=[5,4,3,2,1])
        
        try:
            # For Frequency: Higher frequency = higher score
            rfm['F_Score'] = pd.qcut(rfm['Frequency_Orders'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            # Fallback to rank-based approach
            rfm['F_Score'] = pd.cut(rfm['Frequency_Orders'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        try:
            # For Monetary: Higher monetary value = higher score
            rfm['M_Score'] = pd.qcut(rfm['Monetary_Value'], 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            # Fallback to rank-based approach
            rfm['M_Score'] = pd.cut(rfm['Monetary_Value'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Handle potential None values from qcut/cut operations
        rfm['R_Score'] = rfm['R_Score'].fillna(3)  # Default to middle score
        rfm['F_Score'] = rfm['F_Score'].fillna(3)
        rfm['M_Score'] = rfm['M_Score'].fillna(3)
        
        # Combined RFM Score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Add RFM segment interpretation
        def rfm_segment(row):
            score = row['RFM_Score']
            
            # High-value segments
            if score in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif score in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif score in ['520', '530', '521', '531', '540', '541']:
                return 'New Customers'
            elif score in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif score in ['155', '254', '245', '253', '254', '245']:
                return 'Cannot Lose Them'
            elif score in ['331', '321', '312', '231', '241', '251']:
                return 'Hibernating'
            else:
                return 'Others'
        
        rfm['RFM_Segment'] = rfm.apply(rfm_segment, axis=1)
        
        return rfm
    except Exception as e:
        st.error(f"Error calculating RFM analysis: {str(e)}")
        return None

def calculate_market_basket_analysis(df):
    """Calculate market basket analysis - products frequently bought together"""
    try:
        # Group by invoice date and retailer to find co-occurring products
        basket = df.groupby(['Invoice Date', 'Retailer'])['Product'].apply(list).reset_index()
        
        # Count product combinations
        product_combinations = {}
        for products in basket['Product']:
            if len(products) > 1:
                products = sorted(products)
                for i in range(len(products)):
                    for j in range(i+1, len(products)):
                        combo = f"{products[i]} + {products[j]}"
                        product_combinations[combo] = product_combinations.get(combo, 0) + 1
        
        # Convert to DataFrame
        if product_combinations:
            basket_df = pd.DataFrame(list(product_combinations.items()), 
                                   columns=['Product_Combination', 'Frequency'])
            basket_df = basket_df.sort_values('Frequency', ascending=False).head(10)
            return basket_df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error calculating market basket analysis: {str(e)}")
        return pd.DataFrame()

def calculate_sales_velocity(df):
    """Calculate sales velocity metrics"""
    try:
        # Calculate daily sales velocity by product
        velocity = df.groupby(['Product', 'Invoice Date'])['Units Sold'].sum().reset_index()
        velocity_metrics = velocity.groupby('Product')['Units Sold'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        velocity_metrics.columns = ['Avg_Daily_Units', 'Std_Daily_Units', 'Min_Daily_Units', 'Max_Daily_Units']
        velocity_metrics['Velocity_Consistency'] = (velocity_metrics['Avg_Daily_Units'] / 
                                                   (velocity_metrics['Std_Daily_Units'] + 0.01)).round(2)
        return velocity_metrics.reset_index()
    except Exception as e:
        st.error(f"Error calculating sales velocity: {str(e)}")
        return pd.DataFrame()

def calculate_seasonal_patterns(df):
    """Calculate seasonal sales patterns"""
    try:
        df['Month'] = df['Invoice Date'].dt.month
        df['Quarter'] = df['Invoice Date'].dt.quarter
        df['Day_of_Week'] = df['Invoice Date'].dt.dayofweek
        df['Week_of_Year'] = df['Invoice Date'].dt.isocalendar().week
        
        # Monthly patterns
        monthly_pattern = df.groupby('Month')['Total Sales'].sum().reset_index()
        monthly_pattern['Month_Name'] = monthly_pattern['Month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        # Day of week patterns
        dow_pattern = df.groupby('Day_of_Week')['Total Sales'].sum().reset_index()
        dow_pattern['Day_Name'] = dow_pattern['Day_of_Week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        return monthly_pattern, dow_pattern
    except Exception as e:
        st.error(f"Error calculating seasonal patterns: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_customer_lifetime_value(df):
    """Calculate Customer Lifetime Value (CLV) by retailer"""
    try:
        # Calculate CLV components
        clv_metrics = df.groupby('Retailer').agg({
            'Total Sales': ['sum', 'mean', 'count'],
            'Invoice Date': ['min', 'max']
        })
        
        clv_metrics.columns = ['Total_Revenue', 'Avg_Order_Value', 'Purchase_Frequency', 'First_Purchase', 'Last_Purchase']
        clv_metrics['Customer_Lifespan_Days'] = (clv_metrics['Last_Purchase'] - clv_metrics['First_Purchase']).dt.days
        clv_metrics['CLV_Score'] = (clv_metrics['Avg_Order_Value'] * 
                                   clv_metrics['Purchase_Frequency'] * 
                                   (clv_metrics['Customer_Lifespan_Days'] / 365)).round(2)
        
        return clv_metrics.reset_index()
    except Exception as e:
        st.error(f"Error calculating CLV: {str(e)}")
        return pd.DataFrame()

def calculate_abc_analysis(df):
    """Calculate ABC analysis for inventory management"""
    try:
        # Calculate cumulative sales percentage by product
        product_sales = df.groupby('Product')['Total Sales'].sum().reset_index()
        product_sales = product_sales.sort_values('Total Sales', ascending=False)
        product_sales['Cumulative_Sales'] = product_sales['Total Sales'].cumsum()
        product_sales['Cumulative_Percentage'] = (product_sales['Cumulative_Sales'] / 
                                                 product_sales['Total Sales'].sum() * 100).round(2)
        
        # Classify products into A, B, C categories
        product_sales['ABC_Category'] = pd.cut(product_sales['Cumulative_Percentage'], 
                                             bins=[0, 80, 95, 100], 
                                             labels=['A', 'B', 'C'])
        
        return product_sales
    except Exception as e:
        st.error(f"Error calculating ABC analysis: {str(e)}")
        return pd.DataFrame()

# Main app
def main():
    st.markdown('<h1 class="main-header">👟 Nike Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Snowflake table configuration
    st.sidebar.header("⚙️ Database Configuration")
    
    # Pre-configured for your specific table
    database_name = st.sidebar.text_input("Database", value="SF_PROJECT", help="Snowflake database name")
    schema_name = st.sidebar.text_input("Schema", value="NIKE", help="Snowflake schema name")
    table_name = st.sidebar.text_input("Table", value="NIKE_SALES", help="Snowflake table name")
    
    # Full table path
    full_table_name = f'"{database_name}"."{schema_name}"."{table_name}"'
    st.sidebar.info(f"📍 Full path: {full_table_name}")
    
    # Test connection button
    if st.sidebar.button("🔍 Test Connection"):
        try:
            conn = st.connection("snowflake")
            test_query = f"SELECT COUNT(*) as record_count FROM {full_table_name}"
            result = conn.query(test_query)
            st.sidebar.success("✅ Connected to Snowflake successfully!")
            st.sidebar.info(f"📊 Found {result.iloc[0]['RECORD_COUNT']:,} records")
            
            # Show table info
            table_info = get_table_info(full_table_name)
            if table_info is not None:
                st.sidebar.success(f"✅ Table '{full_table_name}' found!")
                with st.sidebar.expander("📋 Table Structure"):
                    st.dataframe(table_info[['name', 'type']], use_container_width=True)
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
    
    # Load data from Snowflake
    base_query = f"""
    SELECT 
        "INVOICE_DATE" as "Invoice Date",
        "PRODUCT" as "Product",
        "REGION" as "Region", 
        "RETAILER" as "Retailer",
        "SALES_METHOD" as "Sales Method",
        "STATE" as "State",
        "PRICE_PER_UNIT" as "Price per Unit",
        "TOTAL_SALES" as "Total Sales",
        "UNITS_SOLD" as "Units Sold"
    FROM {full_table_name}
    ORDER BY "INVOICE_DATE" DESC
    """
    
    # Load initial data
    df = load_data_from_snowflake(base_query)
    
    if df is None or df.empty:
        st.error("❌ No data found. Please check your table path and permissions.")
        st.info("📋 Expected table structure for SF_PROJECT.NIKE.NIKE_SALES:")
        st.code("""
        Column Names (case-insensitive):
        - INVOICE_DATE (DATE/TIMESTAMP)
        - PRODUCT (VARCHAR)
        - REGION (VARCHAR)
        - RETAILER (VARCHAR)
        - SALES_METHOD (VARCHAR)
        - STATE (VARCHAR)
        - PRICE_PER_UNIT (NUMBER/DECIMAL)
        - TOTAL_SALES (NUMBER/DECIMAL)
        - UNITS_SOLD (NUMBER/INTEGER)
        """)
        
        # Show troubleshooting for Snowflake Streamlit
        st.error("🔧 Troubleshooting Steps for Snowflake Streamlit:")
        st.markdown("""
        1. **Verify table exists and has data**:
           ```sql
           SELECT COUNT(*) FROM SF_PROJECT.NIKE.NIKE_SALES;
           ```
        
        2. **Check permissions**:
           - Ensure your Snowflake role has access to SF_PROJECT database
           - Verify SELECT privileges on NIKE schema and NIKE_SALES table
        
        3. **Verify column names**:
           ```sql
           DESCRIBE TABLE SF_PROJECT.NIKE.NIKE_SALES;
           ```
        """)
        return
    
    st.success(f"✅ Loaded {len(df):,} records from Snowflake")
    
    # Handle different possible column name formats
    column_mapping = {
        'INVOICE_DATE': 'Invoice Date',
        'PRODUCT': 'Product',
        'REGION': 'Region',
        'RETAILER': 'Retailer', 
        'SALES_METHOD': 'Sales Method',
        'STATE': 'State',
        'PRICE_PER_UNIT': 'Price per Unit',
        'TOTAL_SALES': 'Total Sales',
        'UNITS_SOLD': 'Units Sold'
    }
    
    # Rename columns if they're in uppercase
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Ensure Invoice Date is datetime
    if 'Invoice Date' in df.columns:
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
    
    # Display data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Custom query option
    st.sidebar.header("🔧 Advanced Options")
    use_custom_query = st.sidebar.checkbox("Use Custom SQL Query")
    
    if use_custom_query:
        custom_query = st.sidebar.text_area(
            "Custom SQL Query",
            value=base_query,
            height=150,
            help="Write your custom SQL query to filter data at the database level"
        )
        
        if st.sidebar.button("🔄 Execute Query"):
            df = load_data_from_snowflake(custom_query)
            if df is not None:
                st.success(f"✅ Custom query executed successfully! Loaded {len(df):,} records")
                # Apply column renaming for custom queries too
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})
                if 'Invoice Date' in df.columns:
                    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
            else:
                st.error("❌ Custom query failed. Please check your SQL syntax.")
    
    # Quick data insights
    if df is not None and not df.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Quick Insights")
        st.sidebar.metric("Total Records", f"{len(df):,}")
        st.sidebar.metric("Date Range", f"{df['Invoice Date'].min().strftime('%Y-%m-%d')} to {df['Invoice Date'].max().strftime('%Y-%m-%d')}")
        st.sidebar.metric("Unique Products", f"{df['Product'].nunique():,}")
        st.sidebar.metric("Total Sales", f"${df['Total Sales'].sum():,.2f}")
    
    # Sidebar filters
    st.sidebar.header("🎯 Filter Options")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Invoice Date'].min(), df['Invoice Date'].max()),
        min_value=df['Invoice Date'].min(),
        max_value=df['Invoice Date'].max()
    )
    
    # Multi-select filters
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=df['Product'].unique(),
        default=df['Product'].unique()
    )
    
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    selected_retailers = st.sidebar.multiselect(
        "Select Retailers",
        options=df['Retailer'].unique(),
        default=df['Retailer'].unique()
    )
    
    selected_sales_methods = st.sidebar.multiselect(
        "Select Sales Methods",
        options=df['Sales Method'].unique(),
        default=df['Sales Method'].unique()
    )
    
    selected_states = st.sidebar.multiselect(
        "Select States",
        options=df['State'].unique(),
        default=df['State'].unique()
    )
    
    # Apply filters
    if len(date_range) == 2:
        df_filtered = df[
            (df['Invoice Date'] >= pd.to_datetime(date_range[0])) &
            (df['Invoice Date'] <= pd.to_datetime(date_range[1])) &
            (df['Product'].isin(selected_products)) &
            (df['Region'].isin(selected_regions)) &
            (df['Retailer'].isin(selected_retailers)) &
            (df['Sales Method'].isin(selected_sales_methods)) &
            (df['State'].isin(selected_states))
        ]
    else:
        df_filtered = df[
            (df['Product'].isin(selected_products)) &
            (df['Region'].isin(selected_regions)) &
            (df['Retailer'].isin(selected_retailers)) &
            (df['Sales Method'].isin(selected_sales_methods)) &
            (df['State'].isin(selected_states))
        ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df_filtered['Total Sales'].sum()
        st.metric("💰 Total Sales", f"${total_sales:,.2f}")
    
    with col2:
        total_units = df_filtered['Units Sold'].sum()
        st.metric("📦 Total Units Sold", f"{total_units:,}")
    
    with col3:
        avg_price = df_filtered['Price per Unit'].mean()
        st.metric("💵 Average Price", f"${avg_price:.2f}")
    
    with col4:
        total_orders = len(df_filtered)
        st.metric("📋 Total Orders", f"{total_orders:,}")
    
    # Visualization options
    st.header("📊 Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Sales Trends", "🌍 Regional Analysis", "🏪 Retailer Performance", "👕 Product Analysis", "📱 Sales Method"])
    
    with tab1:
        st.subheader("Sales Trends Over Time")
        
        # Time series chart
        daily_sales = df_filtered.groupby('Invoice Date')['Total Sales'].sum().reset_index()
        fig_trend = px.line(
            daily_sales, 
            x='Invoice Date', 
            y='Total Sales',
            title='Daily Sales Trend',
            color_discrete_sequence=['#FF6B35']
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Monthly sales
        df_filtered['Month'] = df_filtered['Invoice Date'].dt.to_period('M')
        monthly_sales = df_filtered.groupby('Month')['Total Sales'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['Month'].astype(str)
        
        fig_monthly = px.bar(
            monthly_sales, 
            x='Month', 
            y='Total Sales',
            title='Monthly Sales Performance',
            color='Total Sales',
            color_continuous_scale='viridis'
        )
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab2:
        st.subheader("Regional Sales Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional pie chart
            regional_sales = df_filtered.groupby('Region')['Total Sales'].sum().reset_index()
            fig_pie = px.pie(
                regional_sales, 
                values='Total Sales', 
                names='Region',
                title='Sales Distribution by Region'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # State-wise sales
            state_sales = df_filtered.groupby('State')['Total Sales'].sum().reset_index().sort_values('Total Sales', ascending=False).head(10)
            fig_state = px.bar(
                state_sales, 
                x='State', 
                y='Total Sales',
                title='Top 10 States by Sales',
                color='Total Sales',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_state, use_container_width=True)
    
    with tab3:
        st.subheader("Retailer Performance Analysis")
        
        retailer_metrics = df_filtered.groupby('Retailer').agg({
            'Total Sales': 'sum',
            'Units Sold': 'sum',
            'Price per Unit': 'mean'
        }).reset_index()
        
        # Retailer sales comparison
        fig_retailer = px.bar(
            retailer_metrics, 
            x='Retailer', 
            y='Total Sales',
            title='Sales by Retailer',
            color='Total Sales',
            color_continuous_scale='oranges'
        )
        fig_retailer.update_xaxes(tickangle=45)
        st.plotly_chart(fig_retailer, use_container_width=True)
        
        # Retailer performance table
        st.subheader("Retailer Performance Table")
        retailer_metrics['Average Price'] = retailer_metrics['Price per Unit'].round(2)
        retailer_metrics = retailer_metrics.drop('Price per Unit', axis=1)
        retailer_metrics['Total Sales'] = retailer_metrics['Total Sales'].round(2)
        st.dataframe(retailer_metrics.sort_values('Total Sales', ascending=False), use_container_width=True)
    
    with tab4:
        st.subheader("Product Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top products by sales
            product_sales = df_filtered.groupby('Product')['Total Sales'].sum().reset_index().sort_values('Total Sales', ascending=False)
            fig_product = px.bar(
                product_sales, 
                x='Product', 
                y='Total Sales',
                title='Product Sales Performance',
                color='Total Sales',
                color_continuous_scale='greens'
            )
            fig_product.update_xaxes(tickangle=45)
            st.plotly_chart(fig_product, use_container_width=True)
        
        with col2:
            # Units sold by product
            product_units = df_filtered.groupby('Product')['Units Sold'].sum().reset_index().sort_values('Units Sold', ascending=False)
            fig_units = px.bar(
                product_units, 
                x='Product', 
                y='Units Sold',
                title='Units Sold by Product',
                color='Units Sold',
                color_continuous_scale='reds'
            )
            fig_units.update_xaxes(tickangle=45)
            st.plotly_chart(fig_units, use_container_width=True)
    
    with tab5:
        st.subheader("Sales Method Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales method pie chart
            method_sales = df_filtered.groupby('Sales Method')['Total Sales'].sum().reset_index()
            fig_method_pie = px.pie(
                method_sales, 
                values='Total Sales', 
                names='Sales Method',
                title='Sales Distribution by Method'
            )
            st.plotly_chart(fig_method_pie, use_container_width=True)
        
        with col2:
            # Sales method performance over time
            method_time = df_filtered.groupby(['Invoice Date', 'Sales Method'])['Total Sales'].sum().reset_index()
            fig_method_time = px.line(
                method_time, 
                x='Invoice Date', 
                y='Total Sales',
                color='Sales Method',
                title='Sales Method Performance Over Time'
            )
            st.plotly_chart(fig_method_time, use_container_width=True)
    
    # Advanced Analytics - Replaced correlation with multiple business metrics
    st.header("🔍 Advanced Business Analytics")
    
    # Create tabs for different advanced analytics
    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "📊 RFM Analysis", "🛒 Market Basket", "📈 Sales Velocity", "⏱️ Seasonal Patterns"
    ])
    
    with analytics_tab1:
        st.subheader("RFM Analysis (Recency, Frequency, Monetary)")
        
        rfm_data = calculate_rfm_analysis(df_filtered)
        if rfm_data is not None and not rfm_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # RFM Score Distribution
                fig_rfm = px.scatter(
                    rfm_data, 
                    x='Frequency_Orders', 
                    y='Monetary_Value',
                    size='Recency_Days',
                    color='RFM_Score',
                    hover_data=['Retailer'],
                    title='RFM Analysis - Retailer Segmentation'
                )
                st.plotly_chart(fig_rfm, use_container_width=True)
            
            with col2:
                # Top retailers by RFM components
                fig_rfm_bar = px.bar(
                    rfm_data.sort_values('Monetary_Value', ascending=False).head(10),
                    x='Retailer',
                    y='Monetary_Value',
                    color='Frequency_Orders',
                    title='Top 10 Retailers by Monetary Value'
                )
                fig_rfm_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_rfm_bar, use_container_width=True)
            
            # RFM Table
            st.subheader("RFM Analysis Table")
            st.dataframe(rfm_data.sort_values('Monetary_Value', ascending=False), use_container_width=True)
    
    with analytics_tab2:
        st.subheader("Market Basket Analysis")
        
        basket_data = calculate_market_basket_analysis(df_filtered)
        if not basket_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Product combinations chart
                fig_basket = px.bar(
                    basket_data.head(10),
                    x='Frequency',
                    y='Product_Combination',
                    orientation='h',
                    title='Top 10 Product Combinations',
                    color='Frequency',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_basket, use_container_width=True)
            
            with col2:
                # Cross-selling opportunities
                st.subheader("Cross-Selling Insights")
                if len(basket_data) > 0:
                    st.write("**Most Frequent Product Combinations:**")
                    for idx, row in basket_data.head(5).iterrows():
                        st.write(f"• {row['Product_Combination']}: {row['Frequency']} times")
                else:
                    st.write("No product combinations found in the filtered data.")
            
            # Basket analysis table
            st.subheader("Market Basket Analysis Table")
            st.dataframe(basket_data, use_container_width=True)
        else:
            st.info("No product combinations found in the filtered data.")
    
    with analytics_tab3:
        st.subheader("Sales Velocity Analysis")
        
        velocity_data = calculate_sales_velocity(df_filtered)
        if not velocity_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sales velocity chart
                fig_velocity = px.scatter(
                    velocity_data,
                    x='Avg_Daily_Units',
                    y='Velocity_Consistency',
                    size='Max_Daily_Units',
                    color='Product',
                    title='Sales Velocity vs Consistency',
                    hover_data=['Min_Daily_Units', 'Max_Daily_Units']
                )
                st.plotly_chart(fig_velocity, use_container_width=True)
            
            with col2:
                # Top products by velocity
                fig_velocity_bar = px.bar(
                    velocity_data.sort_values('Avg_Daily_Units', ascending=False).head(10),
                    x='Product',
                    y='Avg_Daily_Units',
                    color='Velocity_Consistency',
                    title='Top 10 Products by Average Daily Units'
                )
                fig_velocity_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_velocity_bar, use_container_width=True)
            
            # Velocity analysis table
            st.subheader("Sales Velocity Table")
            st.dataframe(velocity_data.sort_values('Avg_Daily_Units', ascending=False), use_container_width=True)
        else:
            st.info("Unable to calculate sales velocity with current data.")
    
    with analytics_tab4:
        st.subheader("Seasonal Patterns Analysis")
        
        monthly_pattern, dow_pattern = calculate_seasonal_patterns(df_filtered.copy())
        
        if not monthly_pattern.empty and not dow_pattern.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly seasonality
                fig_monthly_pattern = px.line(
                    monthly_pattern,
                    x='Month_Name',
                    y='Total Sales',
                    title='Monthly Sales Seasonality',
                    markers=True
                )
                st.plotly_chart(fig_monthly_pattern, use_container_width=True)
            
            with col2:
                # Day of week patterns
                fig_dow_pattern = px.bar(
                    dow_pattern,
                    x='Day_Name',
                    y='Total Sales',
                    title='Day of Week Sales Pattern',
                    color='Total Sales',
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig_dow_pattern, use_container_width=True)
            
            # Seasonal insights
            st.subheader("Seasonal Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                best_month = monthly_pattern.loc[monthly_pattern['Total Sales'].idxmax(), 'Month_Name']
                worst_month = monthly_pattern.loc[monthly_pattern['Total Sales'].idxmin(), 'Month_Name']
                st.metric("Best Month", best_month)
                st.metric("Worst Month", worst_month)
            
            with col2:
                best_day = dow_pattern.loc[dow_pattern['Total Sales'].idxmax(), 'Day_Name']
                worst_day = dow_pattern.loc[dow_pattern['Total Sales'].idxmin(), 'Day_Name']
                st.metric("Best Day", best_day)
                st.metric("Worst Day", worst_day)
        else:
            st.info("Unable to calculate seasonal patterns with current data.")
    
    # Additional Business Metrics
    st.header("💼 Business Intelligence Metrics")
    
    business_col1, business_col2 = st.columns(2)
    
    with business_col1:
        st.subheader("Customer Lifetime Value (CLV)")
        
        clv_data = calculate_customer_lifetime_value(df_filtered)
        if not clv_data.empty:
            # CLV visualization
            fig_clv = px.scatter(
                clv_data,
                x='Purchase_Frequency',
                y='CLV_Score',
                size='Total_Revenue',
                color='Avg_Order_Value',
                hover_data=['Retailer'],
                title='Customer Lifetime Value Analysis'
            )
            st.plotly_chart(fig_clv, use_container_width=True)
            
            # CLV table
            st.dataframe(clv_data.sort_values('CLV_Score', ascending=False).head(10), use_container_width=True)
        else:
            st.info("Unable to calculate CLV with current data.")
    
    with business_col2:
        st.subheader("ABC Analysis (Inventory)")
        
        abc_data = calculate_abc_analysis(df_filtered)
        if not abc_data.empty:
            # ABC category distribution
            abc_summary = abc_data.groupby('ABC_Category').agg({
                'Product': 'count',
                'Total Sales': 'sum'
            }).reset_index()
            abc_summary.columns = ['Category', 'Product_Count', 'Total_Sales']
            
            fig_abc = px.pie(
                abc_summary,
                values='Total_Sales',
                names='Category',
                title='ABC Analysis - Sales Distribution',
                color_discrete_map={'A': '#FF6B35', 'B': '#F7931E', 'C': '#FFD700'}
            )
            st.plotly_chart(fig_abc, use_container_width=True)
            
            # ABC insights
            st.subheader("ABC Category Insights")
            for category in ['A', 'B', 'C']:
                cat_data = abc_summary[abc_summary['Category'] == category]
                if not cat_data.empty:
                    st.write(f"**Category {category}:** {cat_data.iloc[0]['Product_Count']} products, ${cat_data.iloc[0]['Total_Sales']:,.2f} sales")
        else:
            st.info("Unable to calculate ABC analysis with current data.")
    
    # Performance Benchmarking
    st.header("📏 Performance Benchmarking")
    
    benchmark_col1, benchmark_col2 = st.columns(2)
    
    with benchmark_col1:
        st.subheader("Top vs Bottom Performers")
        
        # Product performance comparison
        product_performance = df_filtered.groupby('Product')['Total Sales'].sum().reset_index()
        top_products = product_performance.nlargest(5, 'Total Sales')
        bottom_products = product_performance.nsmallest(5, 'Total Sales')
        
        performance_comparison = pd.concat([
            top_products.assign(Performance='Top 5'),
            bottom_products.assign(Performance='Bottom 5')
        ])
        
        fig_performance = px.bar(
            performance_comparison,
            x='Product',
            y='Total Sales',
            color='Performance',
            title='Top vs Bottom Product Performance',
            color_discrete_map={'Top 5': '#00CC96', 'Bottom 5': '#EF553B'}
        )
        fig_performance.update_xaxes(tickangle=45)
        st.plotly_chart(fig_performance, use_container_width=True)
    
    with benchmark_col2:
        st.subheader("Regional Performance Matrix")
        
        # Region vs Sales Method performance
        region_method_performance = df_filtered.groupby(['Region', 'Sales Method'])['Total Sales'].sum().reset_index()
        
        fig_heatmap = px.density_heatmap(
            region_method_performance,
            x='Sales Method',
            y='Region',
            z='Total Sales',
            title='Regional vs Sales Method Performance Heatmap',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Growth Analysis
    st.header("📈 Growth Analysis")
    
    growth_col1, growth_col2 = st.columns(2)
    
    with growth_col1:
        st.subheader("Month-over-Month Growth")
        
        # Calculate MoM growth
        df_growth = df_filtered.copy()
        df_growth['Year_Month'] = df_growth['Invoice Date'].dt.to_period('M')
        monthly_growth = df_growth.groupby('Year_Month')['Total Sales'].sum().reset_index()
        monthly_growth['Year_Month'] = monthly_growth['Year_Month'].astype(str)
        monthly_growth['MoM_Growth'] = monthly_growth['Total Sales'].pct_change() * 100
        
        fig_growth = px.line(
            monthly_growth,
            x='Year_Month',
            y='MoM_Growth',
            title='Month-over-Month Growth Rate (%)',
            markers=True
        )
        fig_growth.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with growth_col2:
        st.subheader("Product Growth Trends")
        
        # Product growth comparison
        product_growth = df_filtered.groupby(['Product', df_filtered['Invoice Date'].dt.to_period('M')])['Total Sales'].sum().reset_index()
        product_growth['Year_Month'] = product_growth['Invoice Date'].astype(str)
        
        # Focus on top 5 products for clarity
        top_5_products = df_filtered.groupby('Product')['Total Sales'].sum().nlargest(5).index.tolist()
        product_growth_filtered = product_growth[product_growth['Product'].isin(top_5_products)]
        
        fig_product_growth = px.line(
            product_growth_filtered,
            x='Year_Month',
            y='Total Sales',
            color='Product',
            title='Top 5 Products Growth Trends',
            markers=True
        )
        st.plotly_chart(fig_product_growth, use_container_width=True)
    
    # Data table with pagination
    st.header("📋 Filtered Data")
    
    # Add pagination for large datasets
    page_size_options = [10, 25, 50, 100]
    page_size = st.selectbox("Records per page", page_size_options, index=1)
    total_pages = (len(df_filtered) - 1) // page_size + 1
    
    if total_pages > 1:
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        displayed_df = df_filtered.sort_values('Invoice Date', ascending=False).iloc[start_idx:end_idx]
        st.info(f"Showing page {page_number} of {total_pages} ({len(df_filtered):,} total records)")
    else:
        displayed_df = df_filtered.sort_values('Invoice Date', ascending=False)
    
    st.dataframe(displayed_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download filtered data
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"nike_sales_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download summary report
        summary_data = {
            'Metric': ['Total Sales', 'Total Units', 'Average Price', 'Total Orders', 'Unique Products', 'Unique Retailers'],
            'Value': [
                f"${df_filtered['Total Sales'].sum():,.2f}",
                f"{df_filtered['Units Sold'].sum():,}",
                f"${df_filtered['Price per Unit'].mean():.2f}",
                f"{len(df_filtered):,}",
                f"{df_filtered['Product'].nunique():,}",
                f"{df_filtered['Retailer'].nunique():,}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Report",
            data=summary_csv,
            file_name=f"nike_sales_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export to Snowflake (optional - create a new table with filtered data)
        if st.button("💾 Save Filtered Data to Snowflake"):
            new_table_name = f'"SF_PROJECT"."NIKE"."NIKE_SALES_FILTERED_{datetime.now().strftime("%Y%m%d_%H%M%S")}"'
            try:
                conn = st.connection("snowflake")
                if conn:
                    # Convert column names back to uppercase for Snowflake
                    df_export = df_filtered.copy()
                    df_export.columns = [col.upper().replace(' ', '_') for col in df_export.columns]
                    
                    # Use Snowflake connector to write data
                    conn.create_table(new_table_name, df_export, if_not_exists=True)
                    st.success(f"✅ Data exported to table: {new_table_name}")
                    st.info(f"📊 {len(df_export):,} rows exported")
                else:
                    st.error("❌ Failed to connect to Snowflake")
            except Exception as e:
                st.error(f"Error exporting to Snowflake: {str(e)}")
    
    # Performance metrics
    st.header("⚡ Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("🔍 Filtered Records", f"{len(df_filtered):,}")
    
    with col3:
        filter_percentage = (len(df_filtered) / len(df)) * 100 if len(df) > 0 else 0
        st.metric("📈 Filter Coverage", f"{filter_percentage:.1f}%")

if __name__ == "__main__":
    main()