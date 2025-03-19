import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Cornerstone A&D Design Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to style the sidebar and elements for better dark mode compatibility
sidebar_style = """
<style>
    [data-testid="stSidebar"] {
        background-color: #1E2A45;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stSelectbox span {
        color: #E0E0E0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #2C3B5A;
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] .stSuccess {
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Milestone box styles with dark mode compatibility */
    .milestone-box {
        border: 1px solid #4A4A4A;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        background-color: #2C3B5A;
    }
    .milestone-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #E0E0E0;
    }
    .milestone-counts {
        display: flex;
        justify-content: space-between;
    }
    .forecast-box {
        background-color: #1E3A5F;
        padding: 10px;
        border-radius: 3px;
        flex: 1;
        margin-right: 2px;
        color: #E0E0E0;
    }
    .actual-box {
        background-color: #2C5F2E;
        padding: 10px;
        border-radius: 3px;
        flex: 1;
        margin-left: 2px;
        color: #E0E0E0;
    }
    .count-number {
        font-size: 1.5em;
        font-weight: bold;
        color: #E0E0E0;
    }
    .count-label {
        font-size: 0.8em;
        color: #B0B0B0;
    }
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Sidebar header with sticky positioning
with st.sidebar.container():
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=150, caption="Cornerstone A&D")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# Title and description
st.title("Cornerstone A&D Design Dashboard")
st.markdown("This dashboard provides insights into the Cornerstone A&D Design Master Tracker data.")

# Load data
@st.cache_data
def load_data():
    notifications = []  # Collect notifications here
    try:
        # Load main dataset without any filtering
        main_df = pd.read_csv("Cornerstone - A&D - Design Master Tracker.csv", low_memory=False)
        
        # Clean column names by stripping whitespace
        main_df.columns = main_df.columns.str.strip()
        
        # Clean KTL Project Name data
        if 'KTL Project Name' in main_df.columns:
            # Store original count
            original_count = len(main_df)
            
            # Replace "ICSS Batch 1" with "ICSS"
            main_df['KTL Project Name'] = main_df['KTL Project Name'].replace('ICSS Batch 1', 'ICSS')
            
            # Remove TEST projects
            main_df = main_df[~main_df['KTL Project Name'].str.contains('TEST', case=False, na=False)]
            
            # Report cleaning results
            removed_count = original_count - len(main_df)
            if removed_count > 0:
                notifications.append(f"ℹ️ Removed {removed_count} TEST projects from the dataset")
        
        # Report initial row count
        initial_total = len(main_df)
        notifications.append(f"✅ Loaded dataset with {initial_total} rows and {len(main_df.columns)} columns")
        
        # Debug information about columns
        notifications.append(f"ℹ️ Available columns: {', '.join(main_df.columns)}")
        
        # Basic data info
        if 'Site ID' in main_df.columns:
            unique_sites = main_df['Site ID'].nunique()
            notifications.append(f"ℹ️ Found {unique_sites} unique Site IDs")
        
        if 'Internal ID' in main_df.columns:
            valid_ids = main_df['Internal ID'].notna().sum()
            notifications.append(f"ℹ️ Found {valid_ids} records with Internal ID")
        
        return main_df, notifications

    except Exception as e:
        notifications.append(f"❌ Error loading data: {e}")
        return None, notifications

# Clean and prepare data
@st.cache_data
def prepare_data(df):
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Explicitly define date columns that need conversion
    date_columns = [
        "RAMS\n(A)",
        "Survey\n(F)",
        "Survey (A)",
        "DRONE SURVEY DATE (F)",
        "Drone Survey (A)",
        "RMSV DATE (F)",
        "RMSV(A)",
        "GA\n(F)",
        "GA Issued Client (A)",
        "GA Approved Client (A)",
        "GA Issued Operator (A)",
        "GA Approved Operator (A)",
        "DD\n(F)",
        "Dependency\n(F)",
        "DD Issued Client (A)",
        "DD Approved Client (A)",
        "DD Issued to Operator (A)",
        "DD Approved to Operator (A)",
        "Instruction Date"
    ]
    
    # Additional date columns from the dataset that might contain dates
    additional_date_columns = [col for col in data.columns 
                              if any(word in col.lower() for word in 
                                    ['date', 'issued', 'approved', 'instruction']) 
                              and col not in date_columns]
    
    # Combine all date columns
    all_date_columns = date_columns + additional_date_columns
    
    # Convert date columns with UK format (day first)
    for col in all_date_columns:
        if col in data.columns:
            try:
                # First, replace common placeholder values with NaN
                data[col] = data[col].astype(str).replace(['1/1/1900', '01/01/1900', '1900-01-01',
                                                           '1/1/2020', '01/01/2020', '2020-01-01',
                                                           '1/1/2126', '01/01/2126', '2126-01-01',
                                                           'nan', 'NaT', 'NaN', '', 'None'], pd.NA)
                
                # Convert to datetime with day first format (UK date format)
                data[col] = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
                
                # Replace placeholder dates with NaT
                data.loc[data[col].isin([pd.Timestamp('1900-01-01'),
                                       pd.Timestamp('2020-01-01'),
                                       pd.Timestamp('2126-01-01')]), col] = pd.NaT
            except Exception as e:
                print(f"Error converting column {col}: {e}")
    
    return data

# Function to create milestone summary box
def create_milestone_box(title, forecast_col, actual_col, data):
    # Count non-blank entries in each column
    # For forecast column
    if forecast_col in data.columns:
        # Convert to string first to handle different formats
        forecast_data = data[forecast_col].astype(str)
        # Count entries that are not empty strings, NaN, 'nan', or '01/01/1900' (often used as null date)
        forecast_count = sum((forecast_data.notna()) & 
                             (forecast_data != '') & 
                             (forecast_data != 'nan') & 
                             (forecast_data != 'NaT') &
                             (forecast_data != '01/01/1900'))
    else:
        forecast_count = 0
    
    # For actual column
    if actual_col in data.columns:
        # Convert to string first to handle different formats
        actual_data = data[actual_col].astype(str)
        # Count entries that are not empty strings, NaN, 'nan', or '01/01/1900'
        actual_count = sum((actual_data.notna()) & 
                           (actual_data != '') & 
                           (actual_data != 'nan') & 
                           (actual_data != 'NaT') &
                           (actual_data != '01/01/1900'))
    else:
        actual_count = 0
    
    # Create HTML for the box
    html = f"""
    <div class="milestone-box">
        <div class="milestone-title">{title}</div>
        <div class="milestone-counts">
            <div class="forecast-box">
                <div class="count-number">{forecast_count}</div>
                <div class="count-label">Forecast</div>
            </div>
            <div class="actual-box">
                <div class="count-number">{actual_count}</div>
                <div class="count-label">Actual</div>
            </div>
        </div>
    </div>
    """
    return html

# Main function
def main():
    df, notifications = load_data()
    
    # Group notifications into an expandable section
    with st.sidebar.expander("🔔 Notifications", expanded=False):
        for note in notifications:
            if note.startswith("✅"):
                st.success(note[2:].strip())
            elif note.startswith("⚠️"):
                st.warning(note[2:].strip())
            elif note.startswith("❌"):
                st.error(note[2:].strip())
            else:
                st.info(note[2:].strip())
    
    if df is not None:
        data = prepare_data(df)
        
        # Show data loading status
        st.sidebar.success("Data loaded successfully!")
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Initialize session state for tracking filter reset
        if 'reset_filters' not in st.session_state:
            st.session_state.reset_filters = False
        
        # Filter by NS Status
        if 'NS\nStatus' in data.columns:
            status_options = ['All'] + sorted(data['NS\nStatus'].dropna().unique().tolist())
            selected_status = st.sidebar.selectbox(
                'NS Status', 
                status_options,
                index=0 if st.session_state.reset_filters else st.session_state.get("ns_status_index", 0),
                key="ns_status"
            )
            
            if selected_status != 'All':
                data = data[data['NS\nStatus'] == selected_status]
        
        # Filter by Site Type
        if 'Site Type' in data.columns:
            site_type_options = ['All'] + sorted(data['Site Type'].dropna().unique().tolist())
            selected_site_type = st.sidebar.selectbox(
                'Site Type', 
                site_type_options,
                index=0 if st.session_state.reset_filters else st.session_state.get("site_type_index", 0),
                key="site_type"
            )
            
            if selected_site_type != 'All':
                data = data[data['Site Type'] == selected_site_type]
        
        # Filter by Client Priority
        if 'Client Priority' in data.columns:
            priority_options = ['All'] + sorted(data['Client Priority'].dropna().unique().tolist())
            selected_priority = st.sidebar.selectbox(
                'Client Priority', 
                priority_options,
                index=0 if st.session_state.reset_filters else st.session_state.get("priority_index", 0),
                key="priority"
            )
            
            if selected_priority != 'All':
                data = data[data['Client Priority'] == selected_priority]
        
        # Filter by KTL Project Name
        if 'KTL Project Name' in data.columns:
            # Get unique project names and sort them
            project_names = data['KTL Project Name'].dropna().unique().tolist()
            project_names = sorted([str(name) for name in project_names if str(name).strip()])
            
            project_options = ['All'] + project_names
            selected_project = st.sidebar.selectbox(
                'KTL Project Name', 
                project_options,
                index=0 if st.session_state.reset_filters else st.session_state.get("project_index", 0),
                key="project"
            )
            
            if selected_project != 'All':
                data = data[data['KTL Project Name'] == selected_project]
        
        # Add Clear Filters button
        st.sidebar.markdown("---")
        
        def reset_filters():
            st.session_state.reset_filters = True
            # Reset all filter session states
            for key in ["ns_status", "site_type", "priority", "project"]:
                if key in st.session_state:
                    del st.session_state[key]
            return
        
        clear_filters = st.sidebar.button('🔄 Clear All Filters', on_click=reset_filters)
        
        # Reset the flag after all filters have been processed
        if st.session_state.reset_filters:
            st.session_state.reset_filters = False
            st.rerun()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Overview", "Project Status", "GA Overview", "DD Overview", 
            "KPI Metrics", "Resources", "Project Metrics"
        ])
        
        with tab1:
            # Create two columns for header and total count
            header_col, total_col = st.columns([3, 1])
            
            with header_col:
                st.header("Overview")
            
            # NS Status breakdown
            if 'NS\nStatus' in data.columns:
                # Calculate total count
                total_projects = len(data)
                
                # Display total count in a box
                with total_col:
                    st.markdown("""
                        <div style="border:1px solid rgba(128, 128, 128, 0.2); border-radius:5px; padding:10px; text-align:center; margin-top:10px; background-color: rgba(128, 128, 128, 0.1);">
                            <div style="font-size:0.8em; color: inherit;">Total Projects</div>
                            <div style="font-size:1.8em; font-weight:bold; color: inherit;">{}</div>
                        </div>
                    """.format(total_projects), unsafe_allow_html=True)
                
                # Create status counts dataframe
                status_counts = data['NS\nStatus'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                
                try:
                    # Create bar chart
                    fig = px.bar(
                        status_counts,
                        x='Status',
                        y='Count',  # Show count values on bars
                        text='Count',  # Show count values on bars
                        title='Project Count by Status',
                        color='Count',
                        color_continuous_scale='Plasma'  # Changed from Viridis for better contrast
                    )
                    
                    # Update layout for better readability
                    fig.update_traces(textposition='outside')  # Place count labels outside bars
                    fig.update_layout(
                        xaxis_title="Status",
                        yaxis_title="Number of Projects",
                        showlegend=False,
                        height=500,  # Increased height
                        margin=dict(t=100),  # Added top margin for labels
                        yaxis=dict(
                            range=[0, max(status_counts['Count']) * 1.15]  # Extend y-axis range by 15%
                        )
                    )
                    
                    # Display the chart with a unique key
                    st.plotly_chart(fig, use_container_width=True, key="status_count_chart")
                    
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
            else:
                st.error("NS Status column not found in the dataset.")

            # Add GA Status breakdown
            if 'GA Status' in data.columns:
                # Create status counts dataframe for GA
                ga_status_counts = data['GA Status'].value_counts().reset_index()
                ga_status_counts.columns = ['Status', 'Count']
                
                try:
                    # Create bar chart for GA Status
                    ga_fig = px.bar(
                        ga_status_counts,
                        x='Status',
                        y='Count',
                        text='Count',  # Show count values on bars
                        title='GA Status Distribution',
                        color='Count',
                        color_continuous_scale='Plasma'  # Changed from Viridis for better contrast
                    )
                    
                    # Update layout for better readability
                    ga_fig.update_traces(textposition='outside')  # Place count labels outside bars
                    ga_fig.update_layout(
                        xaxis_title="GA Status",
                        yaxis_title="Number of Projects",
                        showlegend=False,
                        height=500,  # Increased height
                        margin=dict(t=100),  # Added top margin for labels
                        yaxis=dict(
                            range=[0, max(ga_status_counts['Count']) * 1.15]  # Extend y-axis range by 15%
                        )
                    )
                    
                    # Display the GA Status chart with a unique key
                    st.plotly_chart(ga_fig, use_container_width=True, key="ga_status_chart")
                    
                except Exception as e:
                    st.error(f"Error creating GA Status chart: {str(e)}")
            else:
                st.info("GA Status column not found in the dataset.")

            # Add DD Status breakdown
            if 'DD Status' in data.columns:
                # Create status counts dataframe for DD
                dd_status_counts = data['DD Status'].value_counts().reset_index()
                dd_status_counts.columns = ['Status', 'Count']
                
                try:
                    # Create bar chart for DD Status
                    dd_fig = px.bar(
                        dd_status_counts,
                        x='Status',
                        y='Count',
                        text='Count',  # Show count values on bars
                        title='DD Status Distribution',
                        color='Count',
                        color_continuous_scale='Plasma'  # Changed from Viridis for better contrast
                    )
                    
                    # Update layout for better readability
                    dd_fig.update_traces(textposition='outside')  # Place count labels outside bars
                    dd_fig.update_layout(
                        xaxis_title="DD Status",
                        yaxis_title="Number of Projects",
                        showlegend=False,
                        height=500,  # Increased height
                        margin=dict(t=100),  # Added top margin for labels
                        yaxis=dict(
                            range=[0, max(dd_status_counts['Count']) * 1.15]  # Extend y-axis range by 15%
                        )
                    )
                    
                    # Display the DD Status chart with a unique key
                    st.plotly_chart(dd_fig, use_container_width=True, key="dd_status_chart")
                    
                except Exception as e:
                    st.error(f"Error creating DD Status chart: {str(e)}")
            else:
                st.info("DD Status column not found in the dataset.")
        
        with tab2:
            st.header("Project Status")
            
            # Blocker status analysis
            if all(col in data.columns for col in ['Blocker Status', 'Blocker Owner', 'Blocker Reason']):
                st.subheader("Blocker Analysis")
                
                # Filter only blocked projects
                blocked_data = data[data['Blocker Status'].notna() & (data['Blocker Status'] != 'Cleared')]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Blocker Owner Distribution
                    blocker_owner_counts = blocked_data['Blocker Owner'].value_counts().reset_index()
                    blocker_owner_counts.columns = ['Blocker Owner', 'Count']
                    
                    fig = px.bar(blocker_owner_counts, x='Blocker Owner', y='Count',
                                title='Blocker Owner Distribution',
                                color='Count')
                    st.plotly_chart(fig, use_container_width=True, key="blocker_owner_chart")
                
                with col2:
                    # Blocker Reason Distribution
                    blocker_reason_counts = blocked_data['Blocker Reason'].value_counts().reset_index()
                    blocker_reason_counts.columns = ['Blocker Reason', 'Count']
                    
                    fig = px.pie(blocker_reason_counts, values='Count', names='Blocker Reason',
                                title='Blocker Reason Distribution')
                    st.plotly_chart(fig, use_container_width=True, key="blocker_reason_pie")
            
            # Priority analysis
            if 'Client Priority' in data.columns:
                st.subheader("Priority Analysis")
                
                priority_counts = data['Client Priority'].value_counts().reset_index()
                priority_counts.columns = ['Priority', 'Count']
                
                fig = px.bar(priority_counts, x='Priority', y='Count',
                            title='Project Distribution by Priority',
                            color='Count',
                            color_continuous_scale='Plasma')
                st.plotly_chart(fig, use_container_width=True, key="priority_distribution_chart")
            
            # Status by Site Type (if both columns exist)
            if all(col in data.columns for col in ['NS\nStatus', 'Site Type']):
                st.subheader("Status by Site Type")
                
                # Create a cross-tabulation of Site Type and NS Status
                cross_tab = pd.crosstab(data['Site Type'], data['NS\nStatus'])
                
                # Convert to long format for plotting
                cross_tab_long = cross_tab.reset_index().melt(id_vars=['Site Type'], 
                                                            var_name='Status', 
                                                            value_name='Count')
                
                fig = px.bar(cross_tab_long, x='Site Type', y='Count', color='Status',
                            title='Project Status by Site Type',
                            barmode='group')
                st.plotly_chart(fig, use_container_width=True, key="status_by_site_type_chart")
        
        with tab3:
            st.header("GA Overview")
            
            # Define GA-related columns
            ga_columns = [
                "GA\n(F)",
                "GA Issued Client (A)",
                "GA Approved Client (A)",
                "GA Issued Operator (A)",
                "GA Approved Operator (A)"
            ]
            
            # Check if all required columns exist
            if all(col in data.columns for col in ga_columns):
                # Create a copy of the data with only GA columns
                ga_data = data[ga_columns].copy()
                
                # Convert all dates to datetime if they aren't already
                for col in ga_columns:
                    if not pd.api.types.is_datetime64_dtype(ga_data[col]):
                        ga_data[col] = pd.to_datetime(ga_data[col], errors='coerce')
                
                # Create a complete date range from start of 2024
                start_date = pd.Timestamp('2024-01-01')
                end_date = pd.Timestamp('2025-12-31')
                all_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
                
                # Function to get week-month-year format
                def get_week_month_year(date):
                    if pd.isna(date):
                        return None
                    # Get week number and year
                    week = date.isocalendar()[1]
                    year = date.year
                    month = date.strftime('%b')
                    # Create a sortable string format (YYYY-WW-MMM)
                    return f"{year}-W{week:02d}-{month}"
                
                # Create a base dataframe with all weeks
                base_weeks = pd.DataFrame({
                    'Date': all_dates,
                    'Week': [get_week_month_year(d) for d in all_dates]
                })
                
                # Convert dates to week-month-year format for actual data
                for col in ga_columns:
                    ga_data[f"{col}_week"] = ga_data[col].apply(get_week_month_year)
                
                # Create the counts table with all weeks
                counts_data = []
                for _, row in base_weeks.iterrows():
                    week = row['Week']
                    # Convert the sortable format back to display format
                    year, week_num, month = week.split('-')
                    display_week = f"{week_num}-{month}-{year}"
                    
                    row_data = {'Week': display_week}
                    for col in ga_columns:
                        # Count how many dates fall into this week
                        count = sum(ga_data[f"{col}_week"] == week)
                        row_data[col] = count
                    counts_data.append(row_data)
                
                # Convert to DataFrame
                counts_df = pd.DataFrame(counts_data)
                
                # Rename columns for better display
                column_rename = {
                    "GA\n(F)": "GA Forecast",
                    "GA Issued Client (A)": "GA Issued to Client",
                    "GA Approved Client (A)": "GA Approved by Client",
                    "GA Issued Operator (A)": "GA Issued to Operator",
                    "GA Approved Operator (A)": "GA Approved by Operator"
                }
                counts_df = counts_df.rename(columns=column_rename)
                
                # Display the pivoted table
                st.subheader("GA Tasks by Week")
                
                # Add date range filter
                st.write("Filter date range:")
                ga_col1, ga_col2 = st.columns(2)
                with ga_col1:
                    ga_start_date = st.date_input(
                        "Start Date",
                        value=pd.Timestamp('2024-01-01').date(),
                        key="ga_start_date"
                    )
                with ga_col2:
                    ga_end_date = st.date_input(
                        "End Date",
                        value=pd.Timestamp('2025-12-31').date(),
                        key="ga_end_date"
                    )
                
                # Create a copy of the DataFrame for display
                display_df = counts_df.copy()
                
                # Filter the weeks based on date range
                filtered_weeks = []
                for week in display_df['Week']:
                    # Parse the week string (format: "W[week_number]-[month]-[year]")
                    try:
                        week_parts = week.split('-')
                        if len(week_parts) == 3:
                            week_date = pd.to_datetime(f"{week_parts[2]}-{week_parts[1]}-01")
                            if ga_start_date <= week_date.date() <= ga_end_date:
                                filtered_weeks.append(week)
                    except:
                        continue
                
                # Filter the display DataFrame
                filtered_display_df = display_df[display_df['Week'].isin(filtered_weeks)]
                
                # Pivot the table to have weeks as columns and task types as rows
                pivoted_df = filtered_display_df.set_index('Week').transpose()
                
                # Style the DataFrame
                styled_df = (pivoted_df.style.format("{:.0f}")
                    .background_gradient(cmap='RdYlBu_r')  # Changed from YlOrRd for better contrast
                    .set_properties(**{'text-align': 'center'})
                    .set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('color', '#E0E0E0')]},
                        {'selector': 'td', 'props': [('text-align', 'center'), ('color', '#E0E0E0')]}
                    ]))
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Add download button for the table
                csv = pivoted_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download GA Overview as CSV",
                    data=csv,
                    file_name="ga_overview.csv",
                    mime="text/csv",
                )
                
                # Add visualization
                st.subheader("GA Tasks Timeline")
                
                # Convert weeks to month-year and calculate cumulative sums
                plot_df = counts_df.copy()
                
                # Extract month-year from the Week column
                plot_df['Month-Year'] = pd.to_datetime(plot_df['Week'].str.split('-').str[-2:].str.join('-'), format='%b-%Y').dt.strftime('%Y-%m')
                
                # Group by month-year and sum the counts
                plot_df = plot_df.groupby('Month-Year').agg({
                    'GA Forecast': 'sum',
                    'GA Issued to Client': 'sum',
                    'GA Approved by Client': 'sum',
                    'GA Issued to Operator': 'sum',
                    'GA Approved by Operator': 'sum'
                }).reset_index()
                
                # Sort by Month-Year to ensure correct cumulative calculation
                plot_df = plot_df.sort_values('Month-Year')
                
                # Calculate cumulative sums for each column
                value_columns = [
                    'GA Forecast',
                    'GA Issued to Client',
                    'GA Approved by Client',
                    'GA Issued to Operator',
                    'GA Approved by Operator'
                ]
                for col in value_columns:
                    plot_df[col] = plot_df[col].cumsum()
                
                # Melt the dataframe for plotting
                plot_df = plot_df.melt(
                    id_vars=['Month-Year'],
                    value_vars=value_columns,
                    var_name='Task Type',
                    value_name='Cumulative Count'
                )
                
                # Create line plot with a unique key
                fig = px.line(
                    plot_df,
                    x='Month-Year',
                    y='Cumulative Count',
                    color='Task Type',
                    title='Cumulative GA Tasks Over Time',
                    markers=True
                )
                
                # Update layout for better readability
                fig.update_layout(
                    xaxis_title="Month-Year",
                    yaxis_title="Cumulative Number of Tasks",
                    xaxis_tickangle=-45,
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, key="ga_cumulative_chart")
                
            else:
                missing_cols = [col for col in ga_columns if col not in data.columns]
                st.warning(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure all GA-related columns are present in the dataset to view this analysis.")

            # Add GA Projects Snapshot section at the bottom
            st.subheader("GA Projects Snapshot")
            
            # Define required columns for the snapshot
            snapshot_columns = [
                'Internal ID',
                'Site ID',
                'Site Type',
                'GA Supplier',
                'GA\n(F)',
                'GA Design Notes'
            ]
            
            # Check if all required columns exist
            if all(col in data.columns for col in snapshot_columns + ['RMSV(A)', 'GA Issued Client (A)', 'NS\nStatus']):
                # Create a copy of the data with required columns
                snapshot_df = data[snapshot_columns].copy()
                
                # Apply filters:
                # 1. Exclude sites with null RMSV(A)
                # 2. Exclude sites with non-null GA Issued Client (A)
                # 3. Exclude sites with NS Status = In Progress
                filtered_indices = (
                    (data['RMSV(A)'].notna()) &  # Changed from isna() to notna()
                    (data['GA Issued Client (A)'].isna()) &
                    (data['NS\nStatus'] == 'In Progress')
                )
                
                snapshot_df = snapshot_df[filtered_indices]
                
                # Convert GA\n(F) to datetime if it isn't already
                if not pd.api.types.is_datetime64_dtype(snapshot_df['GA\n(F)']):
                    snapshot_df['GA\n(F)'] = pd.to_datetime(snapshot_df['GA\n(F)'], errors='coerce')
                
                # Add date range filter for GA\n(F)
                st.write("Filter by GA Forecast Date Range:")
                ga_snapshot_col1, ga_snapshot_col2 = st.columns(2)
                
                min_date = snapshot_df['GA\n(F)'].min()
                max_date = snapshot_df['GA\n(F)'].max()
                
                with ga_snapshot_col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date.date() if pd.notnull(min_date) else pd.Timestamp('2024-01-01').date(),
                        key="ga_snapshot_start_date"
                    )
                with ga_snapshot_col2:
                    end_date = st.date_input(
                        "End Date",
                        value=max_date.date() if pd.notnull(max_date) else pd.Timestamp('2025-12-31').date(),
                        key="ga_snapshot_end_date"
                    )
                
                # Apply date range filter if dates are selected
                if start_date and end_date:
                    # Convert the GA\n(F) column to datetime64[ns] if it isn't already
                    if not pd.api.types.is_datetime64_dtype(snapshot_df['GA\n(F)']):
                        snapshot_df['GA\n(F)'] = pd.to_datetime(snapshot_df['GA\n(F)'])
                    
                    # Convert start_date and end_date to pandas datetime
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    # Apply the filter using datetime64[ns] comparison
                    snapshot_df = snapshot_df[
                        (snapshot_df['GA\n(F)'] >= start_dt) &
                        (snapshot_df['GA\n(F)'] <= end_dt)
                    ]
                
                # Sort by GA\n(F) date
                date_filtered_df = snapshot_df.sort_values('GA\n(F)', ascending=True)
                
                # Display the dataframe with styling
                st.dataframe(
                    date_filtered_df.style.format({
                        'GA\n(F)': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                    }),
                    use_container_width=True
                )
                
                # Add download button for the snapshot
                csv = date_filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download GA Snapshot as CSV",
                    data=csv,
                    file_name="ga_snapshot.csv",
                    mime="text/csv",
                )
            else:
                missing_cols = [col for col in snapshot_columns + ['RMSV(A)', 'GA Issued Client (A)', 'NS\nStatus'] 
                              if col not in data.columns]
                st.warning(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure all required columns are present in the dataset to view the GA snapshot.")

        with tab4:
            st.header("DD Overview")
            
            # Define DD-related columns
            dd_columns = [
                "DD\n(F)",
                "DD Issued Client (A)",
                "DD Approved Client (A)",
                "DD Issued to Operator (A)",
                "DD Approved to Operator (A)"
            ]
            
            # Check if all required columns exist
            if all(col in data.columns for col in dd_columns):
                # Create a copy of the data with only DD columns
                dd_data = data[dd_columns].copy()
                
                # Convert all dates to datetime if they aren't already
                for col in dd_columns:
                    if not pd.api.types.is_datetime64_dtype(dd_data[col]):
                        dd_data[col] = pd.to_datetime(dd_data[col], errors='coerce')
                
                # Create a complete date range from start of 2024
                start_date = pd.Timestamp('2024-01-01')
                end_date = pd.Timestamp('2025-12-31')
                all_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
                
                # Function to get week-month-year format
                def get_week_month_year(date):
                    if pd.isna(date):
                        return None
                    # Get week number and year
                    week = date.isocalendar()[1]
                    year = date.year
                    month = date.strftime('%b')
                    # Create a sortable string format (YYYY-WW-MMM)
                    return f"{year}-W{week:02d}-{month}"
                
                # Create a base dataframe with all weeks
                base_weeks = pd.DataFrame({
                    'Date': all_dates,
                    'Week': [get_week_month_year(d) for d in all_dates]
                })
                
                # Convert dates to week-month-year format for actual data
                for col in dd_columns:
                    dd_data[f"{col}_week"] = dd_data[col].apply(get_week_month_year)
                
                # Create the counts table with all weeks
                counts_data = []
                for _, row in base_weeks.iterrows():
                    week = row['Week']
                    # Convert the sortable format back to display format
                    year, week_num, month = week.split('-')
                    display_week = f"{week_num}-{month}-{year}"
                    
                    row_data = {'Week': display_week}
                    for col in dd_columns:
                        # Count how many dates fall into this week
                        count = sum(dd_data[f"{col}_week"] == week)
                        row_data[col] = count
                    counts_data.append(row_data)
                
                # Convert to DataFrame
                counts_df = pd.DataFrame(counts_data)
                
                # Rename columns for better display
                column_rename = {
                    "DD\n(F)": "DD Forecast",
                    "DD Issued Client (A)": "DD Issued to Client",
                    "DD Approved Client (A)": "DD Approved by Client",
                    "DD Issued to Operator (A)": "DD Issued to Operator",
                    "DD Approved to Operator (A)": "DD Approved by Operator"
                }
                counts_df = counts_df.rename(columns=column_rename)
                
                # Display the pivoted table
                st.subheader("DD Tasks by Week")
                
                # Add date range filter
                st.write("Filter date range:")
                dd_col1, dd_col2 = st.columns(2)
                with dd_col1:
                    dd_start_date = st.date_input(
                        "Start Date",
                        value=pd.Timestamp('2024-01-01').date(),
                        key="dd_start_date"
                    )
                with dd_col2:
                    dd_end_date = st.date_input(
                        "End Date",
                        value=pd.Timestamp('2025-12-31').date(),
                        key="dd_end_date"
                    )
                
                # Create a copy of the DataFrame for display
                display_df = counts_df.copy()
                
                # Filter the weeks based on date range
                filtered_weeks = []
                for week in display_df['Week']:
                    # Parse the week string (format: "W[week_number]-[month]-[year]")
                    try:
                        week_parts = week.split('-')
                        if len(week_parts) == 3:
                            week_date = pd.to_datetime(f"{week_parts[2]}-{week_parts[1]}-01")
                            if dd_start_date <= week_date.date() <= dd_end_date:
                                filtered_weeks.append(week)
                    except:
                        continue
                
                # Filter the display DataFrame
                filtered_display_df = display_df[display_df['Week'].isin(filtered_weeks)]
                
                # Pivot the table to have weeks as columns and task types as rows
                pivoted_df = filtered_display_df.set_index('Week').transpose()
                
                # Style the DataFrame for DD section
                styled_df = (pivoted_df.style.format("{:.0f}")
                    .background_gradient(cmap='RdYlBu_r')  # Changed from YlOrRd for better contrast
                    .set_properties(**{'text-align': 'center'})
                    .set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('color', '#E0E0E0')]},
                        {'selector': 'td', 'props': [('text-align', 'center'), ('color', '#E0E0E0')]}
                    ]))
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Add download button for the table
                csv = pivoted_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download DD Overview as CSV",
                    data=csv,
                    file_name="dd_overview.csv",
                    mime="text/csv",
                )
                
                # Add visualization
                st.subheader("DD Tasks Timeline")
                
                # Convert weeks to month-year and calculate cumulative sums
                plot_df = counts_df.copy()
                
                # Extract month-year from the Week column
                plot_df['Month-Year'] = pd.to_datetime(plot_df['Week'].str.split('-').str[-2:].str.join('-'), format='%b-%Y').dt.strftime('%Y-%m')
                
                # Group by month-year and sum the counts
                plot_df = plot_df.groupby('Month-Year').agg({
                    'DD Forecast': 'sum',
                    'DD Issued to Client': 'sum',
                    'DD Approved by Client': 'sum',
                    'DD Issued to Operator': 'sum',
                    'DD Approved by Operator': 'sum'
                }).reset_index()
                
                # Sort by Month-Year to ensure correct cumulative calculation
                plot_df = plot_df.sort_values('Month-Year')
                
                # Calculate cumulative sums for each column
                value_columns = [
                    'DD Forecast',
                    'DD Issued to Client',
                    'DD Approved by Client',
                    'DD Issued to Operator',
                    'DD Approved by Operator'
                ]
                for col in value_columns:
                    plot_df[col] = plot_df[col].cumsum()
                
                # Melt the dataframe for plotting
                plot_df = plot_df.melt(
                    id_vars=['Month-Year'],
                    value_vars=value_columns,
                    var_name='Task Type',
                    value_name='Cumulative Count'
                )
                
                # Create line plot with a unique key
                fig = px.line(
                    plot_df,
                    x='Month-Year',
                    y='Cumulative Count',
                    color='Task Type',
                    title='Cumulative DD Tasks Over Time',
                    markers=True
                )
                
                # Update layout for better readability
                fig.update_layout(
                    xaxis_title="Month-Year",
                    yaxis_title="Cumulative Number of Tasks",
                    xaxis_tickangle=-45,
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, key="dd_cumulative_chart")
                
            else:
                missing_cols = [col for col in dd_columns if col not in data.columns]
                st.warning(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure all DD-related columns are present in the dataset to view this analysis.")
            
            # Add DD process milestone counts at the top of DD Overview tab
            st.subheader("DD Process Counts")
            
            # Define the DD process columns to count
            dd_process_columns = [
                "DD Issued Client (A)",
                "DD Approved Client (A)",
                "DD Issued to Operator (A)"
            ]
            
            # Create a 3-column layout for the DD process boxes
            dd_cols = st.columns(3)
            
            # Add DD process count boxes
            for i, column in enumerate(dd_process_columns):
                with dd_cols[i]:
                    if column in data.columns:
                        # Convert to string first to handle different formats
                        col_data = data[column].astype(str)
                        # Count entries that are not empty strings, NaN, 'nan', or '01/01/1900'
                        count = sum((col_data.notna()) & 
                                 (col_data != '') & 
                                 (col_data != 'nan') & 
                                 (col_data != 'NaT') &
                                 (col_data != '01/01/1900'))
                        
                        # Create an HTML box with just the count (no forecast column needed)
                        box_title = column.replace(" (A)", "").replace("\n", " ")
                        html = f"""
                        <div class="milestone-box">
                            <div class="milestone-title">{box_title}</div>
                            <div class="count-number" style="font-size: 2em; font-weight: bold; color: #142656; padding: 10px 0;">{count}</div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.warning(f"Column '{column}' not found in the dataset")
            
            # Add DD Forecast vs Actual line graph
            st.subheader("DD Forecast vs Actual Dates")
            
            # Check if both DD (F) and DD Issued Client (A) columns exist and are datetime
            if all(col in data.columns for col in ["DD\n(F)", "DD Issued Client (A)"]):
                # Filter out rows where either date is missing
                date_data = data[(data["DD\n(F)"].notna()) & (data["DD Issued Client (A)"].notna())].copy()
                
                if not date_data.empty:
                    # Convert data for visualization
                    # Create a new dataframe with the dates and their frequencies by month
                    dd_f_counts = date_data.groupby(date_data["DD\n(F)"].dt.strftime('%Y-%m')).size().reset_index()
                    dd_f_counts.columns = ['Month', 'Count']
                    # Sort by month to ensure correct cumulative calculation
                    dd_f_counts = dd_f_counts.sort_values('Month')
                    # Calculate cumulative sum
                    dd_f_counts['Cumulative'] = dd_f_counts['Count'].cumsum()
                    dd_f_counts['Type'] = 'Forecast'
                    
                    dd_actual_counts = date_data.groupby(date_data["DD Issued Client (A)"].dt.strftime('%Y-%m')).size().reset_index()
                    dd_actual_counts.columns = ['Month', 'Count']
                    # Sort by month to ensure correct cumulative calculation
                    dd_actual_counts = dd_actual_counts.sort_values('Month')
                    # Calculate cumulative sum
                    dd_actual_counts['Cumulative'] = dd_actual_counts['Count'].cumsum()
                    dd_actual_counts['Type'] = 'Actual'
                    
                    # Combine the dataframes
                    combined_df = pd.concat([dd_f_counts, dd_actual_counts])
                    
                    # Create the line chart with cumulative values
                    fig = px.line(
                        combined_df, 
                        x='Month', 
                        y='Cumulative', 
                        color='Type',
                        title='Cumulative DD Forecast vs Actual Dates by Month',
                        markers=True,
                        labels={'Month': 'Month-Year', 'Cumulative': 'Cumulative Number of Projects'},
                        color_discrete_map={'Forecast': '#3366CC', 'Actual': '#FF9900'}
                    )
                    
                    # Customize the layout
                    fig.update_layout(
                        xaxis_title="Month-Year",
                        yaxis_title="Cumulative Number of Projects",
                        legend_title="Date Type",
                        hovermode="x unified"
                    )
                    
                    # Show the chart with a unique key
                    st.plotly_chart(fig, use_container_width=True, key="dd_forecast_actual_cumulative")
                    
                    # Add monthly comparison graph as well
                    st.subheader("Monthly DD Forecast vs Actual")
                    monthly_fig = px.bar(
                        combined_df,
                        x='Month',
                        y='Count',
                        color='Type',
                        barmode='group',
                        title='Monthly DD Forecast vs Actual Counts',
                        labels={'Month': 'Month-Year', 'Count': 'Number of Projects'},
                        color_discrete_map={'Forecast': '#3366CC', 'Actual': '#FF9900'}
                    )
                    
                    # Customize the monthly layout
                    monthly_fig.update_layout(
                        xaxis_title="Month-Year",
                        yaxis_title="Number of Projects",
                        legend_title="Date Type"
                    )
                    
                    st.plotly_chart(monthly_fig, use_container_width=True, key="dd_forecast_actual_monthly")
                    
                    # Add analysis of on-time delivery
                    on_time_count = sum(date_data["DD Issued Client (A)"] <= date_data["DD\n(F)"])
                    delayed_count = sum(date_data["DD Issued Client (A)"] > date_data["DD\n(F)"])
                    total_count = len(date_data)
                    
                    # Calculate percentages
                    on_time_pct = (on_time_count / total_count) * 100 if total_count > 0 else 0
                    delayed_pct = (delayed_count / total_count) * 100 if total_count > 0 else 0
                    
                    # Display metrics
                    st.subheader("DD Delivery Performance")
                    metrics_cols = st.columns(3)
                    
                    with metrics_cols[0]:
                        st.metric("Total Projects With Both Dates", total_count)
                    
                    with metrics_cols[1]:
                        st.metric("On Time or Early Delivery", f"{on_time_count} ({on_time_pct:.1f}%)")
                    
                    with metrics_cols[2]:
                        st.metric("Delayed Delivery", f"{delayed_count} ({delayed_pct:.1f}%)")
                    
                    # Calculate average delay for delayed projects
                    if delayed_count > 0:
                        delayed_projects = date_data[date_data["DD Issued Client (A)"] > date_data["DD\n(F)"]]
                        avg_delay = (delayed_projects["DD Issued Client (A)"] - delayed_projects["DD\n(F)"]).dt.days.mean()
                        
                        st.info(f"For delayed projects, the average delay is {avg_delay:.1f} days.")
                else:
                    st.info("No projects found with both forecast and actual DD dates.")
            else:
                missing_cols = []
                if "DD\n(F)" not in data.columns:
                    missing_cols.append("DD (F)")
                if "DD Issued Client (A)" not in data.columns:
                    missing_cols.append("DD Issued Client (A)")
                
                st.warning(f"Cannot create comparison chart. Missing columns: {', '.join(missing_cols)}")
            
            # Check if DD Status column exists
            if 'DD Status' in data.columns:
                # Count by DD Status
                dd_status_counts = data['DD Status'].value_counts().reset_index()
                dd_status_counts.columns = ['Status', 'Count']
                
                # Create visualization section
                st.subheader("DD Status Distribution")
                
                # Create two columns for different chart types
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        dd_status_counts,
                        x='Status',
                        y='Count',
                        title='DD Status Distribution',
                        color='Count',
                        color_continuous_scale='Plasma'  # Changed from Viridis for better contrast
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key="dd_status_bar_chart")
                
                with col2:
                    # Pie chart
                    fig_pie = px.pie(
                        dd_status_counts,
                        values='Count',
                        names='Status',
                        title='DD Status Distribution (%)'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, key="dd_status_pie_chart")
                
                # Table of status counts
                st.subheader("DD Status Counts")
                st.dataframe(dd_status_counts.sort_values('Count', ascending=False), use_container_width=True)
                
                # Show additional metrics if possible
                if 'DD Issued Client (A)' in data.columns:
                    # Calculate stats about DD issuance
                    dd_issued_count = len(data[data['DD Issued Client (A)'].notna()])
                    total_count = len(data)
                    
                    # Create metrics row
                    st.subheader("DD Key Metrics")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.metric("Total Projects", total_count)
                    
                    with metric_cols[1]:
                        st.metric("DD Issued Count", dd_issued_count)
                    
                    with metric_cols[2]:
                        dd_issued_pct = (dd_issued_count / total_count) * 100 if total_count > 0 else 0
                        st.metric("DD Issued Percentage", f"{dd_issued_pct:.1f}%")
            else:
                st.warning("The 'DD Status' column was not found in your dataset. Please ensure this column exists to view DD status distributions.")
                
                # Show available columns as reference
                with st.expander("Available columns in dataset"):
                    st.write(data.columns.tolist())
                
                # Look for similar columns that might contain DD status info
                dd_related_cols = [col for col in data.columns if 'dd' in col.lower() and 'status' in col.lower()]
                if dd_related_cols:
                    st.info("The following columns might contain DD status information:")
                    st.write(dd_related_cols)
                    
                    # Allow selecting an alternative column
                    alt_status_col = st.selectbox(
                        "Select an alternative column for DD Status:",
                        options=['None'] + dd_related_cols,
                        index=0
                    )
                    
                    if alt_status_col != 'None':
                        # Use the alternative column for visualization
                        alt_status_counts = data[alt_status_col].value_counts().reset_index()
                        alt_status_counts.columns = ['Status', 'Count']
                        
                        st.subheader(f"Status Distribution using '{alt_status_col}'")
                        fig = px.pie(
                            alt_status_counts,
                            values='Count',
                            names='Status',
                            title=f"{alt_status_col} Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="alt_status_pie_chart")

        with tab5:
            st.header("KPI Metrics")
            
            # Check for NS Status column and provide column selection if not found
            status_column = None
            
            if 'NS\nStatus' in data.columns:
                status_column = 'NS\nStatus'
                st.success("Found 'NS Status' column in the dataset.")
            else:
                st.warning("The column named exactly 'NS Status' was not found in your dataset.")
                # Show all column names to help identify the actual status column
                st.subheader("Available Columns")
                st.write("Here are the first 15 columns in your dataset:")
                st.write(data.columns[:15].tolist())
                
                # Try to find columns that might be the status column
                possible_status_cols = [col for col in data.columns if 'status' in col.lower() or 'state' in col.lower()]
                
                if possible_status_cols:
                    st.write("These columns might contain status information:")
                    
                    # Let user select which column to use as status
                    status_column = st.selectbox(
                        "Select a column to use as status:",
                        options=['None'] + possible_status_cols,
                        index=0
                    )
                    
                    if status_column == 'None':
                        status_column = None
                    else:
                        st.success(f"Using '{status_column}' as the status column")
                else:
                    st.error("No columns with 'status' or 'state' in their names were found.")
            
            # Calculate key metrics
            col1, col2, col3 = st.columns(3)
            
            # 1. Instructed Pot (count of NS Status = In Progress)
            instructed_pot = 0
            if status_column in data.columns:
                instructed_pot = len(data[data[status_column] == 'In Progress'])
            
            # 2. DD Issued (count of DD Issued Client (A) with values)
            dd_issued = 0
            if 'DD Issued Client (A)' in data.columns:
                dd_issued = len(data[data['DD Issued Client (A)'].notna()])
            
            # 3. DD Pending Issue (Instructed Pot - DD Issued)
            dd_pending = instructed_pot - dd_issued
            
            # Display metrics
            with col1:
                st.metric("Instructed Pot", instructed_pot)
                st.markdown("*Projects with 'In Progress' status*")
            
            with col2:
                st.metric("DD Issued", dd_issued)
                st.markdown("*Projects with DD already issued to client*")
            
            with col3:
                st.metric("DD Pending Issue", dd_pending)
                st.markdown("*Projects waiting for DD to be issued*")
            
            # If status column is available, show the rest of the content
            if status_column:
                # Show progress visualization
                st.subheader("DD Issuance Progress")
                
                # Calculate percentage completion
                if instructed_pot > 0:
                    completion_pct = (dd_issued / instructed_pot) * 100
                else:
                    completion_pct = 0
                
                # Create progress bar
                st.progress(completion_pct / 100)
                st.markdown(f"**{completion_pct:.1f}%** of in-progress projects have had DD issued")
                
                # Create columns for pie chart and bar chart
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart showing DD issued vs pending
                    pie_data = pd.DataFrame({
                        'Status': ['DD Issued', 'DD Pending'],
                        'Count': [dd_issued, dd_pending]
                    })
                    
                    fig = px.pie(
                        pie_data,
                        values='Count',
                        names='Status',
                        title='DD Issuance Status',
                        color='Status',
                        color_discrete_map={
                            'DD Issued': '#00BFA5',  # Bright teal
                            'DD Pending': '#FF6E40'  # Bright orange
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True, key="dd_issuance_status_pie_kpi")
                
                with col2:
                    # Bar chart comparing instructed pot vs DD issued
                    bar_data = pd.DataFrame({
                        'Category': ['Instructed Pot', 'DD Issued', 'DD Pending'],
                        'Count': [instructed_pot, dd_issued, dd_pending]
                    })
                    
                    fig = px.bar(
                        bar_data,
                        x='Category',
                        y='Count',
                        title='Project Metrics Comparison',
                        color='Category',
                        color_discrete_map={
                            'Instructed Pot': '#64B5F6',  # Bright blue
                            'DD Issued': '#00BFA5',      # Bright teal
                            'DD Pending': '#FF6E40'      # Bright orange
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True, key="project_metrics_comparison_kpi")
                
                # Site Type breakdown for DD Pending projects
                if all(col in data.columns for col in ['Site Type', 'DD Issued Client (A)']):
                    st.subheader("DD Pending by Site Type")
                    
                    # Filter for in progress projects that don't have DD issued
                    pending_projects = data[
                        (data[status_column] == 'In Progress') & 
                        (data['DD Issued Client (A)'].isna())
                    ]
                    
                    site_type_counts = pending_projects['Site Type'].value_counts().reset_index()
                    site_type_counts.columns = ['Site Type', 'Pending Count']
                    
                    # Create horizontal bar chart sorted by count
                    site_type_counts = site_type_counts.sort_values('Pending Count', ascending=True)
                    
                    fig = px.bar(
                        site_type_counts,
                        y='Site Type',
                        x='Pending Count',
                        title='DD Pending by Site Type',
                        orientation='h',
                        color='Pending Count',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="dd_pending_by_site_type_kpi")
                
                # Timeline projection
                st.subheader("DD Issuance Timeline Projection")
                
                # Calculate average time from in progress to DD issuance (for completed projects)
                if all(col in data.columns for col in ['DD Issued Client (A)', 'Instruction Date']):
                    # Filter for projects that have both instruction date and DD issued date
                    completed_projects = data[
                        (data['DD Issued Client (A)'].notna()) & 
                        (data['Instruction Date'].notna())
                    ].copy()
                    
                    if not completed_projects.empty:
                        # Calculate duration in days
                        completed_projects['Duration_Days'] = (
                            completed_projects['DD Issued Client (A)'] - 
                            completed_projects['Instruction Date']
                        ).dt.days
                        
                        # Filter out negative durations (likely data errors)
                        completed_projects = completed_projects[completed_projects['Duration_Days'] >= 0]
                        
                        if not completed_projects.empty:
                            # Calculate average duration
                            avg_duration = completed_projects['Duration_Days'].median()  # Using median to avoid outlier effects
                            
                            # Create projection chart
                            st.markdown(f"Based on historical data, projects take an average of **{avg_duration:.0f} days** from instruction to DD issuance.")
                            
                            # Make sure we have the required columns for pending projects
                            if all(col in data.columns for col in ['DD Issued Client (A)', 'Instruction Date']):
                                # Filter for pending projects with instruction dates
                                pending_projects = data[
                                    (data[status_column] == 'In Progress') & 
                                    (data['DD Issued Client (A)'].isna())
                                ]
                                
                                # Show projects that are approaching or have exceeded the average duration
                                pending_with_dates = pending_projects[pending_projects['Instruction Date'].notna()].copy()
                                
                                if not pending_with_dates.empty:
                                    # Calculate days since instruction
                                    pending_with_dates['Days_Since_Instruction'] = (
                                        pd.Timestamp.now() - 
                                        pending_with_dates['Instruction Date']
                                    ).dt.days
                                    
                                    # Calculate percentage of average duration
                                    pending_with_dates['Percent_of_Avg'] = (
                                        pending_with_dates['Days_Since_Instruction'] / avg_duration
                                    ) * 100
                                    
                                    # Flag projects as: On Track, At Risk, or Overdue
                                    pending_with_dates['Status'] = pending_with_dates['Percent_of_Avg'].apply(
                                        lambda x: 'On Track' if x < 80 else ('At Risk' if x < 100 else 'Overdue')
                                    )
                                    
                                    # Count by status
                                    status_counts = pending_with_dates['Status'].value_counts().reset_index()
                                    status_counts.columns = ['Status', 'Count']
                                    
                                    # Create horizontal bar chart
                                    fig = px.bar(
                                        status_counts,
                                        y='Status',
                                        x='Count',
                                        title='DD Pending Projects Timeline Status',
                                        orientation='h',
                                        color='Status',
                                        color_discrete_map={
                                            'On Track': '#00BFA5',    # Bright teal
                                            'At Risk': '#FFB74D',     # Bright amber
                                            'Overdue': '#FF5252'      # Bright red
                                        },
                                        category_orders={"Status": ["On Track", "At Risk", "Overdue"]}
                                    )
                                    st.plotly_chart(fig, use_container_width=True, key="dd_pending_timeline_status_kpi")
                                    
                                    # Show table of overdue projects
                                    overdue_projects = pending_with_dates[pending_with_dates['Status'] == 'Overdue']
                                    if not overdue_projects.empty:
                                        st.subheader("Overdue Projects")
                                        
                                        # Format the display dataframe
                                        display_columns = ['Site ID', 'Site Type', 'Instruction Date', 'Days_Since_Instruction']
                                        display_df = overdue_projects[
                                            [col for col in display_columns if col in overdue_projects.columns]
                                        ].copy()
                                        
                                        # Rename columns for better display
                                        col_rename = {
                                            'Days_Since_Instruction': 'Days Since Instruction'
                                        }
                                        display_df.rename(columns=col_rename, inplace=True)
                                        
                                        # Sort by days since instruction (descending)
                                        display_df = display_df.sort_values('Days Since Instruction', ascending=False)
                                        
                                        # Show the table
                                        st.dataframe(display_df, use_container_width=True)
                                else:
                                    st.info("No pending projects with instruction dates found.")
                            else:
                                st.info("Missing required columns to calculate pending project timelines.")
                        else:
                            st.info("No valid duration data available after filtering.")
                    else:
                        st.info("No projects with both instruction and DD issued dates found.")
                else:
                    st.info("Missing required columns for timeline projection analysis.")
            else:
                # If no status column is available/selected, provide helpful information
                st.warning("""
                To use this tab properly, you need to select a column containing status information. 
                Without this, we can't calculate the number of in-progress projects or pending DDs.
                
                Look for a column in your dataset that contains values like 'In Progress', 'Closed', etc.
                """)
                
                # Display info about all columns
                with st.expander("Show all columns in your dataset"):
                    st.write(data.columns.tolist())
            
            # Always show information about DD Issued column
            if 'DD Issued Client (A)' not in data.columns:
                st.warning("Note: The 'DD Issued Client (A)' column is not available in your dataset. This tab requires this column to calculate metrics properly.")

        with tab6:
            st.header("Resources Analysis")
            
            # Define the resource columns to analyze
            resource_columns = [
                "GA Supplier",
                "GA Designer",
                "Dependency Owner",
                "DD Designer"
            ]
            
            # Filter columns that exist in the dataset
            existing_resource_columns = [col for col in resource_columns if col in data.columns]
            
            if existing_resource_columns:
                # Create tabs for different resource views
                resource_tabs = st.tabs(["Individual Workload", "Company Workload", "Project Allocation"])
                
                # Individual Workload Analysis Tab
                with resource_tabs[0]:
                    st.subheader("Individual Resource Workload")
                    
                    # Select which resource column to analyze
                    selected_resource = st.selectbox(
                        "Select Resource Type:",
                        existing_resource_columns
                    )
                    
                    # Get data for the selected resource
                    resource_data = data[data[selected_resource].notna()].copy()
                    
                    if not resource_data.empty:
                        # Count projects per resource person
                        resource_counts = resource_data[selected_resource].value_counts().reset_index()
                        resource_counts.columns = ['Resource', 'Project Count']
                        
                        # Sort by count descending
                        resource_counts = resource_counts.sort_values('Project Count', ascending=False)
                        
                        # Limit to top 20 resources for better visualization
                        if len(resource_counts) > 20:
                            top_resources = resource_counts.head(20)['Resource'].tolist()
                            st.info(f"Showing top 20 resources out of {len(resource_counts)} total.")
                            # Filter data to only include top resources
                            filtered_resource_data = resource_data[resource_data[selected_resource].isin(top_resources)]
                        else:
                            top_resources = resource_counts['Resource'].tolist()
                            filtered_resource_data = resource_data
                        
                        # Check if status column exists for stacked visualization
                        if 'NS\nStatus' in resource_data.columns:
                            # Create stacked bar chart by status
                            st.subheader(f"Project Count by {selected_resource} with Status Breakdown")
                            
                            # First, ensure there's a status for all rows (replace NaN with "Unknown")
                            filtered_resource_data['NS\nStatus'] = filtered_resource_data['NS\nStatus'].fillna("Unknown")
                            
                            # Create a cross-tab of resource and status
                            status_crosstab = pd.crosstab(
                                filtered_resource_data[selected_resource], 
                                filtered_resource_data['NS\nStatus']
                            ).reset_index()
                            
                            # Melt the dataframe for plotting
                            status_data = status_crosstab.melt(
                                id_vars=[selected_resource],
                                var_name='Status',
                                value_name='Count'
                            )
                            
                            # Sort by total count
                            resource_order = resource_counts[resource_counts['Resource'].isin(top_resources)].sort_values('Project Count', ascending=False)['Resource'].tolist()
                            
                            # Create stacked bar chart with a unique key
                            fig = px.bar(
                                status_data,
                                x=selected_resource,
                                y='Count',
                                color='Status',
                                title=f"Project Count by {selected_resource} with Status Breakdown",
                                category_orders={selected_resource: resource_order}
                            )
                            
                            # Rotate x-axis labels for better readability
                            fig.update_layout(xaxis_tickangle=-45)
                            
                            st.plotly_chart(fig, use_container_width=True, key="resource_status_chart")
                        else:
                            # Create standard bar chart if no status column
                            displayed_counts = resource_counts.head(20) if len(resource_counts) > 20 else resource_counts
                            
                            # Create bar chart
                            fig = px.bar(
                                displayed_counts,
                                x='Resource',
                                y='Project Count',
                                title=f"Project Count by {selected_resource}",
                                color='Project Count',
                                color_continuous_scale='Viridis'
                            )
                            
                            # Rotate x-axis labels for better readability
                            fig.update_layout(xaxis_tickangle=-45)
                            
                            st.plotly_chart(fig, use_container_width=True, key="resource_count_chart")
                        
                        # Project Status Distribution for selected resource
                        if 'NS\nStatus' in resource_data.columns:
                            st.subheader(f"Project Status Distribution by {selected_resource}")
                            
                            # Select a specific resource to analyze
                            top_resources = resource_counts['Resource'].head(10).tolist()
                            selected_person = st.selectbox(
                                f"Select {selected_resource} to analyze:",
                                top_resources
                            )
                            
                            # Filter data for the selected person
                            person_data = resource_data[resource_data[selected_resource] == selected_person]
                            
                            # Create status distribution chart
                            status_counts = person_data['NS\nStatus'].value_counts().reset_index()
                            status_counts.columns = ['Status', 'Count']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(
                                    status_counts,
                                    values='Count',
                                    names='Status',
                                    title=f"Status Distribution for {selected_person}"
                                )
                                st.plotly_chart(fig, use_container_width=True, key="person_status_pie")
                            
                            with col2:
                                # Calculate metrics
                                total_projects = len(person_data)
                                in_progress = len(person_data[person_data['NS\nStatus'] == 'In Progress'])
                                completed = len(person_data[person_data['NS\nStatus'] == 'Closed'])
                                
                                # Show key metrics
                                st.metric("Total Projects", total_projects)
                                st.metric("In Progress", in_progress, f"{(in_progress/total_projects)*100:.1f}%")
                                st.metric("Completed", completed, f"{(completed/total_projects)*100:.1f}%")
                                
                                # Check if there's Blocker Status information
                                if 'Blocker Status' in person_data.columns:
                                    blocked = len(person_data[person_data['Blocker Status'].notna() & 
                                                    (person_data['Blocker Status'] != 'Cleared')])
                                    st.metric("Blocked Projects", blocked, f"{(blocked/total_projects)*100:.1f}%")
                    else:
                        st.warning(f"No data available for {selected_resource}.")
                
                # Company Workload Analysis Tab
                with resource_tabs[1]:
                    st.subheader("Company Workload Analysis")
                    
                    # Select which company column to analyze
                    # Try to identify company columns based on common names
                    company_columns = [col for col in data.columns if any(word in col.lower() for word in 
                                                                         ['company', 'supplier', 'vendor', 'contractor'])]
                    
                    if company_columns:
                        selected_company_col = st.selectbox(
                            "Select Company Type:",
                            company_columns
                        )
                        
                        # Get data for the selected company type
                        company_data = data[data[selected_company_col].notna()].copy()
                        
                        if not company_data.empty:
                            # Count projects per company
                            company_counts = company_data[selected_company_col].value_counts().reset_index()
                            company_counts.columns = ['Company', 'Project Count']
                            
                            # Sort by count descending
                            company_counts = company_counts.sort_values('Project Count', ascending=False)
                            
                            # Limit to top 20 companies for better visualization
                            if len(company_counts) > 20:
                                top_companies = company_counts.head(20)['Company'].tolist()
                                st.info(f"Showing top 20 companies out of {len(company_counts)} total.")
                                # Filter data to only include top companies
                                filtered_company_data = company_data[company_data[selected_company_col].isin(top_companies)]
                            else:
                                top_companies = company_counts['Company'].tolist()
                                filtered_company_data = company_data
                            
                            # Check if status column exists for stacked visualization
                            if 'NS\nStatus' in company_data.columns:
                                # Create stacked bar chart by status
                                st.subheader(f"Project Count by {selected_company_col} with Status Breakdown")
                                
                                # First, ensure there's a status for all rows (replace NaN with "Unknown")
                                filtered_company_data['NS\nStatus'] = filtered_company_data['NS\nStatus'].fillna("Unknown")
                                
                                # Create a cross-tab of company and status
                                status_crosstab = pd.crosstab(
                                    filtered_company_data[selected_company_col], 
                                    filtered_company_data['NS\nStatus']
                                ).reset_index()
                                
                                # Melt the dataframe for plotting
                                status_data = status_crosstab.melt(
                                    id_vars=[selected_company_col],
                                    var_name='Status',
                                    value_name='Count'
                                )
                                
                                # Sort by total count
                                company_order = company_counts[company_counts['Company'].isin(top_companies)].sort_values('Project Count', ascending=False)['Company'].tolist()
                                
                                # Create stacked bar chart with a unique key
                                fig = px.bar(
                                    status_data,
                                    x=selected_company_col,
                                    y='Count',
                                    color='Status',
                                    title=f"Project Count by {selected_company_col} with Status Breakdown",
                                    category_orders={selected_company_col: company_order}
                                )
                                
                                # Rotate x-axis labels for better readability
                                fig.update_layout(xaxis_tickangle=-45)
                                
                                st.plotly_chart(fig, use_container_width=True, key="company_status_chart")
                            else:
                                # Create standard bar chart if no status column
                                displayed_counts = company_counts.head(20) if len(company_counts) > 20 else company_counts
                                
                                # Create bar chart
                                fig = px.bar(
                                    displayed_counts,
                                    x='Company',
                                    y='Project Count',
                                    title=f"Project Count by {selected_company_col}",
                                    color='Project Count',
                                    color_continuous_scale='Viridis'
                                )
                                
                                # Rotate x-axis labels for better readability
                                fig.update_layout(xaxis_tickangle=-45)
                                
                                st.plotly_chart(fig, use_container_width=True, key="company_count_chart")
                            
                            # Project type distribution for selected company
                            if 'Site Type' in company_data.columns:
                                # Select a specific company to analyze
                                top_companies = company_counts['Company'].head(10).tolist()
                                selected_company = st.selectbox(
                                    f"Select {selected_company_col} to analyze:",
                                    top_companies
                                )
                                
                                # Filter data for the selected company
                                company_specific_data = company_data[company_data[selected_company_col] == selected_company]
                                
                                # Create site type distribution chart
                                site_type_counts = company_specific_data['Site Type'].value_counts().reset_index()
                                site_type_counts.columns = ['Site Type', 'Count']
                                
                                st.subheader(f"Project Types for {selected_company}")
                                fig = px.pie(
                                    site_type_counts,
                                    values='Count',
                                    names='Site Type',
                                    title=f"Site Type Distribution for {selected_company}"
                                )
                                st.plotly_chart(fig, use_container_width=True, key="company_site_type_pie")
                        else:
                            st.warning(f"No data available for {selected_company_col}.")
                    else:
                        st.warning("No company columns identified in the dataset.")
                
                # Project Allocation Tab
                with resource_tabs[2]:
                    st.subheader("Project Resource Allocation")
                    
                    # Select a specific project to analyze
                    if 'KTL Project Name' in data.columns:
                        project_names = data['KTL Project Name'].dropna().unique().tolist()
                        project_names = sorted([str(name) for name in project_names if str(name).strip()])
                        
                        selected_project = st.selectbox(
                            "Select Project to Analyze:",
                            project_names
                        )
                        
                        # Filter data for the selected project
                        project_data = data[data['KTL Project Name'] == selected_project].copy()
                        
                        if not project_data.empty:
                            # Create resource allocation table
                            st.subheader(f"Resource Allocation for {selected_project}")
                            
                            # Display basic project info
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if 'NS\nStatus' in project_data.columns:
                                    status = project_data['NS\nStatus'].iloc[0]
                                    st.metric("Project Status", status)
                            
                            with col2:
                                if 'Site Type' in project_data.columns:
                                    site_type = project_data['Site Type'].iloc[0]
                                    st.metric("Site Type", site_type)
                            
                            with col3:
                                if 'Client Priority' in project_data.columns:
                                    priority = project_data['Client Priority'].iloc[0]
                                    st.metric("Priority", priority)
                            
                            # Create resource allocation table
                            st.subheader("Resource Team")
                            
                            # Gather all resource info for this project
                            resource_info = {}
                            for col in existing_resource_columns:
                                if col in project_data.columns and not project_data[col].isna().all():
                                    resource_info[col] = project_data[col].iloc[0]

                            if resource_info:
                                resource_df = pd.DataFrame({
                                    'Resource Type': list(resource_info.keys()),
                                    'Assigned To': list(resource_info.values())
                                })
                                st.dataframe(resource_df, use_container_width=True)
                            else:
                                st.info("No resource information available for this project.")
                        else:
                            st.warning(f"No data available for project {selected_project}.")
                    else:
                        st.warning("No project names found in the dataset.")

        with tab7:
            st.header("Project Metrics")
            
            # Check for NS Status column and provide column selection if not found
            status_column = None
            
            if 'NS\nStatus' in data.columns:
                status_column = 'NS\nStatus'
                st.success("Found 'NS Status' column in the dataset.")
            else:
                st.warning("The column named exactly 'NS Status' was not found in your dataset.")
                # Show all column names to help identify the actual status column
                st.subheader("Available Columns")
                st.write("Here are the first 15 columns in your dataset:")
                st.write(data.columns[:15].tolist())
                
                # Try to find columns that might be the status column
                possible_status_cols = [col for col in data.columns if 'status' in col.lower() or 'state' in col.lower()]
                
                if possible_status_cols:
                    st.write("These columns might contain status information:")
                    
                    # Let user select which column to use as status
                    status_column = st.selectbox(
                        "Select a column to use as status:",
                        options=['None'] + possible_status_cols,
                        index=0
                    )
                    
                    if status_column == 'None':
                        status_column = None
                    else:
                        st.success(f"Using '{status_column}' as the status column")
                else:
                    st.error("No columns with 'status' or 'state' in their names were found.")
            
            # Calculate key metrics
            col1, col2, col3 = st.columns(3)
            
            # 1. Instructed Pot (count of NS Status = In Progress)
            instructed_pot = 0
            if status_column in data.columns:
                instructed_pot = len(data[data[status_column] == 'In Progress'])
            
            # 2. DD Issued (count of DD Issued Client (A) with values)
            dd_issued = 0
            if 'DD Issued Client (A)' in data.columns:
                dd_issued = len(data[data['DD Issued Client (A)'].notna()])
            
            # 3. DD Pending Issue (Instructed Pot - DD Issued)
            dd_pending = instructed_pot - dd_issued
            
            # Display metrics
            with col1:
                st.metric("Instructed Pot", instructed_pot)
                st.markdown("*Projects with 'In Progress' status*")
            
            with col2:
                st.metric("DD Issued", dd_issued)
                st.markdown("*Projects with DD already issued to client*")
            
            with col3:
                st.metric("DD Pending Issue", dd_pending)
                st.markdown("*Projects waiting for DD to be issued*")
            
            # If status column is available, show the rest of the content
            if status_column:
                # Show progress visualization
                st.subheader("DD Issuance Progress")
                
                # Calculate percentage completion
                if instructed_pot > 0:
                    completion_pct = (dd_issued / instructed_pot) * 100
                else:
                    completion_pct = 0
                
                # Create progress bar
                st.progress(completion_pct / 100)
                st.markdown(f"**{completion_pct:.1f}%** of in-progress projects have had DD issued")
                
                # Create columns for pie chart and bar chart
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart showing DD issued vs pending
                    pie_data = pd.DataFrame({
                        'Status': ['DD Issued', 'DD Pending'],
                        'Count': [dd_issued, dd_pending]
                    })
                    
                    fig = px.pie(
                        pie_data,
                        values='Count',
                        names='Status',
                        title='DD Issuance Status',
                        color='Status',
                        color_discrete_map={
                            'DD Issued': '#00BFA5',  # Bright teal
                            'DD Pending': '#FF6E40'  # Bright orange
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True, key="dd_issuance_status_pie_metrics")
                
                with col2:
                    # Bar chart comparing instructed pot vs DD issued
                    bar_data = pd.DataFrame({
                        'Category': ['Instructed Pot', 'DD Issued', 'DD Pending'],
                        'Count': [instructed_pot, dd_issued, dd_pending]
                    })
                    
                    fig = px.bar(
                        bar_data,
                        x='Category',
                        y='Count',
                        title='Project Metrics Comparison',
                        color='Category',
                        color_discrete_map={
                            'Instructed Pot': '#64B5F6',  # Bright blue
                            'DD Issued': '#00BFA5',      # Bright teal
                            'DD Pending': '#FF6E40'      # Bright orange
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True, key="project_metrics_comparison_metrics")
                
                # Site Type breakdown for DD Pending projects
                if all(col in data.columns for col in ['Site Type', 'DD Issued Client (A)']):
                    st.subheader("DD Pending by Site Type")
                    
                    # Filter for in progress projects that don't have DD issued
                    pending_projects = data[
                        (data[status_column] == 'In Progress') & 
                        (data['DD Issued Client (A)'].isna())
                    ]
                    
                    site_type_counts = pending_projects['Site Type'].value_counts().reset_index()
                    site_type_counts.columns = ['Site Type', 'Pending Count']
                    
                    # Create horizontal bar chart sorted by count
                    site_type_counts = site_type_counts.sort_values('Pending Count', ascending=True)
                    
                    fig = px.bar(
                        site_type_counts,
                        y='Site Type',
                        x='Pending Count',
                        title='DD Pending by Site Type',
                        orientation='h',
                        color='Pending Count',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="dd_pending_by_site_type_metrics")
                
                # Timeline projection
                st.subheader("DD Issuance Timeline Projection")
                
                # Calculate average time from in progress to DD issuance (for completed projects)
                if all(col in data.columns for col in ['DD Issued Client (A)', 'Instruction Date']):
                    # Filter for projects that have both instruction date and DD issued date
                    completed_projects = data[
                        (data['DD Issued Client (A)'].notna()) & 
                        (data['Instruction Date'].notna())
                    ].copy()
                    
                    if not completed_projects.empty:
                        # Calculate duration in days
                        completed_projects['Duration_Days'] = (
                            completed_projects['DD Issued Client (A)'] - 
                            completed_projects['Instruction Date']
                        ).dt.days
                        
                        # Filter out negative durations (likely data errors)
                        completed_projects = completed_projects[completed_projects['Duration_Days'] >= 0]
                        
                        if not completed_projects.empty:
                            # Calculate average duration
                            avg_duration = completed_projects['Duration_Days'].median()  # Using median to avoid outlier effects
                            
                            # Create projection chart
                            st.markdown(f"Based on historical data, projects take an average of **{avg_duration:.0f} days** from instruction to DD issuance.")
                            
                            # Make sure we have the required columns for pending projects
                            if all(col in data.columns for col in ['DD Issued Client (A)', 'Instruction Date']):
                                # Filter for pending projects with instruction dates
                                pending_projects = data[
                                    (data[status_column] == 'In Progress') & 
                                    (data['DD Issued Client (A)'].isna())
                                ]
                                
                                # Show projects that are approaching or have exceeded the average duration
                                pending_with_dates = pending_projects[pending_projects['Instruction Date'].notna()].copy()
                                
                                if not pending_with_dates.empty:
                                    # Calculate days since instruction
                                    pending_with_dates['Days_Since_Instruction'] = (
                                        pd.Timestamp.now() - 
                                        pending_with_dates['Instruction Date']
                                    ).dt.days
                                    
                                    # Calculate percentage of average duration
                                    pending_with_dates['Percent_of_Avg'] = (
                                        pending_with_dates['Days_Since_Instruction'] / avg_duration
                                    ) * 100
                                    
                                    # Flag projects as: On Track, At Risk, or Overdue
                                    pending_with_dates['Status'] = pending_with_dates['Percent_of_Avg'].apply(
                                        lambda x: 'On Track' if x < 80 else ('At Risk' if x < 100 else 'Overdue')
                                    )
                                    
                                    # Count by status
                                    status_counts = pending_with_dates['Status'].value_counts().reset_index()
                                    status_counts.columns = ['Status', 'Count']
                                    
                                    # Create horizontal bar chart
                                    fig = px.bar(
                                        status_counts,
                                        y='Status',
                                        x='Count',
                                        title='DD Pending Projects Timeline Status',
                                        orientation='h',
                                        color='Status',
                                        color_discrete_map={
                                            'On Track': '#00BFA5',    # Bright teal
                                            'At Risk': '#FFB74D',     # Bright amber
                                            'Overdue': '#FF5252'      # Bright red
                                        },
                                        category_orders={"Status": ["On Track", "At Risk", "Overdue"]}
                                    )
                                    st.plotly_chart(fig, use_container_width=True, key="dd_pending_timeline_status_metrics")
                                    
                                    # Show table of overdue projects
                                    overdue_projects = pending_with_dates[pending_with_dates['Status'] == 'Overdue']
                                    if not overdue_projects.empty:
                                        st.subheader("Overdue Projects")
                                        
                                        # Format the display dataframe
                                        display_columns = ['Site ID', 'Site Type', 'Instruction Date', 'Days_Since_Instruction']
                                        display_df = overdue_projects[
                                            [col for col in display_columns if col in overdue_projects.columns]
                                        ].copy()
                                        
                                        # Rename columns for better display
                                        col_rename = {
                                            'Days_Since_Instruction': 'Days Since Instruction'
                                        }
                                        display_df.rename(columns=col_rename, inplace=True)
                                        
                                        # Sort by days since instruction (descending)
                                        display_df = display_df.sort_values('Days Since Instruction', ascending=False)
                                        
                                        # Show the table
                                        st.dataframe(display_df, use_container_width=True)
                                else:
                                    st.info("No pending projects with instruction dates found.")
                            else:
                                st.info("Missing required columns to calculate pending project timelines.")
                        else:
                            st.info("No valid duration data available after filtering.")
                    else:
                        st.info("No projects with both instruction and DD issued dates found.")
                else:
                    st.info("Missing required columns for timeline projection analysis.")
            else:
                # If no status column is available/selected, provide helpful information
                st.warning("""
                To use this tab properly, you need to select a column containing status information. 
                Without this, we can't calculate the number of in-progress projects or pending DDs.
                
                Look for a column in your dataset that contains values like 'In Progress', 'Closed', etc.
                """)
                
                # Display info about all columns
                with st.expander("Show all columns in your dataset"):
                    st.write(data.columns.tolist())
            
            # Always show information about DD Issued column
            if 'DD Issued Client (A)' not in data.columns:
                st.warning("Note: The 'DD Issued Client (A)' column is not available in your dataset. This tab requires this column to calculate metrics properly.")

if __name__ == "__main__":
    main() 