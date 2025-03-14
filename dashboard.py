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

# Custom CSS to style the sidebar
sidebar_style = """
<style>
    [data-testid="stSidebar"] {
        background-color: #142656;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stSelectbox span {
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white;
        color: black;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: white;
    }
    [data-testid="stSidebar"] .stSuccess {
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Sticky sidebar header */
    .sidebar-header {
        position: sticky;
        top: 0;
        background-color: #142656;
        padding-bottom: 10px;
        z-index: 999;
    }

    /* Milestone box styles */
    .milestone-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    .milestone-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #142656;
    }
    .milestone-counts {
        display: flex;
        justify-content: space-between;
    }
    .forecast-box {
        background-color: #e6f2ff;
        padding: 10px;
        border-radius: 3px;
        flex: 1;
        margin-right: 2px;
    }
    .actual-box {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 3px;
        flex: 1;
        margin-left: 2px;
    }
    .count-number {
        font-size: 1.5em;
        font-weight: bold;
    }
    .count-label {
        font-size: 0.8em;
        color: #666;
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
        main_df = pd.read_csv("Cornerstone - A&D - Design Master Tracker.csv", low_memory=False)

        additional_files = [
            {"file": "KTL - Internal Electricals Workbook - Electricals Pending.csv", "name": "Electricals"},
            {"file": "KTL - Internal Structural Workbook - Structural to be Completed.csv", "name": "Structural"},
            {"file": "Design - Dependencies - STATS Required.csv", "name": "Dependencies"}
        ]

        merged_df = main_df.copy()

        for file_info in additional_files:
            try:
                df = pd.read_csv(file_info["file"], low_memory=False)
                connection_field = "Site ID"

                if connection_field in df.columns and connection_field in merged_df.columns:
                    exclude_cols = [col for col in df.columns if col in merged_df.columns and col != connection_field]
                    include_cols = [col for col in df.columns if col not in exclude_cols or col == connection_field]

                    rename_dict = {col: f"{file_info['name']}_{col}" for col in include_cols if col != connection_field}
                    df_to_merge = df[include_cols].rename(columns=rename_dict)

                    merged_df = pd.merge(merged_df, df_to_merge, on=connection_field, how="left")
                    notifications.append(f"✅ Merged {file_info['name']} data successfully.")
                else:
                    notifications.append(f"⚠️ '{connection_field}' not found in both dataframes for {file_info['name']}.")
            except Exception as e:
                notifications.append(f"⚠️ Could not load {file_info['name']} data: {e}")

        notifications.append(f"ℹ️ Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")

        return merged_df, notifications

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
                                                           'nan', 'NaT', 'NaN', '', 'None'], pd.NA)
                
                # Convert to datetime with day first format (UK date format)
                data[col] = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
                
                # Make sure 1/1/1900 dates are set to NaT if pandas parsed them
                jan_1_1900 = pd.Timestamp('1900-01-01')
                data.loc[data[col] == jan_1_1900, col] = pd.NaT
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
        
        # Date Range Filter
        st.sidebar.subheader("Date Range Filter")
        
        # Select date filter granularity
        date_granularity = st.sidebar.radio(
            "Select Date Granularity", 
            ["Month-Year", "Week-Month-Year", "Week-Year"],
            horizontal=True,
            key="date_granularity"
        )
        
        # Find all date columns that have valid dates
        date_columns_with_data = []
        for col in data.columns:
            if pd.api.types.is_datetime64_dtype(data[col]) and data[col].notna().sum() > 0:
                date_columns_with_data.append(col)
        
        # Select which date column to filter by
        if date_columns_with_data:
            selected_date_column = st.sidebar.selectbox(
                "Select Date Column for Filtering",
                date_columns_with_data,
                key="date_column"
            )
            
            # Get min and max dates from the selected column
            valid_dates = data[selected_date_column].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                
                # Create date range selector based on granularity
                if date_granularity == "Month-Year":
                    # Create a list of month-year options
                    months = pd.date_range(
                        start=pd.Timestamp(min_date.replace(day=1)),
                        end=pd.Timestamp(max_date),
                        freq='MS'  # Month Start
                    )
                    month_options = [d.strftime('%b %Y') for d in months]
                    
                    # Select start and end month-year
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        start_month_idx = st.selectbox("Start Month", 
                                                     range(len(month_options)), 
                                                     format_func=lambda x: month_options[x],
                                                     index=0 if st.session_state.reset_filters else st.session_state.get("start_month_idx", 0),
                                                     key="start_month_idx")
                    with col2:
                        end_month_idx = st.selectbox("End Month", 
                                                   range(len(month_options)), 
                                                   format_func=lambda x: month_options[x],
                                                   index=len(month_options)-1 if st.session_state.reset_filters else st.session_state.get("end_month_idx", len(month_options)-1),
                                                   key="end_month_idx")
                    
                    # Convert selected indices to dates
                    start_date = pd.to_datetime(month_options[start_month_idx]).date()
                    # Set end date to the last day of the selected month
                    end_month = pd.to_datetime(month_options[end_month_idx])
                    end_date = (end_month + pd.offsets.MonthEnd(1)).date()
                
                elif date_granularity == "Week-Month-Year":
                    # Create a list of week options
                    weeks = pd.date_range(
                        start=pd.Timestamp(min_date),
                        end=pd.Timestamp(max_date),
                        freq='W-MON'  # Weekly, starting on Monday
                    )
                    week_options = [f"Week {d.isocalendar()[1]} - {d.strftime('%b %Y')}" for d in weeks]
                    
                    # Select start and end week
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        start_week_idx = st.selectbox("Start Week", 
                                                    range(len(week_options)), 
                                                    format_func=lambda x: week_options[x],
                                                    index=0 if st.session_state.reset_filters else st.session_state.get("start_week_idx", 0),
                                                    key="start_week_idx")
                    with col2:
                        end_week_idx = st.selectbox("End Week", 
                                                  range(len(week_options)), 
                                                  format_func=lambda x: week_options[x],
                                                  index=len(week_options)-1 if st.session_state.reset_filters else st.session_state.get("end_week_idx", len(week_options)-1),
                                                  key="end_week_idx")
                    
                    # Convert selected indices to dates
                    start_date = weeks[start_week_idx].date()
                    end_date = (weeks[end_week_idx] + pd.Timedelta(days=6)).date()
                
                else:  # Week-Year
                    # Create a list of week-year options
                    weeks = pd.date_range(
                        start=pd.Timestamp(min_date),
                        end=pd.Timestamp(max_date),
                        freq='W-MON'  # Weekly, starting on Monday
                    )
                    week_options = [f"Week {d.isocalendar()[1]} - {d.year}" for d in weeks]
                    
                    # Select start and end week-year
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        start_week_idx = st.selectbox("Start Week", 
                                                    range(len(week_options)), 
                                                    format_func=lambda x: week_options[x],
                                                    index=0 if st.session_state.reset_filters else st.session_state.get("start_week_idx", 0),
                                                    key="start_week_idx")
                    with col2:
                        end_week_idx = st.selectbox("End Week", 
                                                  range(len(week_options)), 
                                                  format_func=lambda x: week_options[x],
                                                  index=len(week_options)-1 if st.session_state.reset_filters else st.session_state.get("end_week_idx", len(week_options)-1),
                                                  key="end_week_idx")
                    
                    # Convert selected indices to dates
                    start_date = weeks[start_week_idx].date()
                    end_date = (weeks[end_week_idx] + pd.Timedelta(days=6)).date()
                
                # Show selected date range
                st.sidebar.info(f"Selected Range: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
                
                # Apply date filter to the data
                data = data[(data[selected_date_column].dt.date >= start_date) & 
                            (data[selected_date_column].dt.date <= end_date)]
                
                # Display count of filtered records
                st.sidebar.success(f"Showing {len(data)} records in selected date range.")
        
        # Filter by NS Status
        if 'NS Status' in data.columns:
            status_options = ['All'] + sorted(data['NS Status'].dropna().unique().tolist())
            selected_status = st.sidebar.selectbox(
                'NS Status', 
                status_options,
                index=0 if st.session_state.reset_filters else st.session_state.get("ns_status_index", 0),
                key="ns_status"
            )
            
            if selected_status != 'All':
                data = data[data['NS Status'] == selected_status]
        
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
            for key in ["date_granularity", "date_column", "start_month_idx", "end_month_idx", 
                        "start_week_idx", "end_week_idx", "ns_status", "site_type", 
                        "priority", "project"]:
                if key in st.session_state:
                    del st.session_state[key]
            return
        
        clear_filters = st.sidebar.button('🔄 Clear All Filters', on_click=reset_filters)
        
        # Reset the flag after all filters have been processed
        if st.session_state.reset_filters:
            st.session_state.reset_filters = False
            st.rerun()
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, new_tab, dd_tab = st.tabs([
            "Overview", "Project Status", "Timeline Analysis", "KPI Metrics", 
            "Resources", "Data Integration", "Project Metrics", "Detailed Data", 
            "Weekly Metrics", "Detailed Design"
        ])
        
        with tab1:
            st.header("Overview")
            
            # Milestone summary boxes
            st.subheader("Major Milestone Summary")
            
            # Define milestone pairs (forecast and actual)
            milestone_pairs = [
                ("RMSV", "RMSV DATE (F)", "RMSV(A)"),
                ("GA", "GA\n(F)", "GA Issued Client (A)"),
                ("DD", "DD\n(F)", "DD Issued Client (A)")
            ]
            
            # Create a 3-column layout for the milestone boxes
            cols = st.columns(3)
            
            # Add milestone boxes to the columns
            for i, (title, forecast_col, actual_col) in enumerate(milestone_pairs):
                with cols[i]:
                    milestone_html = create_milestone_box(title, forecast_col, actual_col, data)
                    st.markdown(milestone_html, unsafe_allow_html=True)
            
            # Key metrics
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Projects", len(data))
            
            with col2:
                if 'NS Status' in data.columns:
                    in_progress = len(data[data['NS Status'] == 'In Progress'])
                    st.metric("In Progress Projects", in_progress)
            
            with col3:
                if 'NS Status' in data.columns:
                    closed = len(data[data['NS Status'] == 'Closed'])
                    st.metric("Closed Projects", closed)
            
            with col4:
                if 'Blocker Status' in data.columns:
                    blocked = len(data[data['Blocker Status'].notna() & (data['Blocker Status'] != 'Cleared')])
                    st.metric("Blocked Projects", blocked)
            
            # Status distribution chart
            if 'NS Status' in data.columns:
                st.subheader("Project Status Distribution")
                status_counts = data['NS Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                
                fig = px.pie(status_counts, values='Count', names='Status', 
                            title='Project Status Distribution',
                            color_discrete_sequence=px.colors.qualitative.Plotly)
                st.plotly_chart(fig, use_container_width=True)
            
            # Site Type distribution
            if 'Site Type' in data.columns:
                st.subheader("Site Type Distribution")
                site_type_counts = data['Site Type'].value_counts().reset_index()
                site_type_counts.columns = ['Site Type', 'Count']
                
                fig = px.bar(site_type_counts, x='Site Type', y='Count',
                            title='Site Type Distribution',
                            color='Count',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
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
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Blocker Reason Distribution
                    blocker_reason_counts = blocked_data['Blocker Reason'].value_counts().reset_index()
                    blocker_reason_counts.columns = ['Blocker Reason', 'Count']
                    
                    fig = px.pie(blocker_reason_counts, values='Count', names='Blocker Reason',
                                title='Blocker Reason Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Priority analysis
            if 'Client Priority' in data.columns:
                st.subheader("Priority Analysis")
                
                priority_counts = data['Client Priority'].value_counts().reset_index()
                priority_counts.columns = ['Priority', 'Count']
                
                fig = px.bar(priority_counts, x='Priority', y='Count',
                            title='Project Distribution by Priority',
                            color='Count',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            # Status by Site Type (if both columns exist)
            if all(col in data.columns for col in ['NS Status', 'Site Type']):
                st.subheader("Status by Site Type")
                
                # Create a cross-tabulation of Site Type and NS Status
                cross_tab = pd.crosstab(data['Site Type'], data['NS Status'])
                
                # Convert to long format for plotting
                cross_tab_long = cross_tab.reset_index().melt(id_vars=['Site Type'], 
                                                            var_name='Status', 
                                                            value_name='Count')
                
                fig = px.bar(cross_tab_long, x='Site Type', y='Count', color='Status',
                            title='Project Status by Site Type',
                            barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Timeline Analysis")
            
            # Instruction Date analysis
            if 'Instruction Date' in data.columns:
                st.subheader("Projects by Instruction Date")
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data['Instruction Date']):
                    data['Instruction Date'] = pd.to_datetime(data['Instruction Date'], errors='coerce')
                
                # Filter out rows with missing Instruction Date
                date_data = data[data['Instruction Date'].notna()].copy()
                
                if not date_data.empty:
                    # Extract month and year
                    date_data['Month-Year'] = date_data['Instruction Date'].dt.strftime('%Y-%m')
                    
                    # Count projects by month-year
                    monthly_counts = date_data['Month-Year'].value_counts().sort_index().reset_index()
                    monthly_counts.columns = ['Month-Year', 'Count']
                    
                    fig = px.line(monthly_counts, x='Month-Year', y='Count',
                                title='Projects by Instruction Date',
                                markers=True)
                    fig.update_layout(xaxis_title="Month-Year", yaxis_title="Number of Projects")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid instruction date data available.")
            
            # Project duration analysis
            st.subheader("Project Duration Analysis")
            
            # Identify columns that might contain start and end dates
            potential_start_cols = [col for col in data.columns if any(word in col.lower() for word in ['start', 'instruction', 'initiated'])]
            potential_end_cols = [col for col in data.columns if any(word in col.lower() for word in ['end', 'close', 'completed', 'approved'])]
            
            # If we can identify start and end date columns, calculate duration
            if potential_start_cols and potential_end_cols:
                start_col = st.selectbox("Select start date column:", potential_start_cols)
                end_col = st.selectbox("Select end date column:", potential_end_cols)
                
                # Convert columns to datetime
                try:
                    data[start_col] = pd.to_datetime(data[start_col], errors='coerce')
                    data[end_col] = pd.to_datetime(data[end_col], errors='coerce')
                    
                    # Calculate duration for completed projects
                    duration_data = data[(data[start_col].notna()) & (data[end_col].notna())].copy()
                    
                    if not duration_data.empty:
                        duration_data['Duration (days)'] = (duration_data[end_col] - duration_data[start_col]).dt.days
                        duration_data = duration_data[duration_data['Duration (days)'] >= 0]  # Filter out negative durations
                        
                        if not duration_data.empty:
                            # Histogram of project durations
                            fig = px.histogram(duration_data, x='Duration (days)',
                                            title=f'Project Duration Distribution ({start_col} to {end_col})',
                                            nbins=30)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Duration (days)", f"{duration_data['Duration (days)'].mean():.1f}")
                            with col2:
                                st.metric("Median Duration (days)", f"{duration_data['Duration (days)'].median():.1f}")
                            with col3:
                                st.metric("Max Duration (days)", f"{duration_data['Duration (days)'].max():.1f}")
                        else:
                            st.info("No valid duration data available (after filtering out negative durations).")
                    else:
                        st.info("No projects with both start and end dates available.")
                except:
                    st.error("Error calculating project durations. Please check the selected date columns.")
        
        with tab4:
            st.header("KPI Metrics")
            
            # Define the milestone columns to analyze
            milestone_columns = {
                "RMSV": "RMSV(A)",
                "GA Issued": "GA Issued Client (A)",
                "DD Issued": "DD Issued Client (A)"
            }
            
            # Ensure we have the milestone columns
            milestone_cols_exist = all(col in data.columns for col in milestone_columns.values())
            
            if milestone_cols_exist:
                st.subheader("Milestone Durations")
                
                # Create dataframe for duration analysis
                duration_df = data.copy()
                
                # Calculate durations between milestones
                duration_metrics = []
                
                # RMSV to GA Issued
                if pd.api.types.is_datetime64_dtype(duration_df[milestone_columns["RMSV"]]) and \
                   pd.api.types.is_datetime64_dtype(duration_df[milestone_columns["GA Issued"]]):
                    
                    # Filter data where both dates exist
                    valid_data = duration_df[
                        (duration_df[milestone_columns["RMSV"]].notna()) & 
                        (duration_df[milestone_columns["GA Issued"]].notna())
                    ].copy()
                    
                    if not valid_data.empty:
                        # Calculate duration in days
                        valid_data['RMSV_to_GA_days'] = (valid_data[milestone_columns["GA Issued"]] - 
                                                        valid_data[milestone_columns["RMSV"]]).dt.days
                        
                        # Filter out negative durations (likely data errors)
                        valid_data = valid_data[valid_data['RMSV_to_GA_days'] >= 0]
                        
                        if not valid_data.empty:
                            duration_metrics.append({
                                "From": "RMSV",
                                "To": "GA Issued",
                                "Mean Days": valid_data['RMSV_to_GA_days'].mean(),
                                "Median Days": valid_data['RMSV_to_GA_days'].median(),
                                "Min Days": valid_data['RMSV_to_GA_days'].min(),
                                "Max Days": valid_data['RMSV_to_GA_days'].max(),
                                "Sample Size": len(valid_data),
                                "Data": valid_data
                            })
                
                # RMSV to DD Issued
                if pd.api.types.is_datetime64_dtype(duration_df[milestone_columns["RMSV"]]) and \
                   pd.api.types.is_datetime64_dtype(duration_df[milestone_columns["DD Issued"]]):
                    
                    # Filter data where both dates exist
                    valid_data = duration_df[
                        (duration_df[milestone_columns["RMSV"]].notna()) & 
                        (duration_df[milestone_columns["DD Issued"]].notna())
                    ].copy()
                    
                    if not valid_data.empty:
                        # Calculate duration in days
                        valid_data['RMSV_to_DD_days'] = (valid_data[milestone_columns["DD Issued"]] - 
                                                        valid_data[milestone_columns["RMSV"]]).dt.days
                        
                        # Filter out negative durations (likely data errors)
                        valid_data = valid_data[valid_data['RMSV_to_DD_days'] >= 0]
                        
                        if not valid_data.empty:
                            duration_metrics.append({
                                "From": "RMSV",
                                "To": "DD Issued",
                                "Mean Days": valid_data['RMSV_to_DD_days'].mean(),
                                "Median Days": valid_data['RMSV_to_DD_days'].median(),
                                "Min Days": valid_data['RMSV_to_DD_days'].min(),
                                "Max Days": valid_data['RMSV_to_DD_days'].max(),
                                "Sample Size": len(valid_data),
                                "Data": valid_data
                            })
                
                # GA Issued to DD Issued
                if pd.api.types.is_datetime64_dtype(duration_df[milestone_columns["GA Issued"]]) and \
                   pd.api.types.is_datetime64_dtype(duration_df[milestone_columns["DD Issued"]]):
                    
                    # Filter data where both dates exist
                    valid_data = duration_df[
                        (duration_df[milestone_columns["GA Issued"]].notna()) & 
                        (duration_df[milestone_columns["DD Issued"]].notna())
                    ].copy()
                    
                    if not valid_data.empty:
                        # Calculate duration in days
                        valid_data['GA_to_DD_days'] = (valid_data[milestone_columns["DD Issued"]] - 
                                                     valid_data[milestone_columns["GA Issued"]]).dt.days
                        
                        # Filter out negative durations (likely data errors)
                        valid_data = valid_data[valid_data['GA_to_DD_days'] >= 0]
                        
                        if not valid_data.empty:
                            duration_metrics.append({
                                "From": "GA Issued",
                                "To": "DD Issued",
                                "Mean Days": valid_data['GA_to_DD_days'].mean(),
                                "Median Days": valid_data['GA_to_DD_days'].median(),
                                "Min Days": valid_data['GA_to_DD_days'].min(),
                                "Max Days": valid_data['GA_to_DD_days'].max(),
                                "Sample Size": len(valid_data),
                                "Data": valid_data
                            })
                
                # Display milestone duration summary
                if duration_metrics:
                    # Create a summary metrics table
                    summary_data = [{
                        "Milestone Path": f"{m['From']} → {m['To']}",
                        "Avg Days": f"{m['Mean Days']:.1f}",
                        "Median Days": f"{m['Median Days']:.1f}",
                        "Min Days": f"{m['Min Days']:.0f}",
                        "Max Days": f"{m['Max Days']:.0f}",
                        "Count": m['Sample Size']
                    } for m in duration_metrics]
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Create visualizations for each metric
                    st.subheader("Duration Distributions")
                    
                    # Select which duration to visualize
                    duration_options = [f"{m['From']} to {m['To']}" for m in duration_metrics]
                    selected_duration = st.selectbox("Select Duration to Visualize:", duration_options)
                    
                    # Find the selected duration data
                    selected_idx = duration_options.index(selected_duration)
                    selected_metric = duration_metrics[selected_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    # Column for histogram
                    with col1:
                        # Determine which duration column to use
                        if selected_duration == "RMSV to GA Issued":
                            duration_col = 'RMSV_to_GA_days'
                        elif selected_duration == "RMSV to DD Issued":
                            duration_col = 'RMSV_to_DD_days'
                        else:  # GA Issued to DD Issued
                            duration_col = 'GA_to_DD_days'
                        
                        # Create histogram
                        fig = px.histogram(
                            selected_metric['Data'], 
                            x=duration_col,
                            nbins=20,
                            title=f"Distribution of {selected_duration} Duration (days)",
                            labels={duration_col: "Duration (days)"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Column for box plot
                    with col2:
                        fig = px.box(
                            selected_metric['Data'],
                            y=duration_col,
                            title=f"Box Plot of {selected_duration} Duration (days)",
                            labels={duration_col: "Duration (days)"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Site Type breakdown for the selected duration (if available)
                    if 'Site Type' in selected_metric['Data'].columns:
                        st.subheader(f"{selected_duration} by Site Type")
                        
                        # Group by Site Type and calculate average duration
                        site_type_avg = selected_metric['Data'].groupby('Site Type')[duration_col].mean().reset_index()
                        site_type_avg.columns = ['Site Type', 'Average Duration (days)']
                        
                        # Sort by average duration
                        site_type_avg = site_type_avg.sort_values('Average Duration (days)', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            site_type_avg,
                            x='Site Type',
                            y='Average Duration (days)',
                            title=f"Average {selected_duration} Duration by Site Type",
                            color='Average Duration (days)',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Time trend analysis
                    st.subheader(f"{selected_duration} Trend Over Time")
                    
                    # Create a copy of data with start date and duration
                    trend_data = selected_metric['Data'].copy()
                    
                    # Use the "From" milestone as the reference date for the trend
                    from_col = milestone_columns[selected_metric['From']]
                    
                    # Extract year-month from the reference date
                    trend_data['Year-Month'] = trend_data[from_col].dt.strftime('%Y-%m')
                    
                    # Group by Year-Month and calculate average duration
                    monthly_avg = trend_data.groupby('Year-Month')[duration_col].mean().reset_index()
                    monthly_avg.columns = ['Year-Month', 'Average Duration (days)']
                    
                    # Sort by Year-Month
                    monthly_avg = monthly_avg.sort_values('Year-Month')
                    
                    # Create line chart
                    fig = px.line(
                        monthly_avg,
                        x='Year-Month',
                        y='Average Duration (days)',
                        title=f"Average {selected_duration} Duration Trend",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Month", yaxis_title="Average Duration (days)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No valid duration data found. Please ensure milestone dates are correctly formatted.")
            else:
                st.warning("Required milestone columns are missing from the dataset.")
                
            # Forecast vs Actual Analysis
            st.subheader("Forecast vs Actual Analysis")
            
            # Define milestone pairs for forecast vs actual analysis
            milestone_pairs = [
                ("RMSV", "RMSV DATE (F)", "RMSV(A)"),
                ("GA", "GA\n(F)", "GA Issued Client (A)"),
                ("DD", "DD\n(F)", "DD Issued Client (A)")
            ]
            
            # Select which milestone to analyze
            milestone_options = [pair[0] for pair in milestone_pairs]
            selected_milestone = st.selectbox("Select Milestone:", milestone_options)
            
            # Find the selected milestone data
            selected_pair = next((pair for pair in milestone_pairs if pair[0] == selected_milestone), None)
            
            if selected_pair and selected_pair[1] in data.columns and selected_pair[2] in data.columns:
                forecast_col = selected_pair[1]
                actual_col = selected_pair[2]
                
                # Check if columns are datetime
                if pd.api.types.is_datetime64_dtype(data[forecast_col]) and pd.api.types.is_datetime64_dtype(data[actual_col]):
                    # Filter data where both dates exist
                    valid_data = data[
                        (data[forecast_col].notna()) & 
                        (data[actual_col].notna())
                    ].copy()
                    
                    if not valid_data.empty:
                        # Calculate variance in days (positive means delayed, negative means early)
                        valid_data['Variance_days'] = (valid_data[actual_col] - valid_data[forecast_col]).dt.days
                        
                        # Statistics
                        avg_variance = valid_data['Variance_days'].mean()
                        median_variance = valid_data['Variance_days'].median()
                        on_time = sum(valid_data['Variance_days'] <= 0)
                        delayed = sum(valid_data['Variance_days'] > 0)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Average Variance (days)", f"{avg_variance:.1f}")
                        
                        with col2:
                            st.metric("Median Variance (days)", f"{median_variance:.1f}")
                        
                        with col3:
                            on_time_pct = (on_time / len(valid_data)) * 100
                            st.metric("On Time or Early", f"{on_time_pct:.1f}%")
                        
                        with col4:
                            delayed_pct = (delayed / len(valid_data)) * 100
                            st.metric("Delayed", f"{delayed_pct:.1f}%")
                        
                        # Distribution of variance
                        fig = px.histogram(
                            valid_data,
                            x='Variance_days',
                            title=f"Distribution of {selected_milestone} Variance (Actual - Forecast)",
                            labels={'Variance_days': 'Variance (days)'},
                            color_discrete_sequence=['#2c3e50']
                        )
                        # Add a vertical line at 0 (on time)
                        fig.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="On Time")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Variance by Site Type (if available)
                        if 'Site Type' in valid_data.columns:
                            st.subheader(f"{selected_milestone} Variance by Site Type")
                            
                            # Group by Site Type and calculate average variance
                            site_type_avg = valid_data.groupby('Site Type')['Variance_days'].mean().reset_index()
                            site_type_avg.columns = ['Site Type', 'Average Variance (days)']
                            
                            # Sort by average variance
                            site_type_avg = site_type_avg.sort_values('Average Variance (days)', ascending=True)
                            
                            # Calculate color based on variance (negative is good, positive is bad)
                            site_type_avg['Color'] = site_type_avg['Average Variance (days)'].apply(
                                lambda x: 'green' if x <= 0 else 'red'
                            )
                            
                            # Create bar chart
                            fig = px.bar(
                                site_type_avg,
                                x='Site Type',
                                y='Average Variance (days)',
                                title=f"Average {selected_milestone} Variance by Site Type",
                                color='Average Variance (days)',
                                color_continuous_scale='RdYlGn_r'  # Red for positive variance (delays), green for negative (early)
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="black")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No projects found with both forecast and actual dates for {selected_milestone}.")
                else:
                    st.warning(f"The columns for {selected_milestone} are not in the correct datetime format.")
            else:
                st.warning(f"Required columns for {selected_milestone} analysis are missing from the dataset.")

        with tab5:
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
                        if 'NS Status' in resource_data.columns:
                            # Create stacked bar chart by status
                            st.subheader(f"Project Count by {selected_resource} with Status Breakdown")
                            
                            # First, ensure there's a status for all rows (replace NaN with "Unknown")
                            filtered_resource_data['NS Status'] = filtered_resource_data['NS Status'].fillna("Unknown")
                            
                            # Create a cross-tab of resource and status
                            status_crosstab = pd.crosstab(
                                filtered_resource_data[selected_resource], 
                                filtered_resource_data['NS Status']
                            ).reset_index()
                            
                            # Melt the dataframe for plotting
                            status_data = status_crosstab.melt(
                                id_vars=[selected_resource],
                                var_name='Status',
                                value_name='Count'
                            )
                            
                            # Sort by total count
                            resource_order = resource_counts[resource_counts['Resource'].isin(top_resources)].sort_values('Project Count', ascending=False)['Resource'].tolist()
                            
                            # Create stacked bar chart
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
                            
                            st.plotly_chart(fig, use_container_width=True)
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Project Status Distribution for selected resource
                        if 'NS Status' in resource_data.columns:
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
                            status_counts = person_data['NS Status'].value_counts().reset_index()
                            status_counts.columns = ['Status', 'Count']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(
                                    status_counts,
                                    values='Count',
                                    names='Status',
                                    title=f"Status Distribution for {selected_person}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Calculate metrics
                                total_projects = len(person_data)
                                in_progress = len(person_data[person_data['NS Status'] == 'In Progress'])
                                completed = len(person_data[person_data['NS Status'] == 'Closed'])
                                
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
                            if 'NS Status' in company_data.columns:
                                # Create stacked bar chart by status
                                st.subheader(f"Project Count by {selected_company_col} with Status Breakdown")
                                
                                # First, ensure there's a status for all rows (replace NaN with "Unknown")
                                filtered_company_data['NS Status'] = filtered_company_data['NS Status'].fillna("Unknown")
                                
                                # Create a cross-tab of company and status
                                status_crosstab = pd.crosstab(
                                    filtered_company_data[selected_company_col], 
                                    filtered_company_data['NS Status']
                                ).reset_index()
                                
                                # Melt the dataframe for plotting
                                status_data = status_crosstab.melt(
                                    id_vars=[selected_company_col],
                                    var_name='Status',
                                    value_name='Count'
                                )
                                
                                # Sort by total count
                                company_order = company_counts[company_counts['Company'].isin(top_companies)].sort_values('Project Count', ascending=False)['Company'].tolist()
                                
                                # Create stacked bar chart
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
                                
                                st.plotly_chart(fig, use_container_width=True)
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
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
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
                                st.plotly_chart(fig, use_container_width=True)
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
                                if 'NS Status' in project_data.columns:
                                    status = project_data['NS Status'].iloc[0]
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

        with tab6:
            st.header("Data Integration Dashboard")
            
            # Create sections for each data source
            st.subheader("Data Source Integration")
            
            # Metrics row showing integration stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Main Dataset Records", len(data))
            
            # Count records from each source that were successfully merged
            with col2:
                electricals_count = sum("Electricals_Vendor" in data.columns and pd.notna(data["Electricals_Vendor"]))
                st.metric("Electrical Records Merged", electricals_count)
            
            with col3:
                structural_count = sum("Structural_Vendor" in data.columns and pd.notna(data["Structural_Vendor"]))
                st.metric("Structural Records Merged", structural_count)
            
            with col4:
                dependencies_count = sum("Dependencies_STATS Required" in data.columns and pd.notna(data["Dependencies_STATS Required"]))
                st.metric("Dependencies Records Merged", dependencies_count)
            
            # Create a visualization showing the overlap between datasets
            st.subheader("Data Integration Overlap")
            
            # Calculate overlaps
            has_electricals = sum("Electricals_Vendor" in data.columns and pd.notna(data["Electricals_Vendor"]))
            has_structural = sum("Structural_Vendor" in data.columns and pd.notna(data["Structural_Vendor"]))
            has_dependencies = sum("Dependencies_STATS Required" in data.columns and pd.notna(data["Dependencies_STATS Required"]))
            
            has_electricals_and_structural = sum(
                "Electricals_Vendor" in data.columns and 
                "Structural_Vendor" in data.columns and 
                pd.notna(data["Electricals_Vendor"]) & 
                pd.notna(data["Structural_Vendor"])
            )
            
            has_electricals_and_dependencies = sum(
                "Electricals_Vendor" in data.columns and 
                "Dependencies_STATS Required" in data.columns and 
                pd.notna(data["Electricals_Vendor"]) & 
                pd.notna(data["Dependencies_STATS Required"])
            )
            
            has_structural_and_dependencies = sum(
                "Structural_Vendor" in data.columns and 
                "Dependencies_STATS Required" in data.columns and 
                pd.notna(data["Structural_Vendor"]) & 
                pd.notna(data["Dependencies_STATS Required"])
            )
            
            has_all_three = sum(
                "Electricals_Vendor" in data.columns and 
                "Structural_Vendor" in data.columns and 
                "Dependencies_STATS Required" in data.columns and 
                pd.notna(data["Electricals_Vendor"]) & 
                pd.notna(data["Structural_Vendor"]) & 
                pd.notna(data["Dependencies_STATS Required"])
            )
            
            # Create a dataset for visualization
            integration_data = pd.DataFrame([
                {"Source": "Main Dataset Only", "Count": len(data) - (has_electricals + has_structural + has_dependencies) + has_electricals_and_structural + has_electricals_and_dependencies + has_structural_and_dependencies - 2*has_all_three},
                {"Source": "With Electrical Data", "Count": has_electricals},
                {"Source": "With Structural Data", "Count": has_structural},
                {"Source": "With Dependencies Data", "Count": has_dependencies},
                {"Source": "With All Three Sources", "Count": has_all_three}
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create bar chart of data integration
                fig = px.bar(
                    integration_data,
                    x="Source",
                    y="Count",
                    title="Data Integration by Source",
                    color="Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show the overlaps in a pie chart
                fig = px.pie(
                    integration_data,
                    values="Count",
                    names="Source",
                    title="Distribution of Integrated Data Sources"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Status breakdown for integrated data
            st.subheader("Status Analysis by Data Source")
            
            if 'NS Status' in data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Electrical data status breakdown
                    if "Electricals_Status" in data.columns:
                        electrical_status = data[pd.notna(data["Electricals_Status"])]["Electricals_Status"].value_counts().reset_index()
                        electrical_status.columns = ["Status", "Count"]
                        
                        fig = px.pie(
                            electrical_status,
                            values="Count",
                            names="Status",
                            title="Electrical Records by Status"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No electrical status data available")
                
                with col2:
                    # Structural data status breakdown
                    if "Structural_Status" in data.columns:
                        structural_status = data[pd.notna(data["Structural_Status"])]["Structural_Status"].value_counts().reset_index()
                        structural_status.columns = ["Status", "Count"]
                        
                        fig = px.pie(
                            structural_status,
                            values="Count",
                            names="Status",
                            title="Structural Records by Status"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Analysis of dependencies data
            if "Dependencies_STATS Required" in data.columns:
                st.subheader("Dependencies Analysis")
                
                stats_required = data["Dependencies_STATS Required"].value_counts().reset_index()
                stats_required.columns = ["STATS Required", "Count"]
                
                fig = px.pie(
                    stats_required,
                    values="Count",
                    names="STATS Required",
                    title="Distribution of STATS Requirements"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation between different vendor services
            st.subheader("Cross-Service Analysis")
            
            # Check if we have both electrical and structural vendors
            if all(col in data.columns for col in ["Electricals_Vendor", "Structural_Vendor"]):
                # Get records with both electrical and structural data
                combined_data = data[pd.notna(data["Electricals_Vendor"]) & pd.notna(data["Structural_Vendor"])]
                
                if not combined_data.empty:
                    # Create a cross-tabulation of vendors
                    vendor_crosstab = pd.crosstab(
                        combined_data["Electricals_Vendor"], 
                        combined_data["Structural_Vendor"]
                    )
                    
                    # Display heatmap of vendor relationships
                    fig = px.imshow(
                        vendor_crosstab,
                        labels=dict(x="Structural Vendor", y="Electrical Vendor", color="Count"),
                        title="Vendor Relationship Heatmap",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No records with both electrical and structural vendor data")
            else:
                st.info("Missing vendor data for cross-service analysis")

        with tab7:
            st.header("Project Metrics")
            
            # Check for NS Status column and provide column selection if not found
            status_column = None
            
            if 'NS Status' in data.columns:
                status_column = 'NS Status'
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
                        color_discrete_map={'DD Issued': '#2E7D32', 'DD Pending': '#FF5722'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
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
                            'Instructed Pot': '#1976D2', 
                            'DD Issued': '#2E7D32',
                            'DD Pending': '#FF5722'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
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
                    st.plotly_chart(fig, use_container_width=True)
                
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
                                            'On Track': '#4CAF50',
                                            'At Risk': '#FF9800',
                                            'Overdue': '#F44336'
                                        },
                                        category_orders={"Status": ["On Track", "At Risk", "Overdue"]}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
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

        with tab8:
            st.header("Detailed Data")
            
            # Show raw data with search functionality
            st.subheader("Raw Data")
            
            # Text search
            search_term = st.text_input("Search in data:")
            
            if search_term:
                # Search in all columns
                filtered_data = data[data.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
                st.dataframe(filtered_data)
            else:
                st.dataframe(data)
            
            # Download CSV button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name="filtered_cornerstone_data.csv",
                mime="text/csv",
            )

        with new_tab:
            st.header("Weekly Metrics Overview")

            # Example of how you might structure one of the tables
            # You would need to adjust this based on how your data is structured
            # and how you calculate the metrics shown in the image

            # Example data preparation for GA/FEAS
            ga_feas_data = {
                'Week No.': [10, 11, 12, 13],
                'W/C': ['03/03/2025', '10/03/2025', '17/03/2025', '24/03/2025'],
                'FC': [5, 5, 5, 5],
                'AC': [2, 0, 0, 0],
                'Delta': [-3, -5, -5, -5],
                'Cumulative': [-3, -8, -13, -18]
            }
            ga_feas_df = pd.DataFrame(ga_feas_data)

            # Display the table
            st.subheader("GA/FEAS Metrics")
            st.table(ga_feas_df)

            # Repeat the above for each category like DD, CNX Feasibility, etc.
            # You would need to prepare and display a DataFrame for each

        with dd_tab:
            st.header("Detailed Design Analysis")
            
            # Add DD process milestone counts at the top of Detailed Design tab
            st.subheader("Detailed Design Process Counts")
            
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
                    
                    # Show the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
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
                    
                    st.plotly_chart(monthly_fig, use_container_width=True)
                    
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
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig_pie = px.pie(
                        dd_status_counts,
                        values='Count',
                        names='Status',
                        title='DD Status Distribution (%)'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
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
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 