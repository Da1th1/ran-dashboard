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

# Title and description
st.title("Cornerstone A&D Design Dashboard")
st.markdown("This dashboard provides insights into the Cornerstone A&D Design Master Tracker data.")

# Load data
@st.cache_data
def load_data():
    try:
        # Load main dataset
        main_df = pd.read_csv("Cornerstone - A&D - Design Master Tracker.csv", low_memory=False)
        
        # Initialize a dictionary to store all dataframes
        all_dfs = {"Main": main_df}
        
        # Check and load the additional files
        additional_files = [
            {"file": "KTL - Internal Electricals Workbook - Electricals Pending.csv", "name": "Electricals"},
            {"file": "KTL - Internal Structural Workbook - Structural to be Completed.csv", "name": "Structural"},
            {"file": "Design - Dependencies - STATS Required.csv", "name": "Dependencies"}
        ]
        
        for file_info in additional_files:
            try:
                df = pd.read_csv(file_info["file"], low_memory=False)
                all_dfs[file_info["name"]] = df
                st.sidebar.success(f"Loaded {file_info['name']} data: {len(df)} records")
            except Exception as e:
                st.sidebar.warning(f"Could not load {file_info['name']} data: {e}")
        
        # Merge files if they contain Internal ID
        merged_df = main_df.copy()
        
        # For each additional dataframe, merge if it has Internal ID
        for name, df in all_dfs.items():
            if name == "Main":
                continue
                
            # Check if Internal ID column exists
            if "Internal ID" in df.columns:
                # Check if the main dataframe has Internal ID too
                if "Internal ID" not in merged_df.columns:
                    # If not, just add the new dataframes as separate entities
                    st.sidebar.warning(f"Main dataframe doesn't have 'Internal ID', cannot merge {name} data")
                    continue
                
                # Determine columns to merge (exclude duplicates except Internal ID)
                exclude_cols = [col for col in df.columns if col in merged_df.columns and col != "Internal ID"]
                include_cols = [col for col in df.columns if col not in exclude_cols or col == "Internal ID"]
                
                # Rename columns to avoid conflicts
                rename_dict = {col: f"{name}_{col}" for col in include_cols if col != "Internal ID"}
                df_to_merge = df[include_cols].copy()
                df_to_merge.rename(columns=rename_dict, inplace=True)
                
                # Merge with main dataframe
                merged_df = pd.merge(
                    merged_df, 
                    df_to_merge,
                    on="Internal ID",
                    how="left"
                )
                
                st.sidebar.success(f"Merged {name} data based on Internal ID")
            elif "Site ID" in df.columns and "Site ID" in merged_df.columns:
                # If no Internal ID, try using Site ID as alternative
                # Determine columns to merge (exclude duplicates except Site ID)
                exclude_cols = [col for col in df.columns if col in merged_df.columns and col != "Site ID"]
                include_cols = [col for col in df.columns if col not in exclude_cols or col == "Site ID"]
                
                # Rename columns to avoid conflicts
                rename_dict = {col: f"{name}_{col}" for col in include_cols if col != "Site ID"}
                df_to_merge = df[include_cols].copy()
                df_to_merge.rename(columns=rename_dict, inplace=True)
                
                # Merge with main dataframe
                merged_df = pd.merge(
                    merged_df, 
                    df_to_merge,
                    on="Site ID",
                    how="left"
                )
                
                st.sidebar.success(f"Merged {name} data based on Site ID")
            else:
                st.sidebar.warning(f"No common identifier found to merge {name} data")
        
        # Add info about data sources
        st.sidebar.info(f"Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
        
        return merged_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

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
    # Load and display the logo in the sidebar
    try:
        logo = Image.open("logo.png")
        st.sidebar.image(logo, width=150, caption="Cornerstone A&D")
    except Exception as e:
        st.sidebar.error(f"Error loading logo: {e}")
    
    df = load_data()
    
    if df is not None:
        data = prepare_data(df)
        
        # Show data loading status
        st.sidebar.success("Data loaded successfully!")
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Date Range Filter
        st.sidebar.subheader("Date Range Filter")
        
        # Select date filter granularity
        date_granularity = st.sidebar.radio(
            "Select Date Granularity", 
            ["Month-Year", "Week-Month-Year", "Week-Year"],
            horizontal=True
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
                date_columns_with_data
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
                                                     index=0)
                    with col2:
                        end_month_idx = st.selectbox("End Month", 
                                                   range(len(month_options)), 
                                                   format_func=lambda x: month_options[x],
                                                   index=len(month_options)-1)
                    
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
                                                    index=0)
                    with col2:
                        end_week_idx = st.selectbox("End Week", 
                                                  range(len(week_options)), 
                                                  format_func=lambda x: week_options[x],
                                                  index=len(week_options)-1)
                    
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
                                                    index=0)
                    with col2:
                        end_week_idx = st.selectbox("End Week", 
                                                  range(len(week_options)), 
                                                  format_func=lambda x: week_options[x],
                                                  index=len(week_options)-1)
                    
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
            selected_status = st.sidebar.selectbox('NS Status', status_options)
            
            if selected_status != 'All':
                data = data[data['NS Status'] == selected_status]
        
        # Filter by Site Type
        if 'Site Type' in data.columns:
            site_type_options = ['All'] + sorted(data['Site Type'].dropna().unique().tolist())
            selected_site_type = st.sidebar.selectbox('Site Type', site_type_options)
            
            if selected_site_type != 'All':
                data = data[data['Site Type'] == selected_site_type]
        
        # Filter by Client Priority
        if 'Client Priority' in data.columns:
            priority_options = ['All'] + sorted(data['Client Priority'].dropna().unique().tolist())
            selected_priority = st.sidebar.selectbox('Client Priority', priority_options)
            
            if selected_priority != 'All':
                data = data[data['Client Priority'] == selected_priority]
        
        # Filter by KTL Project Name
        if 'KTL Project Name' in data.columns:
            # Get unique project names and sort them
            project_names = data['KTL Project Name'].dropna().unique().tolist()
            project_names = sorted([str(name) for name in project_names if str(name).strip()])
            
            project_options = ['All'] + project_names
            selected_project = st.sidebar.selectbox('KTL Project Name', project_options)
            
            if selected_project != 'All':
                data = data[data['KTL Project Name'] == selected_project]
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Project Status", "Timeline Analysis", "KPI Metrics", "Resources", "Data Integration", "Detailed Data"])
        
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
                            
                            # Create a dataframe for the resources
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
                        st.warning("No project name column identified in the dataset.")
            else:
                st.warning("No resource columns found in the dataset. Please check column names.")

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
                    else:
                        st.info("No structural status data available")
            
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

if __name__ == "__main__":
    main() 