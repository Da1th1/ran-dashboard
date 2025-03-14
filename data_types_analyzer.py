import pandas as pd
import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Data Types Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Cornerstone A&D Design Master Tracker - Data Types Analyzer")
st.markdown("This tool analyzes the data types and statistics for each column in the dataset.")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Cornerstone - A&D - Design Master Tracker.csv", low_memory=False)
        
        # Explicitly define date columns
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
        additional_date_columns = [col for col in df.columns 
                                  if any(word in col.lower() for word in 
                                       ['date', 'issued', 'approved', 'instruction']) 
                                  and col not in date_columns]
        
        # Combine all date columns
        all_date_columns = date_columns + additional_date_columns
        
        # Convert date columns with UK format (day first)
        for col in all_date_columns:
            if col in df.columns:
                try:
                    # First, replace common placeholder values with NaN
                    df[col] = df[col].astype(str).replace(['1/1/1900', '01/01/1900', '1900-01-01', 
                                                          'nan', 'NaT', 'NaN', '', 'None'], pd.NA)
                    
                    # Convert to datetime with day first format (UK date format)
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    
                    # Make sure 1/1/1900 dates are set to NaT if pandas parsed them
                    jan_1_1900 = pd.Timestamp('1900-01-01')
                    df.loc[df[col] == jan_1_1900, col] = pd.NaT
                except Exception as e:
                    st.warning(f"Error converting column {col}: {e}")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main function
def main():
    # Load data
    data = load_data()
    
    if data is not None:
        st.success(f"Data loaded successfully! Found {len(data)} rows and {len(data.columns)} columns.")
        
        # Create a dataframe showing data types for each column
        st.subheader("Column Data Types and Statistics")
        
        # Option to show original or inferred data types
        infer_types = st.checkbox("Infer data types (automatically detect numbers, dates, etc.)", value=True)
        
        if infer_types:
            # Try to convert columns to appropriate types
            for col in data.columns:
                # Skip columns already converted to datetime in load_data
                if pd.api.types.is_datetime64_dtype(data[col]):
                    continue
                    
                # Try numeric conversion
                try:
                    data[col] = pd.to_numeric(data[col])
                    continue
                except:
                    pass
                
                # Try datetime conversion for any remaining date-like columns
                try:
                    if any(word in col.lower() for word in ['date', 'issued', 'approved', 'instruction']):
                        # First, replace common placeholder values with NaN
                        data[col] = data[col].astype(str).replace(['1/1/1900', '01/01/1900', '1900-01-01', 
                                                                  'nan', 'NaT', 'NaN', '', 'None'], pd.NA)
                        
                        # Convert to datetime
                        data[col] = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
                        
                        # Make sure 1/1/1900 dates are set to NaT if pandas parsed them
                        jan_1_1900 = pd.Timestamp('1900-01-01')
                        data.loc[data[col] == jan_1_1900, col] = pd.NaT
                except:
                    pass
        
        # Generate column info
        column_info = []
        for column in data.columns:
            # Get data type
            dtype = data[column].dtype
            
            # Calculate number of non-null values
            non_null_count = data[column].count()
            null_count = len(data) - non_null_count
            
            # Calculate percentage of non-null values
            null_percentage = (null_count / len(data)) * 100
            
            # Count unique values
            unique_count = data[column].nunique()
            
            # Get a sample value (first non-null)
            sample = data[column].dropna().iloc[0] if non_null_count > 0 else None
            
            # Format sample value for display
            if sample is not None:
                if pd.api.types.is_datetime64_dtype(data[column]):
                    sample_str = sample.strftime('%d/%m/%Y')
                else:
                    sample_str = str(sample)
                sample_display = sample_str[:50] + '...' if len(sample_str) > 50 else sample_str
            else:
                sample_display = "None"
            
            # Calculate value frequency for categorical columns
            top_value_str = "None"
            top_count = 0
            frequency = "No data"
            
            if unique_count <= 30 and unique_count > 0 and non_null_count > 0:
                try:
                    top_value = data[column].value_counts().index[0]
                    top_count = data[column].value_counts().iloc[0]
                    
                    # Format top value for display
                    if pd.api.types.is_datetime64_dtype(data[column]):
                        top_value_str = top_value.strftime('%d/%m/%Y')
                    else:
                        top_value_str = str(top_value)
                    
                    frequency = f"{top_value_str} ({top_count} times, {(top_count/len(data))*100:.1f}%)"
                except:
                    frequency = "Error calculating top value"
            elif unique_count > 30:
                frequency = "Too many unique values"
            
            # Add to our column info list
            column_info.append({
                'Column Name': column,
                'Data Type': str(dtype),
                'Non-Null Count': int(non_null_count),
                'Null Count': int(null_count),
                'Null %': f"{null_percentage:.1f}%",
                'Unique Values': int(unique_count) if pd.notna(unique_count) else 0,
                'Top Value': frequency,
                'Sample Value': sample_display
            })
        
        # Create the dataframe
        column_df = pd.DataFrame(column_info)
        
        # Add search functionality
        search_term = st.text_input("Search for specific columns:")
        if search_term:
            filtered_df = column_df[column_df['Column Name'].str.contains(search_term, case=False)]
            st.dataframe(filtered_df)
        else:
            st.dataframe(column_df)
        
        # Option to see all values for a specific column
        st.subheader("Examine Specific Column")
        selected_column = st.selectbox("Select a column to examine in detail:", data.columns)
        
        if selected_column:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write("Basic Statistics:")
                
                # Check if numeric
                if pd.api.types.is_numeric_dtype(data[selected_column]):
                    st.write(data[selected_column].describe())
                elif pd.api.types.is_datetime64_dtype(data[selected_column]):
                    # For datetime columns
                    non_null_dates = data[selected_column].dropna()
                    if not non_null_dates.empty:
                        try:
                            min_date = non_null_dates.min()
                            max_date = non_null_dates.max()
                            range_days = (max_date - min_date).days
                            
                            date_stats = {
                                "count": len(non_null_dates),
                                "min": min_date.strftime('%d/%m/%Y'),
                                "max": max_date.strftime('%d/%m/%Y'),
                                "range (days)": range_days,
                                "null count": data[selected_column].isna().sum(),
                                "null %": f"{(data[selected_column].isna().sum() / len(data)) * 100:.1f}%"
                            }
                            st.write(pd.Series(date_stats))
                        except Exception as e:
                            st.error(f"Error calculating date statistics: {e}")
                            st.write("Basic count:", len(non_null_dates))
                    else:
                        st.write("No valid dates in this column")
                else:
                    # For non-numeric columns
                    try:
                        top_value = data[selected_column].value_counts().index[0] if data[selected_column].count() > 0 else "None"
                        freq = data[selected_column].value_counts().iloc[0] if data[selected_column].count() > 0 else 0
                        
                        stats = {
                            "count": len(data[selected_column]),
                            "unique": data[selected_column].nunique(),
                            "top": top_value,
                            "freq": freq,
                            "null count": data[selected_column].isna().sum(),
                            "null %": f"{(data[selected_column].isna().sum() / len(data)) * 100:.1f}%"
                        }
                        st.write(pd.Series(stats))
                    except Exception as e:
                        st.error(f"Error calculating statistics: {e}")
            
            with col2:
                # Show value counts if not too many unique values
                if data[selected_column].nunique() <= 50:
                    st.write("Value Counts:")
                    
                    try:
                        # Handle datetime columns specially
                        if pd.api.types.is_datetime64_dtype(data[selected_column]):
                            # Create a temporary Series with formatted dates as strings
                            formatted_dates = data[selected_column].dropna().dt.strftime('%d/%m/%Y')
                            value_counts = formatted_dates.value_counts().reset_index()
                            value_counts.columns = [selected_column, 'Count']
                        else:
                            value_counts = data[selected_column].value_counts().reset_index()
                            value_counts.columns = [selected_column, 'Count']
                        
                        # Calculate percentages
                        value_counts['Percentage'] = (value_counts['Count'] / len(data)) * 100
                        value_counts['Percentage'] = value_counts['Percentage'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(value_counts)
                    except Exception as e:
                        st.error(f"Error displaying value counts: {e}")
                else:
                    st.write(f"Too many unique values ({data[selected_column].nunique()}) to display value counts.")
                    st.write("Sample values:")
                    
                    try:
                        # Display sample values, formatted for dates
                        samples = data[selected_column].dropna().sample(min(10, data[selected_column].count()))
                        if pd.api.types.is_datetime64_dtype(data[selected_column]):
                            st.write(samples.dt.strftime('%d/%m/%Y'))
                        else:
                            st.write(samples)
                    except Exception as e:
                        st.error(f"Error displaying samples: {e}")
        
        # Option to download the data types dataframe
        csv = column_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data types analysis as CSV",
            data=csv,
            file_name="column_data_types.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main() 