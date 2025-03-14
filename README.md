# Cornerstone A&D Design Master Tracker Dashboard

A Streamlit dashboard for analyzing and visualizing data from the Cornerstone A&D Design Master Tracker CSV file.

## Features

- **Overview**: Displays key metrics and visualizations of the project status and site type distribution.
- **Project Status**: Detailed analysis of project statuses, blockers, and priorities.
- **Timeline Analysis**: Analysis of project timelines, including instruction dates and project durations.
- **Detailed Data**: Access to the raw data with search functionality and the ability to download filtered data.

## Installation

1. Ensure you have Python 3.7+ installed on your system.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your "Cornerstone - A&D - Design Master Tracker.csv" file in the same directory as the dashboard.py file.
2. Run the dashboard:

```bash
streamlit run dashboard.py
```

3. The dashboard will open in your default web browser at http://localhost:8501.

## Filters

The dashboard includes several filters in the sidebar:
- NS Status
- Site Type
- Client Priority

Use these filters to narrow down the data displayed in the dashboard.

## Notes

- The dashboard automatically attempts to detect and convert date columns to the proper format.
- For the Project Duration Analysis, you need to select appropriate start and end date columns.
- The search functionality in the Detailed Data tab allows you to search across all columns.

## Requirements

- streamlit==1.32.0
- pandas==2.1.4
- plotly==5.18.0
- numpy==1.26.4
- matplotlib==3.8.3 