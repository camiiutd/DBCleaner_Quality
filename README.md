# LINK: https://dbcleanerquality.streamlit.app/
# DATA QUALITY ANALYZER
A web-based tool that automates the cleaning and preprocessing of your .csv database.

This tool helps you:
- Identify missing values, duplicates, and outliers
- Remove common problematic values (UNKNOWN, N/A, --, etc.)
- Remove user-specified custom values
- Automatically convert columns to numeric when appropriate
- Download both the cleaned dataset and the quality report
- Preview the cleaned dataset ready for BI tools like Tableau

# Data Quality Report
Includes for each column:
- Data type  
- Null count / percentage  
- Number of unique values  
- Outlier count  
- Quality score

Report also includes:
- Total removed rows  
- Total removed duplicates  

# Export Options
- `clean_data.csv` – cleaned dataset  
- `quality_report.csv` – quality analysis report
