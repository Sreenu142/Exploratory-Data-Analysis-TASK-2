"""
inspect_crime_data.py
=====================
Statistical diagnostics and exploratory checks for the cleaned India Crime datasets
(cleaned_india_crime_data.csv and cleaned_crime_state_year.csv).
"""

import pandas as pd
import numpy as np

def inspect():
    print("=" * 80)
    print("1. INCIDENT-LEVEL CRIME DATASET (cleaned_india_crime_data.csv)")
    print("=" * 80)
    df_incident = pd.read_csv('cleaned_india_crime_data.csv')
    print(f"Dimensions: {df_incident.shape[0]:,} rows x {df_incident.shape[1]} columns")
    print(f"Missing Values: {df_incident.isnull().sum().sum()}")
    print("\nSummary Statistics:")
    print(df_incident[['cases_reported', 'persons_arrested', 'persons_chargesheeted', 'property_stolen_or_recovered_inr']].describe().T)

    print("\n" + "=" * 80)
    print("2. STATE-YEAR PANEL CRIME DATASET (cleaned_crime_state_year.csv)")
    print("=" * 80)
    df_state_year = pd.read_csv('cleaned_crime_state_year.csv')
    print(f"Dimensions: {df_state_year.shape[0]:,} rows x {df_state_year.shape[1]} columns")
    print(f"States/UTs Covered: {df_state_year['state_ut'].nunique()} | Years: {sorted(df_state_year['year'].unique())}")
    print("\nTop 5 States by Total Reported Crimes:")
    top_states = df_state_year.groupby('state_ut')['total_cases'].sum().sort_values(ascending=False).head(5)
    print(top_states)

if __name__ == '__main__':
    inspect()
