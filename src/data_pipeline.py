"""
KisanMandi AI - Data Pipeline
Reads the primary Agriculture_price_dataset (parquet, csv.gz, or csv),
standardizes units, cleans data, and engineers features.
"""

import pandas as pd
import numpy as np
import os

def find_dataset_path():
    """
    Intelligently searches for the dataset in Streamlit Cloud (/mount/src/trial),
    current dir, project root, or data/ folder in parquet, csv.gz, or csv format.
    """
    possible_names = [
        'Agriculture_price_dataset.parquet',
        'Agriculture_price_dataset.csv.gz',
        'Agriculture_price_dataset.csv',
        'mandi_prices.csv'
    ]
    possible_dirs = [
        '/mount/src/trial',
        '/mount/src/trial/data',
        os.getcwd(),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
        os.path.join(os.getcwd(), 'data')
    ]
    
    for d in possible_dirs:
        for name in possible_names:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
                
    return 'Agriculture_price_dataset.parquet'

def load_and_clean_data(filepath=None):
    """
    Loads raw Parquet / CSV.GZ / CSV and standardizes units and formats.
    """
    if filepath is None:
        filepath = find_dataset_path()
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure Agriculture_price_dataset.parquet is in your repository.")
        
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Parse dates
    df['Price Date'] = pd.to_datetime(df['Price Date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Price Date']).sort_values('Price Date').reset_index(drop=True)
    
    # Standardize state naming
    df['STATE_CLEAN'] = df['STATE'].astype(str).str.strip().replace({'Tamilnadu': 'Tamil Nadu'})
    df['District_Clean'] = df['District Name'].astype(str).str.strip().str.title()
    df['Market_Clean'] = df['Market Name'].astype(str).str.strip()
    df['Commodity_Clean'] = df['Commodity'].astype(str).str.strip().str.title()
    
    # Standardize numeric prices
    for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Domain Rule: Uzhavar Sandhai / retail entry correction
    # If price < 150 INR for wholesale staples, it was reported in Rs/Kg rather than Rs/Quintal (100 kg)
    # Standardize to Rs/Quintal (national Agmarknet standard)
    mask_kg = (df['Modal_Price'] > 0) & (df['Modal_Price'] < 150)
    df.loc[mask_kg, 'Min_Price'] = df.loc[mask_kg, 'Min_Price'] * 100
    df.loc[mask_kg, 'Max_Price'] = df.loc[mask_kg, 'Max_Price'] * 100
    df.loc[mask_kg, 'Modal_Price'] = df.loc[mask_kg, 'Modal_Price'] * 100
    
    return df

def get_daily_series(df, state='Tamil Nadu', district=None, commodity='Onion'):
    """
    Filters and aggregates prices into a continuous daily time-series.
    """
    sub = df[(df['STATE_CLEAN'].str.lower() == state.lower()) & 
             (df['Commodity_Clean'].str.lower() == commodity.lower())]
    
    if district and district.lower() != 'all':
        sub = sub[sub['District_Clean'].str.lower() == district.lower()]
    
    if len(sub) == 0:
        return pd.DataFrame()
        
    daily = sub.groupby('Price Date').agg(
        Modal_Price=('Modal_Price', 'median'),
        Mean_Price=('Modal_Price', 'mean'),
        Min_Price=('Min_Price', 'min'),
        Max_Price=('Max_Price', 'max'),
        Market_Count=('Market_Clean', 'nunique'),
        Record_Count=('Modal_Price', 'count')
    ).reset_index()
    
    daily = daily.set_index('Price Date')
    full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
    daily = daily.reindex(full_idx)
    daily.index.name = 'Date'
    
    daily['Modal_Price'] = daily['Modal_Price'].interpolate(method='linear').bfill().ffill()
    daily['Min_Price'] = daily['Min_Price'].interpolate(method='linear').bfill().ffill()
    daily['Max_Price'] = daily['Max_Price'].interpolate(method='linear').bfill().ffill()
    daily['Market_Count'] = daily['Market_Count'].fillna(0)
    daily['Record_Count'] = daily['Record_Count'].fillna(0)
    
    daily = daily.reset_index()
    return daily

def engineer_features(daily_df):
    """
    Generates temporal lag, rolling statistics, volatility, and seasonality features.
    """
    df = daily_df.copy()
    
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f'lag_{lag}'] = df['Modal_Price'].shift(lag)
        
    for window in [7, 14, 30]:
        df[f'roll_mean_{window}'] = df['Modal_Price'].shift(1).rolling(window=window).mean()
        df[f'roll_std_{window}'] = df['Modal_Price'].shift(1).rolling(window=window).std().fillna(0)
        df[f'roll_min_{window}'] = df['Modal_Price'].shift(1).rolling(window=window).min()
        df[f'roll_max_{window}'] = df['Modal_Price'].shift(1).rolling(window=window).max()
        
    df['price_spread'] = df['Max_Price'] - df['Min_Price']
    df['spread_ratio'] = (df['price_spread'] / df['Modal_Price'].clip(lower=1)).fillna(0)
    df['momentum_7d'] = (df['Modal_Price'].shift(1) - df['lag_7']) / df['lag_7'].clip(lower=1)
    df['momentum_30d'] = (df['Modal_Price'].shift(1) - df['lag_30']) / df['lag_30'].clip(lower=1)
    df['ratio_ma7'] = df['Modal_Price'].shift(1) / df['roll_mean_7'].clip(lower=1)
    df['ratio_ma30'] = df['Modal_Price'].shift(1) / df['roll_mean_30'].clip(lower=1)
    
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    day_of_year = df['Date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    df['target_1d'] = df['Modal_Price'].shift(-1)
    df['target_7d'] = df['Modal_Price'].shift(-7)
    df['target_15d'] = df['Modal_Price'].shift(-15)
    df['target_30d'] = df['Modal_Price'].shift(-30)
    
    return df
