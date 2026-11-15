"""
KisanMandi AI - Machine Learning & Multi-Horizon Forecasting Engine
Trains LightGBM, Random Forest, and Ridge Regressors on real Agmarknet data.
Provides probabilistic uncertainty bounds (P10, P50, P90) and rubric-ready metrics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import os

FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
    'roll_mean_7', 'roll_std_7', 'roll_min_7', 'roll_max_7',
    'roll_mean_14', 'roll_std_14', 'roll_min_14', 'roll_max_14',
    'roll_mean_30', 'roll_std_30', 'roll_min_30', 'roll_max_30',
    'price_spread', 'spread_ratio', 'momentum_7d', 'momentum_30d',
    'ratio_ma7', 'ratio_ma30',
    'day_of_week', 'day_of_month', 'month', 'quarter', 'sin_doy', 'cos_doy'
]

def calculate_metrics(y_true, y_pred, y_base=None):
    """
    Computes RMSE, MAE, MAPE, R2, and Mean Directional Accuracy (MDA).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Avoid zero division in MAPE
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Mean Directional Accuracy (MDA)
    if y_base is not None and len(y_base) == len(y_true):
        actual_dir = np.sign(y_true - y_base)
        pred_dir = np.sign(y_pred - y_base)
        mda = np.mean(actual_dir == pred_dir) * 100
    else:
        mda = 0.0
        
    return {
        'RMSE': round(float(rmse), 2),
        'MAE': round(float(mae), 2),
        'MAPE (%)': round(float(mape), 2),
        'R2 Score': round(float(r2), 4),
        'Directional Accuracy (%)': round(float(mda), 2)
    }

class MandiForecaster:
    """
    Multi-horizon forecaster supporting point estimation and quantile uncertainty.
    """
    def __init__(self, horizon='15d'):
        self.horizon = horizon
        self.target_col = f'target_{horizon}'
        self.models = {}
        self.feature_names = []
        self.evaluation_results = {}
        self.scaler = None
        
    def prepare_data(self, df):
        """
        Filters valid feature and target rows, splits chronologically (train/test).
        """
        data = df.copy()
        # Drop rows where target or 30-day lags are NaN
        data = data.dropna(subset=[self.target_col] + ['lag_30', 'roll_mean_30'])
        
        # Available feature columns
        available_feats = [c for c in FEATURE_COLS if c in data.columns]
        self.feature_names = available_feats
        
        X = data[available_feats].copy()
        y = data[self.target_col].copy()
        y_base = data['Modal_Price'].copy()
        dates = data['Date'].copy()
        
        # Out-of-time chronological train-test split (80% train, 20% test)
        split_idx = int(len(data) * 0.8)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        y_base_test = y_base.iloc[split_idx:]
        test_dates = dates.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, y_base_test, test_dates, available_feats

    def train_and_benchmark(self, df):
        """
        Trains LightGBM, Random Forest, and Ridge baseline.
        """
        X_train, X_test, y_train, y_test, y_base_test, test_dates, feats = self.prepare_data(df)
        
        if len(X_train) < 20:
            raise ValueError(f"Insufficient historical data ({len(X_train)} rows) for robust training.")
            
        # 1. Baseline: Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)
        metrics_ridge = calculate_metrics(y_test.values, pred_ridge, y_base_test.values)
        
        # 2. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        metrics_rf = calculate_metrics(y_test.values, pred_rf, y_base_test.values)
        
        # 3. Champion: LightGBM Point Forecaster (Mean / L2 Loss)
        lgb_point = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=20,
            random_state=42,
            verbose=-1
        )
        lgb_point.fit(X_train, y_train)
        pred_lgb = lgb_point.predict(X_test)
        metrics_lgb = calculate_metrics(y_test.values, pred_lgb, y_base_test.values)
        
        # 4. LightGBM Quantile Forecasters (P10 and P90 for Uncertainty Bands)
        lgb_p10 = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.10,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=16,
            random_state=42,
            verbose=-1
        )
        lgb_p10.fit(X_train, y_train)
        pred_p10 = lgb_p10.predict(X_test)
        
        lgb_p90 = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.90,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=16,
            random_state=42,
            verbose=-1
        )
        lgb_p90.fit(X_train, y_train)
        pred_p90 = lgb_p90.predict(X_test)
        
        # Store models
        self.models = {
            'ridge': ridge,
            'rf': rf,
            'lgb_point': lgb_point,
            'lgb_p10': lgb_p10,
            'lgb_p90': lgb_p90
        }
        
        self.evaluation_results = {
            'Ridge (Baseline)': metrics_ridge,
            'Random Forest': metrics_rf,
            'LightGBM (Champion)': metrics_lgb
        }
        
        # Feature importances from LightGBM
        importances = pd.DataFrame({
            'Feature': feats,
            'Importance': lgb_point.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        test_predictions = pd.DataFrame({
            'Date': test_dates,
            'Actual': y_test.values,
            'Base_Price': y_base_test.values,
            'Pred_LGBM': pred_lgb,
            'Pred_P10': pred_p10,
            'Pred_P90': pred_p90,
            'Pred_RF': pred_rf,
            'Pred_Ridge': pred_ridge
        })
        
        return self.evaluation_results, importances, test_predictions

    def predict_future(self, latest_features_df, days_ahead=15):
        """
        Predicts future price with P10, P50, and P90 uncertainty estimates.
        """
        if 'lgb_point' not in self.models:
            raise RuntimeError("Model has not been trained yet.")
            
        feats = self.feature_names
        X_latest = latest_features_df[feats].tail(1)
        
        p50 = float(self.models['lgb_point'].predict(X_latest)[0])
        p10 = float(self.models['lgb_p10'].predict(X_latest)[0])
        p90 = float(self.models['lgb_p90'].predict(X_latest)[0])
        
        # Ensure logical ordering
        p10 = min(p10, p50)
        p90 = max(p90, p50)
        
        return {
            'horizon': self.horizon,
            'p10_bearish': round(p10, 2),
            'p50_modal_forecast': round(p50, 2),
            'p90_bullish': round(p90, 2)
        }

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

if __name__ == '__main__':
    from data_pipeline import load_and_clean_data, get_daily_series, engineer_features
    print("Loading data...")
    raw = load_and_clean_data()
    series = get_daily_series(raw, state='Tamil Nadu', district='Thanjavur', commodity='Onion')
    feat_df = engineer_features(series)
    
    print("Training 15-day forecaster on Thanjavur Onion...")
    forecaster = MandiForecaster(horizon='15d')
    metrics, importances, preds = forecaster.train_and_benchmark(feat_df)
    
    print("\n--- MODEL COMPARISON BENCHMARK ---")
    print(pd.DataFrame(metrics).T)
    print("\n--- TOP 5 FEATURE IMPORTANCES ---")
    print(importances.head(5))
    
    model_path = os.path.join('models', 'thanjavur_onion_15d.joblib')
    forecaster.save(model_path)
    print(f"\nModel saved successfully to {model_path}")
