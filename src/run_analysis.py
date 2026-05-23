import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def download_data(ticker, period, out_csv):
    print(f"Downloading {ticker} for period={period}...")
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded — check network or ticker symbol")
    df.to_csv(out_csv)
    return df


def feature_engineer(df):
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    # lag features
    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    # moving averages and volatility
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['vol_5'] = df['return'].rolling(5).std()
    # target: next-day return
    df['target'] = df['return'].shift(-1)
    df = df.dropna()
    return df


def train_and_evaluate(df, outputs):
    features = [c for c in df.columns if c.startswith('return_lag_') or c in ('ma_5','ma_10','vol_5')]
    X = df[features].values
    y = df['target'].values

    # time-based split: last 20% as test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"Test RMSE: {rmse:.6f}, R2: {r2:.4f}")

    # save model
    joblib.dump(model, os.path.join(outputs, 'model.joblib'))

    # plots
    dates = df.index[split_idx:]
    plt.figure(figsize=(10,4))
    plt.plot(dates, df['Close'].values[split_idx:], label='Close')
    plt.title('Price (test period)')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs, 'price_test.png'))
    plt.close()

    # predicted vs actual
    plt.figure(figsize=(8,5))
    plt.plot(dates, y_test, label='Actual', alpha=0.8)
    plt.plot(dates, preds, label='Predicted', alpha=0.8)
    plt.legend()
    plt.title('Predicted vs Actual Next-day Return (test)')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs, 'pred_vs_actual.png'))
    plt.close()

    # feature importance
    importances = model.feature_importances_
    fi = pd.Series(importances, index=features).sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs, 'feature_importances.png'))
    plt.close()

    # correlation heatmap
    plt.figure(figsize=(6,6))
    sns.heatmap(df[features + ['target']].corr(), annot=True, fmt='.2f', cmap='vlag')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs, 'correlation.png'))
    plt.close()

    # return summary
    return {'rmse': rmse, 'r2': r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--period', default='5y')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--outputs-dir', default='outputs')
    args = parser.parse_args()

    ensure_dirs([args.data_dir, args.outputs_dir])

    csv_path = os.path.join(args.data_dir, f"{args.ticker}.csv")
    if os.path.exists(csv_path):
        print(f"Loading local CSV {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        df = download_data(args.ticker, args.period, csv_path)

    df_fe = feature_engineer(df)
    summary = train_and_evaluate(df_fe, args.outputs_dir)

    summary_path = os.path.join(args.outputs_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"RMSE: {summary['rmse']:.6f}\nR2: {summary['r2']:.6f}\n")

    print(f"Analysis complete. Outputs in {args.outputs_dir}")


if __name__ == '__main__':
    main()
