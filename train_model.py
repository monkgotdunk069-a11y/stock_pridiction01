"""
train_model.py — Train & Export stock prediction models for the API.
Uses proper TIME-SERIES SPLIT (train on past, test on future).
Saves precision, recall, F1, and accuracy metrics.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ─── CONFIG ──────────────────────────────────────────────
STOCKS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
START_DATE = "2022-01-01"
END_DATE = "2025-01-01"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURES = [
    'MA_5', 'MA_10',
    'Lag_1', 'Lag_2',
    'Volatility',
    'Return',
    'Volume_MA',
    'MA_diff',
    'Price_Change',
    'Volume_Change'
]


def download_data():
    print("📥 Downloading stock data...")
    df_list = []
    for stock in STOCKS:
        temp_df = yf.download(stock, start=START_DATE, end=END_DATE)
        temp_df.reset_index(inplace=True)
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = [col[0] for col in temp_df.columns]
        temp_df['Stock'] = stock
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    print(f"   ✓ Downloaded {len(df)} rows for {len(STOCKS)} stocks")
    return df


def clean_and_engineer(df):
    print("🔧 Cleaning & engineering features...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock']
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]

    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ].astype('float32')

    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    # Feature Engineering
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Volatility'] = df['Close'].rolling(5).std()
    df['Return'] = df['Close'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(5).mean()
    df['MA_diff'] = df['MA_5'] - df['MA_10']
    df['Price_Change'] = df['Close'] - df['Lag_1']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    print(f"   ✓ Shape after engineering: {df.shape}")
    return df


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1."""
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def backtest(pipeline, df, features, split_ratio=0.8):
    """
    Simple walk-forward backtest.
    Simulates trading: BUY when confidence > 0.6, measure win rate & returns.
    """
    split_idx = int(len(df) * split_ratio)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[features]
    y_test = test_df['Target'].values
    prices = test_df['Close'].values

    predictions = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]

    trades = []
    for i in range(len(predictions) - 1):
        confidence = probas[i]
        if confidence > 0.6:  # Only trade when confident
            action = "BUY"
            entry = prices[i]
            exit_price = prices[i + 1]
            pnl = ((exit_price - entry) / entry) * 100
            correct = int(predictions[i] == y_test[i])
            trades.append({
                'pnl_pct': round(float(pnl), 4),
                'correct': correct,
                'confidence': round(float(confidence), 4)
            })
        elif confidence < 0.4:
            action = "SELL"
            entry = prices[i]
            exit_price = prices[i + 1]
            pnl = ((entry - exit_price) / entry) * 100  # Short
            correct = int(predictions[i] == y_test[i])
            trades.append({
                'pnl_pct': round(float(pnl), 4),
                'correct': correct,
                'confidence': round(float(1 - confidence), 4)
            })

    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0, 'total_return': 0}

    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    total_return = sum(t['pnl_pct'] for t in trades)
    avg_return = total_return / len(trades)

    return {
        'total_trades': len(trades),
        'win_rate': round(wins / len(trades), 4),
        'avg_return_pct': round(avg_return, 4),
        'total_return_pct': round(total_return, 4),
        'best_trade_pct': round(max(t['pnl_pct'] for t in trades), 4),
        'worst_trade_pct': round(min(t['pnl_pct'] for t in trades), 4),
    }


def train_models(df):
    print("🤖 Training models with TIME-SERIES SPLIT...")

    # ⚠️ TIME-SERIES SPLIT — train on past, test on future
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURES]
    y_train = train_df['Target']
    X_test = test_df[FEATURES]
    y_test = test_df['Target']

    print(f"   📅 Train: {train_df['Date'].min().date()} → {train_df['Date'].max().date()} ({len(train_df)} rows)")
    print(f"   📅 Test:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()} ({len(test_df)} rows)")

    results = {}

    # ── 1. Random Forest ──
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_metrics = compute_metrics(y_test, rf_pred)
    rf_backtest = backtest(rf_pipeline, df, FEATURES)
    print(f"   ✓ Random Forest  — Acc: {rf_metrics['accuracy']}, F1: {rf_metrics['f1_score']}, Win Rate: {rf_backtest['win_rate']}")
    results['random_forest'] = {
        'pipeline': rf_pipeline, **rf_metrics,
        'train_accuracy': round(rf_pipeline.score(X_train, y_train), 4),
        'test_accuracy': rf_metrics['accuracy'],
        'backtest': rf_backtest
    }

    # ── 2. Decision Tree ──
    dt_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10))
    ])
    dt_pipeline.fit(X_train, y_train)
    dt_pred = dt_pipeline.predict(X_test)
    dt_metrics = compute_metrics(y_test, dt_pred)
    dt_backtest = backtest(dt_pipeline, df, FEATURES)
    print(f"   ✓ Decision Tree  — Acc: {dt_metrics['accuracy']}, F1: {dt_metrics['f1_score']}, Win Rate: {dt_backtest['win_rate']}")
    results['decision_tree'] = {
        'pipeline': dt_pipeline, **dt_metrics,
        'train_accuracy': round(dt_pipeline.score(X_train, y_train), 4),
        'test_accuracy': dt_metrics['accuracy'],
        'backtest': dt_backtest
    }

    # ── 3. Logistic Regression ──
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_metrics = compute_metrics(y_test, lr_pred)
    lr_backtest = backtest(lr_pipeline, df, FEATURES)
    print(f"   ✓ Logistic Reg.  — Acc: {lr_metrics['accuracy']}, F1: {lr_metrics['f1_score']}, Win Rate: {lr_backtest['win_rate']}")
    results['logistic_regression'] = {
        'pipeline': lr_pipeline, **lr_metrics,
        'train_accuracy': round(lr_pipeline.score(X_train, y_train), 4),
        'test_accuracy': lr_metrics['accuracy'],
        'backtest': lr_backtest
    }

    return results


def save_models(results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    metadata = {}

    for name, data in results.items():
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(data['pipeline'], model_path)
        meta = {k: v for k, v in data.items() if k != 'pipeline'}
        metadata[name] = meta
        print(f"   💾 Saved {model_path}")

    metadata['features'] = FEATURES
    metadata['stocks'] = STOCKS
    metadata['split_method'] = 'time_series'

    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   💾 Saved {meta_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  STOCK PREDICTION — MODEL TRAINING PIPELINE")
    print("  ⚡ Using TIME-SERIES SPLIT (no data leakage)")
    print("=" * 60)

    df = download_data()
    df = clean_and_engineer(df)
    results = train_models(df)
    save_models(results)

    print("\n" + "=" * 60)
    print("  ✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModels saved to: {MODEL_DIR}")
    print("Run the API with:  python api.py")
