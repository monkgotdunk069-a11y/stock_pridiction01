"""
api.py — Flask REST API for Stock Prediction Models.

Endpoints:
    GET  /                          → API documentation page
    GET  /api/health                → Health check
    GET  /api/models                → List all available models with metrics
    POST /api/predict               → Predict for a single stock ticker
    POST /api/predict/batch         → Predict for multiple tickers at once
    GET  /api/stocks                → List supported stock tickers
    GET  /api/features              → List model features used

Usage:
    1. First run:  python train_model.py   (trains & saves models)
    2. Then run:   python api.py           (starts the API server)
"""

import os
import sys
import json
import socket
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ─── CONFIG ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL = "random_forest"

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

# ─── APP SETUP ───────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ─── LOAD MODELS ─────────────────────────────────────────
models = {}
metadata = {}


def load_models():
    """Load all saved models from disk."""
    global models, metadata

    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        print("⚠️  No trained models found. Run 'python train_model.py' first.")
        return False

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    for name in ['random_forest', 'decision_tree', 'logistic_regression']:
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"   ✓ Loaded {name} (test accuracy: {metadata[name]['test_accuracy']})")

    return len(models) > 0


# ─── HELPER FUNCTIONS ────────────────────────────────────

def fetch_stock_data(ticker: str, days: int = 60):
    """
    Fetch recent stock data for a given ticker.
    We need ~60 days to compute rolling features reliably.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                     end=end_date.strftime("%Y-%m-%d"), progress=False)

    if df.empty:
        return None

    df.reset_index(inplace=True)

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    return df


def compute_features(df):
    """Compute technical indicator features from OHLCV data."""
    df = df.copy()

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

    df.dropna(inplace=True)

    return df


def generate_signal(confidence: float) -> str:
    """Generate BUY/SELL/HOLD signal from prediction confidence."""
    if confidence > 0.60:
        return "BUY"
    elif confidence < 0.40:
        return "SELL"
    else:
        return "HOLD"


def confidence_quality(confidence: float) -> str:
    """Rate the confidence level."""
    c = max(confidence, 1 - confidence)  # distance from 0.5
    if c > 0.75:
        return "STRONG"
    elif c > 0.60:
        return "MODERATE"
    else:
        return "WEAK"


def predict_stock(ticker: str, model_name: str = DEFAULT_MODEL):
    """
    Full prediction pipeline for a single stock:
    1. Fetch data
    2. Compute features
    3. Run model
    4. Return prediction + metadata
    """
    if model_name not in models:
        return None, f"Model '{model_name}' not found. Available: {list(models.keys())}"

    pipeline = models[model_name]

    # Fetch data
    df = fetch_stock_data(ticker)
    if df is None or len(df) < 15:
        return None, f"Unable to fetch sufficient data for ticker '{ticker}'. " \
                      f"Make sure it's a valid Yahoo Finance ticker (e.g. TCS.NS, AAPL)."

    # Compute features
    df = compute_features(df)
    if df.empty:
        return None, f"Not enough data to compute features for '{ticker}'."

    # Get the latest row for prediction
    latest = df.iloc[[-1]][FEATURES]

    # Predict
    prediction = int(pipeline.predict(latest)[0])
    up_probability = float(pipeline.predict_proba(latest)[0][1])

    # Directional confidence = how confident the model is in its prediction
    # If prediction=UP, confidence = up_probability
    # If prediction=DOWN, confidence = 1 - up_probability (= down probability)
    directional_confidence = up_probability if prediction == 1 else (1 - up_probability)

    # Build response
    latest_row = df.iloc[-1]
    data_date = latest_row['Date']
    if hasattr(data_date, 'date'):
        data_date = data_date.date()
    else:
        data_date = pd.to_datetime(str(data_date)).date()

    # Calculate next trading day (skip weekends)
    tomorrow = data_date + timedelta(days=1)
    while tomorrow.weekday() >= 5:  # 5=Sat, 6=Sun
        tomorrow += timedelta(days=1)

    signal = generate_signal(up_probability)
    quality = confidence_quality(up_probability)

    model_meta = metadata.get(model_name, {})
    result = {
        "ticker": ticker,
        "model": model_name,
        "prediction": prediction,
        "prediction_label": "UP" if prediction == 1 else "DOWN",
        "confidence": round(directional_confidence, 4),
        "up_probability": round(up_probability, 4),
        "confidence_quality": quality,
        "signal": signal,
        "should_trade": quality != "WEAK",
        "current_price": round(float(latest_row['Close']), 2),
        "data_date": str(data_date),
        "prediction_for_date": str(tomorrow),
        "features_used": {feat: round(float(latest_row[feat]), 4) for feat in FEATURES},
        "model_accuracy": model_meta.get('test_accuracy', 'N/A'),
        "model_precision": model_meta.get('precision', 'N/A'),
        "model_recall": model_meta.get('recall', 'N/A'),
        "model_f1": model_meta.get('f1_score', 'N/A'),
    }

    return result, None


# ─── API ROUTES ──────────────────────────────────────────

@app.route("/")
def index():
    """Serve the API documentation page."""
    docs_path = os.path.join(os.path.dirname(__file__), "api_docs.html")
    if os.path.exists(docs_path):
        return send_file(docs_path)
    return jsonify({
        "message": "Stock Prediction API",
        "docs": "Visit /api/health for status"
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all available models with full metrics."""
    model_info = []
    for name in models:
        m = metadata.get(name, {})
        info = {
            "name": name,
            "display_name": name.replace("_", " ").title(),
            "train_accuracy": m.get('train_accuracy', 'N/A'),
            "test_accuracy": m.get('test_accuracy', 'N/A'),
            "precision": m.get('precision', 'N/A'),
            "recall": m.get('recall', 'N/A'),
            "f1_score": m.get('f1_score', 'N/A'),
            "backtest": m.get('backtest', {}),
            "is_default": name == DEFAULT_MODEL
        }
        model_info.append(info)

    return jsonify({
        "models": model_info,
        "default_model": DEFAULT_MODEL,
        "split_method": metadata.get('split_method', 'unknown'),
        "total": len(model_info)
    })


@app.route("/api/stocks", methods=["GET"])
def list_stocks():
    """List the supported stock tickers the model was trained on."""
    stocks = metadata.get('stocks', [])
    stock_info = []
    for s in stocks:
        clean_name = s.replace(".NS", "")
        stock_info.append({
            "ticker": s,
            "name": clean_name,
            "exchange": "NSE" if ".NS" in s else "BSE" if ".BO" in s else "Unknown"
        })

    return jsonify({
        "trained_stocks": stock_info,
        "note": "The model can predict any valid Yahoo Finance ticker, "
                "but was trained on these Indian stocks.",
        "total": len(stock_info)
    })


@app.route("/api/features", methods=["GET"])
def list_features():
    """List the features used by the models."""
    feature_descriptions = {
        'MA_5': '5-day Moving Average of Close price',
        'MA_10': '10-day Moving Average of Close price',
        'Lag_1': 'Previous day Close price',
        'Lag_2': 'Close price 2 days ago',
        'Volatility': '5-day rolling standard deviation of Close',
        'Return': 'Daily percentage return',
        'Volume_MA': '5-day Moving Average of Volume',
        'MA_diff': 'Difference between MA_5 and MA_10 (trend indicator)',
        'Price_Change': 'Day-over-day price change in absolute terms',
        'Volume_Change': 'Day-over-day percentage change in volume'
    }

    features = []
    for feat in FEATURES:
        features.append({
            "name": feat,
            "description": feature_descriptions.get(feat, "")
        })

    return jsonify({
        "features": features,
        "total": len(features),
        "target": "Binary classification — 1 (UP) if next day close > today close, else 0 (DOWN)"
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict stock movement for a single ticker.

    Request Body (JSON):
    {
        "ticker": "TCS.NS",
        "model": "random_forest"  // optional, defaults to random_forest
    }

    Returns prediction, confidence, signal, and feature values.
    """
    data = request.get_json()
    if not data or 'ticker' not in data:
        return jsonify({
            "error": "Missing 'ticker' in request body.",
            "example": {"ticker": "TCS.NS", "model": "random_forest"}
        }), 400

    ticker = data['ticker'].strip().upper()
    model_name = data.get('model', DEFAULT_MODEL).strip().lower()

    result, error = predict_stock(ticker, model_name)

    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        "success": True,
        "data": result
    })


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict stock movement for multiple tickers.

    Request Body (JSON):
    {
        "tickers": ["TCS.NS", "INFY.NS", "RELIANCE.NS"],
        "model": "random_forest"  // optional
    }
    """
    data = request.get_json()
    if not data or 'tickers' not in data:
        return jsonify({
            "error": "Missing 'tickers' array in request body.",
            "example": {
                "tickers": ["TCS.NS", "INFY.NS"],
                "model": "random_forest"
            }
        }), 400

    tickers = data['tickers']
    if not isinstance(tickers, list) or len(tickers) == 0:
        return jsonify({"error": "'tickers' must be a non-empty array."}), 400

    if len(tickers) > 10:
        return jsonify({"error": "Maximum 10 tickers per batch request."}), 400

    model_name = data.get('model', DEFAULT_MODEL).strip().lower()

    results = []
    errors = []

    for ticker in tickers:
        ticker = ticker.strip().upper()
        result, error = predict_stock(ticker, model_name)
        if error:
            errors.append({"ticker": ticker, "error": error})
        else:
            results.append(result)

    return jsonify({
        "success": True,
        "data": results,
        "errors": errors,
        "total_predictions": len(results),
        "total_errors": len(errors)
    })


# ─── ERROR HANDLERS ──────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found.",
        "available_endpoints": [
            "GET  /",
            "GET  /api/health",
            "GET  /api/models",
            "GET  /api/stocks",
            "GET  /api/features",
            "POST /api/predict",
            "POST /api/predict/batch"
        ]
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal server error.",
        "details": str(e)
    }), 500


# ─── MAIN ────────────────────────────────────────────────

def find_free_port(start=5000, end=5010):
    """Find a free port, trying start through end."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return start


if __name__ == "__main__":
    print("=" * 60)
    print("  📈 STOCK PREDICTION API")
    print("=" * 60)

    if load_models():
        port = find_free_port()
        print(f"\n   ✅ {len(models)} models loaded successfully!")
        print(f"\n   🌐 Starting server on http://localhost:{port}")
        print(f"   📄 API Docs:  http://localhost:{port}")
        print(f"   💊 Health:    http://localhost:{port}/api/health")
        print("=" * 60)
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("\n   ❌ No models found!")
        print("   Run 'python train_model.py' first to train and save models.")
        print("=" * 60)
