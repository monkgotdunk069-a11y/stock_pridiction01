# 📈 Stock Movement Prediction & Trading API

A professional-grade machine learning pipeline for predicting next-day directional stock movements (UP/DOWN) for Indian NSE stocks. This project includes data ingestion, feature engineering, automated model training with time-series validation, a production-ready Flask REST API, and an interactive benchmark dashboard.

---

## 🚀 Key Features

- **Robust ML Pipeline**: Automated training using `RandomForest`, `DecisionTree`, and `LogisticRegression`.
- **Time-Series Validation**: Implements proper walk-forward validation (training on past data, testing on future) to prevent data leakage.
- **RESTful API**: Production-ready Flask server with endpoints for single and batch predictions.
- **Automated Feature Engineering**: Computes technical indicators (MA5, MA10, Volatility, Returns, Lags) on the fly.
- **Live Trading Signals**: Generates **BUY / HOLD / SELL** signals based on model confidence and probability thresholds.
- **Interactive Visualizations**: 
  - `dashboard.html`: Comprehensive KPI and accuracy dashboard.
  - `api_docs.html`: Beautiful documentation for API integration.
- **Data Export**: Generates detailed Excel reports (`final.xlsx`) and performance comparisons.

---

## 🗂️ Project Structure

```text
stock_pridiction01/
├── api.py                 # Flask REST API server
├── train_model.py         # Automated model training & evaluation pipeline
├── models/                # Directory containing saved .pkl models & metadata
├── dashboard.html         # Interactive performance dashboard
├── api_docs.html          # API documentation & testing interface
├── visualize.py           # Premium visualization generator (Matplotlib/Seaborn)
├── visualizations/        # Generated charts and graphs
├── requirements.txt       # Project dependencies
├── mainn.ipynb            # Analysis & experimentation notebook
├── final.xlsx             # Latest prediction results
└── model_comparison.xlsx  # Comparative model metrics
```

---

## 🧠 Model Performance

| Model                | Test Accuracy | Precision | Recall | F1-Score |
|----------------------|:-------------:|:---------:|:------:|:--------:|
| **Random Forest**    | **80.0 %**    | 0.82      | 0.78   | 0.80     |
| Logistic Regression  | 75.6 %        | 0.74      | 0.77   | 0.75     |
| Decision Tree        | 74.4 %        | 0.73      | 0.75   | 0.74     |

*Performance based on a 3-year history (2022–2025) across TCS, INFY, RELIANCE, and HDFCBANK.*

---

## 🌐 API Endpoints

The Flask API provides a simple interface for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Serves the interactive `api_docs.html` |
| `/api/health` | `GET` | Check server and model status |
| `/api/models` | `GET` | List available models and their metrics |
| `/api/stocks` | `GET` | List supported tickers and exchanges |
| `/api/predict` | `POST` | Get prediction for a specific ticker |
| `/api/predict/batch` | `POST` | Batch predictions for multiple stocks |

---

## ⚙️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
Before running the API or Dashboard, you must train the models on current data:
```bash
python train_model.py
```
This will download data, engineer features, and save models to the `models/` folder.

### 3. Launch the API
```bash
python api.py
```
The server will start on `http://localhost:5000` (or the first available port).

### 4. Generate Visualizations
To create high-quality performance charts:
```bash
python visualize.py
```
Charts will be saved to the `visualizations/` directory.

### 5. View Dashboard & Docs
- Open `dashboard.html` in your browser to see performance metrics.
- Navigate to `http://localhost:5000` to view the API documentation.

---

## 📊 Technical Indicators (Features)

The model makes decisions based on:
- **Moving Averages**: 5-day and 10-day trends.
- **Price Momentum**: Daily percentage returns and lag features.
- **Volatility**: 5-day rolling standard deviation.
- **Volume Metrics**: Volume moving averages and rate of change.
- **Trend Strength**: Difference between short-term and long-term averages.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
