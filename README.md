# 📈 Stock Prediction Model

A machine-learning pipeline that predicts the next-day directional movement (UP / DOWN) of four major Indian NSE stocks using technical indicators and news sentiment analysis, with an interactive HTML benchmark dashboard.

---

## 🚀 Features

- **Multi-stock data ingestion** — downloads 3 years of OHLCV data (2022–2025) for TCS, Infosys, Reliance, and HDFC Bank via `yfinance`
- **Feature engineering** — moving averages (MA5, MA10), lag features, volatility, daily returns, and volume moving average
- **Sentiment analysis** — real-time news headlines fetched from NewsAPI and scored with TextBlob
- **Machine-learning pipeline** — `StandardScaler` + `RandomForestClassifier` wrapped in a scikit-learn `Pipeline`
- **Model benchmarking** — compares Random Forest, Decision Tree, and Logistic Regression side-by-side
- **Trading signals** — generates BUY / HOLD / SELL signals based on predicted probability thresholds
- **Excel export** — saves prediction results (`final.xlsx`) and model comparison (`model_comparison.xlsx`)
- **Interactive dashboard** — `dashboard.html` visualises KPIs, accuracy charts, and signal distributions

---

## 🗂️ Project Structure

```
stock_pridiction01/
├── mainn.ipynb            # Main Jupyter notebook (data → features → model → results)
├── mainn.pyi              # Type-hint stub for the core imports
├── dashboard.html         # Interactive benchmark dashboard (open in browser)
├── final.xlsx             # Prediction results with BUY/HOLD/SELL signals
├── model_comparison.xlsx  # Accuracy comparison across models
└── README.md
```

---

## 🧠 Model Performance

| Model               | Test Accuracy |
|---------------------|:-------------:|
| **Random Forest**   | **80.0 %**    |
| Logistic Regression | 75.6 %        |
| Decision Tree       | 74.4 %        |

Signal distribution on the test set (590 samples):

| Signal | Count |
|--------|------:|
| BUY    |   224 |
| HOLD   |   184 |
| SELL   |   182 |

---

## 🛠️ Tech Stack

| Layer               | Libraries / Tools                             |
|---------------------|-----------------------------------------------|
| Data collection     | `yfinance`, `newsapi-python`                  |
| NLP / Sentiment     | `TextBlob`                                    |
| Feature engineering | `pandas`, `numpy`                             |
| Machine learning    | `scikit-learn` (Pipeline, RandomForest, etc.) |
| Visualisation       | HTML + Chart.js dashboard                     |
| Export              | `openpyxl`                                    |

---

## ⚙️ Getting Started

### Prerequisites

```bash
pip install pandas numpy yfinance scikit-learn textblob newsapi-python openpyxl
```

### Run the notebook

```bash
jupyter notebook mainn.ipynb
```

Run all cells in order. The notebook will:
1. Download stock data from Yahoo Finance
2. Engineer features and fetch sentiment scores
3. Train and evaluate the Random Forest pipeline
4. Export results to `final.xlsx` and `model_comparison.xlsx`

### View the dashboard

Open `dashboard.html` directly in any modern browser — no server required.

---

## 📊 Stocks Covered

| Ticker         | Company                   |
|----------------|---------------------------|
| `TCS.NS`       | Tata Consultancy Services |
| `INFY.NS`      | Infosys                   |
| `RELIANCE.NS`  | Reliance Industries       |
| `HDFCBANK.NS`  | HDFC Bank                 |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
