import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yfinance as yf
from datetime import datetime, timedelta

# ─── CONFIG & STYLE ──────────────────────────────────────
MODELS_DIR = "models"
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set premium visualization style
plt.style.use('dark_background')
sns.set_palette("viridis")
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#1e1e1e'
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['font.family'] = 'sans-serif'

def plot_model_comparison(metadata):
    """Generates a bar chart comparing train vs test accuracy."""
    print("📊 Plotting model comparison...")
    model_names = [m for m in metadata if isinstance(metadata[m], dict) and 'test_accuracy' in metadata[m]]
    train_acc = [metadata[m]['train_accuracy'] * 100 for m in model_names]
    test_acc = [metadata[m]['test_accuracy'] * 100 for m in model_names]
    
    labels = [m.replace("_", " ").title() for m in model_names]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#00cfc1', alpha=0.8)
    rects2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#6366f1', alpha=0.9)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, facecolor='#2d2d2d')
    ax.set_ylim(0, 105)

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_performance.png'), dpi=300)
    plt.close()

def plot_feature_importance():
    """Generates a chart showing which indicators matter most."""
    print("📉 Plotting feature importance...")
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    
    if not os.path.exists(rf_path) or not os.path.exists(meta_path):
        print("   ⚠️ Models or metadata missing. Skipping feature importance.")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    model = joblib.load(rf_path)
    features = meta.get('features', [])
    
    # Get importance from Random Forest
    if hasattr(model, 'steps'): # Pipeline
        importance = model.steps[-1][1].feature_importances_
    else:
        importance = model.feature_importances_
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma')
    plt.title('Technical Indicator Importance (Random Forest)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()

def plot_correlation_heatmap():
    """Generates a heatmap of feature correlations."""
    print("🌡️ Plotting correlation heatmap...")
    # Fetch sample data to compute correlation
    ticker = "TCS.NS"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    
    df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), 
                     end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    if df.empty: return
    
    # Simple feature engineering
    data = pd.DataFrame()
    data['MA_5'] = df['Close'].rolling(5).mean()
    data['MA_10'] = df['Close'].rolling(10).mean()
    data['Volatility'] = df['Close'].rolling(5).std()
    data['Return'] = df['Close'].pct_change()
    data['Volume_MA'] = df['Volume'].rolling(5).mean()
    data['Price_Change'] = df['Close'].diff()
    data.dropna(inplace=True)

    plt.figure(figsize=(10, 8))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Feature Correlation Heatmap ({ticker})', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation.png'), dpi=300)
    plt.close()

def main():
    print("🚀 Starting visualization generation...")
    
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        print("❌ Error: metadata.json not found in models/ directory.")
        print("   Run 'python train_model.py' first.")
        return

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    plot_model_comparison(metadata)
    plot_feature_importance()
    plot_correlation_heatmap()
    
    print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")
    print("   1. model_performance.png")
    print("   2. feature_importance.png")
    print("   3. feature_correlation.png")

if __name__ == "__main__":
    main()
