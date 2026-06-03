# BTC---Signal-Generator
# 📈 ML Signal Generator — BTC/USDT

A machine learning pipeline for predicting BTC/USDT price direction using technical indicators.

## Overview

This project builds a complete ML workflow to generate buy/sell signals on cryptocurrency markets. It retrieves historical candlestick data from the Binance API, engineers a rich set of technical features, trains two classification models, and outputs the best model ready for live inference.

**Target variable**: Will the next 1-minute candle close higher (Up) or lower (Down)?

---

## Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                      ML SIGNAL GENERATOR                         ║
╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────┐
  │  📥 DATA INGESTION                                          │
  │                                                             │
  │   Binance API  ──▶  Cache Check (30 min)  ──▶  Parquet     │
  └──────────────────────────────────┬──────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  🔧 FEATURE ENGINEERING                                     │
  │                                                             │
  │   OHLCV Parsing  ──▶  Outlier Removal  ──▶  20 Features    │
  │                                                             │
  │   RSI, MACD, ATR, Bollinger Bands, MA10/50, Returns        │
  └──────────────────────────────────┬──────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  🤖 MODEL TRAINING  (80/20 temporal split)                  │
  │                                                             │
  │   ┌──────────────────┐     ┌──────────────────┐            │
  │   │ Logistic         │     │ Random Forest    │            │
  │   │ Regression       │     │ (50 estimators)  │            │
  │   └────────┬─────────┘     └────────┬─────────┘            │
  │            └──────────┬─────────────┘                      │
  └───────────────────────┼─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  📊 EVALUATION & EXPORT                                     │
  │                                                             │
  │   Walk-Forward Validation  ──▶  Model Selection  ──▶  .pkl │
  └─────────────────────────────────────────────────────────────┘
```

---

## Features (20 total)

| Category         | Features                                              |
| ---------------- | ----------------------------------------------------- |
| **Price Action** | PrevClose, High-Low, Open-Close                       |
| **Trend**        | MA10, MA50, MACD, MACD Signal, MACD Histogram         |
| **Momentum**     | RSI (14), Returns (1, 5, 15 periods)                  |
| **Volatility**   | ATR (14), Bollinger Bands (Width, %B, Low, Mid, High) |
| **Volume**       | Raw Volume, Log Volume                                |

---

## Results

**Out-of-sample performance** (~9,500 test samples):

| Model               | Accuracy | F1 Score (Up) |
| ------------------- | -------- | ------------- |
| Logistic Regression | 52.5%    | **0.570**     |
| Random Forest       | 51.0%    | 0.538         |

**Walk-Forward Validation** (6 folds, expanding window):

| Fold | F1 Score (Up) |
| ---- | ------------- |
| 1    | 0.646         |
| 2    | 0.568         |
| 3    | 0.565         |
| 4    | 0.556         |
| 5    | 0.544         |
| 6    | 0.583         |
| **Mean** | **0.577 ± 0.033** |

> Logistic Regression was selected as the best model.

---

## Installation

```bash
git clone https://github.com/chrisayegh/signal-generator.git
cd signal-generator
pip install -r requirements.txt
```

### Dependencies

```
pandas
numpy
scikit-learn
pandas-ta
python-binance
matplotlib
joblib
```

---

## Configuration

```bash
export BINANCE_API_KEY='your_api_key'
export BINANCE_API_SECRET='your_api_secret'
```

> **Note**: Public endpoints work without authentication for historical data.

---

## Usage

### Run the pipeline

```bash
jupyter notebook Signaux2.ipynb
```

### Parameters

| Parameter          | Default   | Description                     |
| ------------------ | --------- | ------------------------------- |
| `symbol`           | `BTCUSDT` | Trading pair                    |
| `interval`         | `1m`      | Candlestick interval            |
| `lookback_candles` | `50000`   | Historical data size (~35 days) |
| `max_age_minutes`  | `30`      | Cache TTL                       |

### Load the trained model for inference

```python
from joblib import load

model = load('best_model.pkl')
prediction = model.predict(features)
probabilities = model.predict_proba(features)
```

---

## Project Structure

```
signal-generator/
├── Signaux2.ipynb       # Full pipeline notebook
├── best_model.pkl       # Saved best model
├── requirements.txt
├── data_cache/          # Local Parquet cache
└── README.md
```

---

## Technical Details

- **Temporal validation**: 80/20 split without shuffling to prevent look-ahead bias
- **Walk-forward validation**: 6-fold TimeSeriesSplit to simulate live trading conditions
- **Class balancing**: Balanced weights to handle class imbalance
- **Outlier detection**: IQR-based filtering on close prices
- **Feature scaling**: StandardScaler applied inside sklearn Pipeline

---

## Limitations

- Modest accuracy (~52%) typical for short-term price prediction
- Transaction costs and slippage not modeled
- Single asset and timeframe tested

---

## 👤 Author

**Christopher Sayegh**
- GitHub: [@chrisayegh](https://github.com/chrisayegh)
