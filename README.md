# 🧠 Hierarchical ML for Market Regime Detection and Trading Strategy Performance Prediction

---

## 📘 Overview

This repository contains the implementation developed as part of my Bachelor Thesis, *“Hierarchical Machine Learning for Market Regime Detection and Trading Strategy Performance Prediction.”*

The project explores the performance of classical trading strategies such as SMA crossovers, RSI, and Bollinger Bands, combined with **market regime fine-tuning**.  
A model that performs well under one regime can fail catastrophically in another if it assumes the underlying structure is stable.  
To address this, the framework integrates **unsupervised regime detection** with **neural network models** that forecast the profitability of trading signals while being *regime-aware*.

> Rather than predicting raw price movements, this framework forecasts when specific trading strategies are likely to succeed or fail, providing a more interpretable, practical, and robust approach to market prediction.

---

## 🧩 Methodology Overview

The architecture follows a hierarchical, two-stage design:

1. **Market Regime Detection**
   - **Stage 1:** A UMAP + GMM clustering model detects broad market regimes (e.g., Bull, Bear, Neutral, U.S. Bull Only).
   - **Stage 2:** Within Bull regime, a UMAP + Spectral Clustering model separates *Aggressive Bull* and *Defensive Bull* subregimes.

2. **Strategy Profitability Prediction**
   - For each day and strategy signal, Long/Short, (e.g., SMA crossover or RSI trigger), features are computed from OHLCV data, technical indicators, and categorical context such as trading asset, and regime-derived variables.
   - A **global neural network** learns general profitability patterns across assets.
   - **Fine-tuned regime-specific networks** adapt the model to each detected regime.

The framework forecasts whether a given trading signal is likely to be profitable over a 10-day horizon.
---

## 🧠 Conceptual Pipeline

Alpaca API (Market Data) → Custom Asset Universe (SPY, QQQ, EEM, EFA, XLP, XLV, XLU, UUP, FXY, FXE, TLT, GLD, ...) → Feature Engineering (Returns, Ratios, Technical Indicators) → UMAP + GMM - Market Regime Clustering (Bull, Bear, Neutral, US Bull) → UMAP + Spectral - Bull Subregimes (Aggressive, Defensive) → Enhanced Dataset (Regime Tags + Technical + Strategy Trading Features and Target Variable) → Global NN + 5 Fine-Tuned Regime Models (TensorFlow) → Profitability Predictions - Output .txt Results Table

<img width="960" height="720" alt="General Flowchart" src="https://github.com/user-attachments/assets/74a4b9a6-147b-42c6-af5b-502424f8d7db" />

---

## 📊 Results Summary

The models were trained using Alpaca API data spanning **2016-01-16 to 2025-07-20**, evaluated on assets such as **SPY, QQQ, EFA,** and **USO**.

### Performance Across Regimes

| Metric | Random Baseline | Global Model | Fine-Tuned Models |
|:-------:|:----------------:|:-------------:|:-----------------:|
| Mean PR-AUC (All Regimes) | 0.444 | 0.488 | **0.505** |

<img width="1160" height="529" alt="pr_auc_neural_network" src="https://github.com/user-attachments/assets/845dcd36-b1fc-4115-a990-37f28042ef6f" />  **Grouped Barplot:** PR-AUC across five regimes (and pooled), comparing Random, Global, and Fine-Tuned models. 

Fine-tuned models outperform both the random baseline and the global model in all regimes except the Neutral one.

🏷️ Regime Labels

| Regime ID | Regime Name                 |
|:----------:|:----------------------------|
| 0          | Bear                        |
| 1          | Defensive / Consolidating Bull |
| 2          | Neutral                     |
| 3          | US Bull Only                |
| 4          | Strong / Aggressive Bull    |

### Return Optimization

When annualized ROI is plotted against the decision threshold:
- The **Global Model** achieves -0.7% ROI at a 0.5 threshold.
- **Fine-Tuned Models** peak at **+7.9% ROI** with a 0.56 threshold.

<img width="869" height="550" alt="ROI" src="https://github.com/user-attachments/assets/fe6e3cdc-d9f4-414b-ab04-0257a28f923f" /> **ROI Curve:** Annualized ROI vs decision threshold for global and fine-tuned models, showing stability regions and optimal thresholds.

Transaction cost assumption: *0.2% per trade.*

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/eSVeeF/hierarchical-regime-trading-ml.git
cd hierarchical-regime-trading-ml
pip install -r requirements.txt
````

---

## 🚀 Usage

The main execution script allows you to generate profitability predictions for trading strategy signals within a chosen date range.

```bash
python main.py --start YYYY-MM-DD --end YYYY-MM-DD
```

**Example:**

```bash
python main.py --start 2025-10-15 --end 2025-10-17
```

### 🧾 Example Output

After running, a `.txt` file like the following is generated in your working directory:

```
Total number of predicted profitable trading strategy signals: 2

timestamp   symbol   strategy                             signal          pred_prob
2025-10-15   USO     SMA Crossover - 10/20d windows        Long Position   0.594590
2025-10-17   SPY     RSI - 14d period                      Short Position  0.583820
```

---

## 📂 Repository Structure

```
📦 hierarchical-regime-trading-ml
├── main.py
├── .gitignore
├── utils/
│   ├── trend_regime_utils.py
│   ├── bull_trend_regime_utils.py
│   ├── strategy_returns_utils.py
│   └── __init__.py
├── models/
│   ├── trend_scaler.pkl
│   ├── trend_umap_model.pkl
│   ├── trend_gmm_model.pkl
│   ├── global_model.keras
│   ├── fine_tuned_regime_*_model.keras
│   └── ...
├── training_notebooks/
│   ├── 01_regime_detection.ipynb
│   ├── 02_bull_subregime_detection.ipynb
│   ├── 03_trading_strategy_returns_prediction.ipynb
├── outputs/                 
│   ├── predicted_profitable_strategy_signals_2025-08-03_2025-10-30.txt
│   ├── predicted_profitable_strategy_signals_2025-09-03_2025-09-05.txt
│   └── predicted_profitable_strategy_signals_2025-11-06_2025-11-09.txt
└── requirements.txt
```

---

## 🧰 Technologies Used

* **Python 3.11+**
* TensorFlow
* scikit-learn
* umap-learn
* joblib
* pandas
* numpy
* alpaca-py

---

## 🎓 Acknowledgment

This project was developed as part of my **Bachelor Thesis** at **Universidad Carlos III de Madrid (UC3M)** for the *Bachelor in Data Science and Engineering* program.

> *"A regime-aware framework that integrates neural networks with classical trading strategies, bridging the gap between academic research, industry practice, and real-world performance."*

---

## 📜 License

This repository is distributed under the **MIT License**.
Feel free to use or extend the code for research or educational purposes.

---

## ⭐ Citation

If you use this code or methodology in your work, please cite:

> Vizcaíno Ferrer, S. (2025). *Hierarchical ML for Market Regime Detection and Trading Strategy Performance Prediction*. Bachelor Thesis, Universidad Carlos III de Madrid.

---
