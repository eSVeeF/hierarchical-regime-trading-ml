#!/usr/bin/env python3
"""
Main execution script for predicting profitable trading strategy signals
using market regime detection and neural models.

Author: Sergio Vizcaino Ferrer
GitHub: https://github.com/eSVeeF/hierarchical-regime-trading-ml.git
"""

import os
import sys
import argparse
import logging
import warnings
# --- Environment settings & warnings ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

import joblib
import tensorflow as tf
import pandas as pd

# --- TensorFlow settings ---
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# --- Local imports ---
from utils.trend_regime_utils import (
    load_trend_data, process_trend_data, create_advanced_feat, mayority_vote_cluster_smooth
)
from utils.bull_trend_regime_utils import (
    load_bull_trend_data, create_advanced_bull_feat, merge_clean_final_clusters
)
from utils.strategy_returns_utils import (
    add_all_strategies, melt_strategy_signals, shift_join_clusters_data,
    add_features, regime_feat_engineering, prepare_and_predict_by_regime, TICKERS_TO_TRADE
)

# --- Constants ---
TZ = ZoneInfo("America/New_York")
PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
BEST_DECISION_THRESHOLD = 0.56
ALPACA_MIN_DATE = datetime(2016, 1, 16, tzinfo=TZ)
LAST_TRAINING_DATE = datetime(2025, 7, 20, tzinfo=TZ)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def setup_logging() -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_api_keys() -> tuple[str, str]:
    """Load API keys from environment variables."""
    load_dotenv(override=True)
    api_key = os.environ.get("ALP_API_KEY")
    sec_key = os.environ.get("ALP_SEC_KEY")
    if not api_key or not sec_key:
        logging.error("API keys not found. Please set ALP_API_KEY and ALP_SEC_KEY.")
        sys.exit(1)
    return api_key, sec_key


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict profitable trading strategy signals using pre-trained regime models."
    )
    parser.add_argument(
        "--start", required=True, help="Earliest date to predict (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", required=True, help="Last date to predict (YYYY-MM-DD)"
    )
    return parser.parse_args()


def validate_dates(start: str, end: str) -> tuple[datetime, datetime]:
    """Validate and return timezone-aware datetime objects."""
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=TZ)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=TZ)
        if end_dt < start_dt:
            raise ValueError("End date cannot be earlier than start date.")
        return start_dt, end_dt
    except ValueError as e:
        logging.error(f"Invalid date input: {e}")
        sys.exit(1)

def adjust_dates(start_date: datetime, end_date: datetime) -> tuple[datetime, datetime]:
    """
    Adjust user input dates to comply with Alpaca data availability
    and model training constraints.

    - Start date cannot be earlier than 2016-01-16 (Alpaca oldest data)
    - End date cannot be today/future
    - Start date for predictions cannot be earlier than the last training date (2025-07-21)
    """
    today = datetime.now(tz=TZ)
    yesterday = today - timedelta(days=1)

    if start_date < ALPACA_MIN_DATE:
        logging.warning(
            f"Start date {start_date:%Y-%m-%d} is earlier than Alpaca's oldest available data. "
            f"Adjusting to {ALPACA_MIN_DATE:%Y-%m-%d}."
        )
        start_date = ALPACA_MIN_DATE

    if end_date >= today:
        logging.warning(
            f"End date {end_date:%Y-%m-%d} is today or in the future. "
            f"Adjusting to {yesterday:%Y-%m-%d}."
        )
        end_date = yesterday

    if start_date <= LAST_TRAINING_DATE:
        adjusted_start = LAST_TRAINING_DATE + timedelta(days=1)
        logging.info(
            f"The models were trained using data through {LAST_TRAINING_DATE:%Y-%m-%d}. "
            f"Predictions can only start from {adjusted_start:%Y-%m-%d}. "
            f"Adjusting start date accordingly."
        )
        start_date = adjusted_start

    if start_date > end_date:
        logging.error(
            f"After adjustments, the start date {start_date:%Y-%m-%d} is after the end date {end_date:%Y-%m-%d}. Exiting."
        )
        sys.exit(1)

    return start_date, end_date


# -----------------------------------------------------------------------------
# Core Processing Functions
# -----------------------------------------------------------------------------
def detect_market_regimes(api_key: str, sec_key: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Detect primary and secondary market regimes."""
    df_trend_raw = load_trend_data(api_key, sec_key, datetime(2016, 1, 16, tzinfo=TZ), end_date)
    df_trend_processed = process_trend_data(df_trend_raw)
    df_trend_feat = create_advanced_feat(df_trend_processed)

    logging.info("Loading Regime Models, Computing Regimes...")
    scaler = joblib.load(os.path.join(MODEL_DIR, "trend_scaler.pkl"))
    umap_model = joblib.load(os.path.join(MODEL_DIR, "trend_umap_model.pkl"))
    gmm_model = joblib.load(os.path.join(MODEL_DIR, "trend_gmm_model.pkl"))

    trend_umap = umap_model.transform(scaler.transform(df_trend_feat))
    trend_labels = gmm_model.predict(trend_umap)
    df_clusters = pd.DataFrame(trend_labels, columns=["cluster"], index=df_trend_feat.index)
    df_cluster_smooth = mayority_vote_cluster_smooth(df_clusters)

    df_bull_raw = load_bull_trend_data(api_key, sec_key, datetime(2016, 1, 16, tzinfo=TZ), end_date)
    bull_features = create_advanced_bull_feat(df_bull_raw)

    bull_days = df_cluster_smooth[df_cluster_smooth == 1]
    only_bull = bull_features[bull_features.index.isin(bull_days.index)]

    bull_scaler = joblib.load(os.path.join(MODEL_DIR, "bull_trend_scaler.pkl"))
    bull_umap = joblib.load(os.path.join(MODEL_DIR, "bull_trend_umap_model.pkl"))
    spectral_model = joblib.load(os.path.join(MODEL_DIR, "bull_trend_spectral_model.pkl"))

    bull_umap_data = bull_umap.transform(bull_scaler.transform(only_bull))
    # SpectralClustering has no .predict(), so we refit it on the full dataset each time
    bull_labels = spectral_model.fit_predict(bull_umap_data)
    
    df_final = merge_clean_final_clusters(bull_labels, only_bull, df_clusters)
    return df_final


def generate_predictions(api_key: str, sec_key: str, start_date: datetime, end_date: datetime, df_final_clusters: pd.DataFrame) -> pd.DataFrame:
    """Generate profitable strategy predictions."""
    df_trade_raw = load_trend_data(api_key, sec_key, datetime(2016, 1, 16, tzinfo=TZ), end_date, all_tickers=TICKERS_TO_TRADE)
    logging.info("Preprocessing and Feature Engineering...")
    df_strats = add_all_strategies(df_trade_raw)
    df_melted = melt_strategy_signals(df_strats)
    df_joined = shift_join_clusters_data(df_final_clusters, df_melted)
    df_features = df_trade_raw.groupby("symbol", group_keys=False).apply(add_features)

    final_df = df_features.set_index("symbol", append=True).join(
        df_joined.set_index("symbol", append=True), how="inner"
    ).reset_index(level="symbol")

    final_df = regime_feat_engineering(final_df)
    predict_df = final_df.loc[start_date:end_date]

    logging.info("Loading Regime-Finetuned Models, Forecasting...")
    fine_tuned_models = {
        f"regime_{i}": tf.keras.models.load_model(os.path.join(MODEL_DIR, f"fine_tuned_regime_{i}_model.keras"))
        for i in range(5)
    }

    preds = prepare_and_predict_by_regime(predict_df, MODEL_DIR, fine_tuned_models)
    preds_bin = (preds >= BEST_DECISION_THRESHOLD).astype(int)
    predict_df["pred_prob"] = preds
    predict_df["pred_signal"] = preds_bin

    return predict_df[predict_df["pred_signal"] == 1].copy()


def save_results(results_df: pd.DataFrame, start: str, end: str, total_trades: int) -> None:
    """Save prediction results as a formatted text file inside /outputs."""
    STRATEGY_LEGEND = {
        "S1_1_signal": "SMA Crossover - 5/10d",
        "S1_2_signal": "SMA Crossover - 10/20d",
        "S1_3_signal": "SMA Crossover - 14/28d",
        "S2_1_signal": "RSI - 7d",
        "S2_2_signal": "RSI - 14d",
        "S2_3_signal": "RSI - 21d",
        "S3_1_signal": "Bollinger Bands - 10d - 1 Std",
        "S3_2_signal": "Bollinger Bands - 20d - 1 Std",
        "S3_3_signal": "Bollinger Bands - 40d - 1.5 Std",
    }

    results_df = results_df[["symbol", "strategy", "signal", "pred_prob"]].copy()
    results_df["signal"] = results_df["signal"].replace({1: "Long", -1: "Short"})
    results_df["strategy"] = results_df["strategy"].map(STRATEGY_LEGEND)
    results_df.reset_index(inplace=True)
    results_df.rename(columns={"index": "timestamp"}, inplace=True)
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"]).dt.strftime("%Y-%m-%d")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    output_filename = f"predicted_profitable_strategy_signals_{start}_{end}.txt"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    table_str = results_df.to_string(index=False, justify="center")

    with open(output_path, "w") as f:
        f.write(f"Total number of predicted profitable signals: {total_trades}\n\n")
        f.write(table_str)

    logging.info(f"Results saved to '{output_path}'")



# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    setup_logging()
    args = parse_args()
    api_key, sec_key = load_api_keys()
    start_dt, end_dt = validate_dates(args.start, args.end)
    start_dt, end_dt = adjust_dates(start_dt, end_dt)

    logging.info(f"Loading Trading Data from {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}")
    df_final_clusters = detect_market_regimes(api_key, sec_key, start_dt, end_dt)
    results_df = generate_predictions(api_key, sec_key, start_dt, end_dt, df_final_clusters)

    total_trades = len(results_df)
    logging.info(f"Predicted {total_trades} profitable trading signals.")
    save_results(results_df, args.start, args.end, total_trades)


if __name__ == "__main__":
    main()
