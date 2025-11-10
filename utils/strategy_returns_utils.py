
import numpy as np
import pandas as pd
import joblib
import os

# tickers we will trade on
TICKERS_TO_TRADE = [
    "SPY",  # S&P 500 ETF
    "EFA",  # MSCI EAFE (Developed Markets ex-US) ETF
    # "EEM",  # MSCI Emerging Markets ETF
    # "TLT",  # 20+ Year Treasury Bond ETF
    # "GLD",  # Gold ETF
    "USO",  # Crude Oil ETF
    "QQQ",  # Nasdaq 100 ETF
    # "IWM"   # Russell 2000 ETF
]

# Compute the 3 trading stategies signals
def add_sma_crossover(df, short_window, long_window, signal_name):
    """
    Add a simple moving average (SMA) crossover trading signal. (Trend-following / Momentum strat type)

    Buys (1) when the short SMA crosses above the long SMA, 
    sells (-1) when it crosses below.

    Parameters
    ----------
    df : DataFrame with a 'close' column
    short_window, long_window : int
        Window sizes for short and long SMAs.
    signal_name : str
        Name for the output signal column.
    """
    df = df.copy()
    df['SMA_short'] = df['close'].rolling(short_window).mean()
    df['SMA_long'] = df['close'].rolling(long_window).mean()
    df[signal_name] = np.where(df['SMA_short'] > df['SMA_long'], 1, -1)
    return df.drop(columns=['SMA_short', 'SMA_long'])

def add_rsi(df, period, signal_name):
    """
    Add an RSI-based mean reversion trading signal. (Mean reversion strat type)

    Buys (1) when RSI < 30 (oversold), sells (-1) when RSI > 70 (overbought).

    Parameters
    ----------
    df : DataFrame with a 'close' column
    period : int
        Lookback period for RSI calculation.
    signal_name : str
        Name for the output signal column.
    """
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    df[signal_name] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0)) # 65 35 for large periods maybe
    return df.drop(columns=['RSI'])


def add_bollinger_bands(df, window, num_std, signal_name):
    """
    Add a Bollinger Bands trading signal. (Volatility breakout & Reversion strat type)

    Buys (1) when price closes below the lower band, sells (-1) when above the upper band.

    Parameters
    ----------
    df : DataFrame with a 'close' column
    window : int
        Rolling window size for the moving average.
    num_std : float
        Number of standard deviations for the bands.
    signal_name : str
        Name for the output signal column.
    """
    df = df.copy()
    sma = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    upper_band = sma + num_std * rolling_std
    lower_band = sma - num_std * rolling_std
    df[signal_name] = np.where(df['close'] < lower_band, 1, np.where(df['close'] > upper_band, -1, 0))
    return df

# ---- Apply all strategies ----
def add_all_strategies(df):
    """
    Signals are numeric:
        1 = long bias,
        -1 = short bias,
        0 = neutral (for RSI & Bollinger).

    No lookahead bias in signal generation: All rolling and EMA operations use only past data.
    """

    df = add_sma_crossover(df, short_window=5, long_window=10, signal_name="S1_1_signal")
    df = add_sma_crossover(df, short_window=10, long_window=20, signal_name="S1_2_signal")
    df = add_sma_crossover(df, short_window=14, long_window=28, signal_name="S1_3_signal")
    df = add_rsi(df, period=7, signal_name="S2_1_signal")
    df = add_rsi(df, period=14, signal_name="S2_2_signal")
    df = add_rsi(df, period=21, signal_name="S2_3_signal")
    df = add_bollinger_bands(df, window=10, num_std=1, signal_name="S3_1_signal")
    df = add_bollinger_bands(df, window=20, num_std=1, signal_name="S3_2_signal")
    df = add_bollinger_bands(df, window=40, num_std=1.5, signal_name="S3_3_signal")
    return df


def melt_strategy_signals(df):
    """
    Reshape wide strategy signal columns into a long-format DataFrame.

    Converts multiple strategy signal columns into a unified structure
    with columns ['timestamp', 'symbol', 'strategy', 'signal'].

    Parameters
    ----------
    df : DataFrame
        Must contain a 'symbol', 'close', and multiple *_signal columns.

    Returns
    -------
    DataFrame
        Long-format DataFrame indexed by 'timestamp' with strategy signals.
    """
    strategy_cols = ["S1_1_signal", "S1_2_signal", "S1_3_signal", "S2_1_signal", "S2_2_signal", "S2_3_signal", "S3_1_signal", "S3_2_signal", "S3_3_signal"]
    records = []
    
    for strat in strategy_cols:
        temp = df[['symbol', 'close', strat]].copy()
        temp = temp.rename(columns={strat: 'signal'})
        temp['strategy'] = strat
        
        # Remove neutral signals to reduce noise so the model can focus on performance conditional on entering a trade 
        temp = temp[temp['signal'] != 0]
        
        records.append(temp[['symbol', 'strategy', 'signal']])
    
    df_strat_returns = pd.concat(records).reset_index()
    return df_strat_returns.set_index('timestamp')


def shift_join_clusters_data(df_final_clusters, df_strat_returns):
    """
    Shift regime labels by one day and join them df_strats.

    Uses the previous day's cluster (regime) since the current day's regime
    isn’t known during trading. Joins on matching timestamps.

    Parameters
    ----------
    df_final_clusters : DataFrame
        Contains 'final_cluster' regime labels by date.
    df_strat_returns : DataFrame
        Contains strategy returns with matching timestamps.

    Returns
    -------
    DataFrame
        Joined data with 'prev_regime' column aligned to each trading day.
    """
    # shift clusters one day back since they are computed at the end of the day, so that info is not available at the start of the day
    shifted_regimes = df_final_clusters[["final_cluster"]].shift(1).dropna().astype(int) #  drop frist day since is nan

    # join with clusters (first months have no cluster so have to drop them)
    df_joined = df_strat_returns.join(shifted_regimes, how='inner')
    df_joined.rename(columns={"final_cluster": "prev_regime"}, inplace=True)

    return df_joined

def add_features(df):
    """
    Add derived technical and statistical features for modeling.

    Includes returns, volatility, momentum, moving averages, 
    price ratios, volume trends, and range-based indicators.

    Parameters
    ----------
    df : DataFrame
        Must include 'close', 'high', 'low', and 'volume' columns.

    Returns
    -------
    DataFrame
        Copy of the input with added feature columns and NaNs dropped.
    """
    df = df.copy()

    # 1. Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)

    # # 2. Volatility (rolling std of returns)
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_5d'] = df['log_return_1d'].rolling(window=5).std()
    df['vol_10d'] = df['log_return_1d'].rolling(window=10).std()
    df.drop(columns="log_return_1d",  inplace=True)

    # 3. Momentum (price relative to N-day ago)
    df['mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['mom_10d'] = df['close'] / df['close'].shift(10) - 1
    df['mom_20d'] = df['close'] / df['close'].shift(20) - 1

    # 4. Moving averages
    df['sma_5d'] = df['close'].rolling(window=5).mean()
    df['sma_10d'] = df['close'].rolling(window=10).mean()
    df['sma_20d'] = df['close'].rolling(window=20).mean()

    # 5. Price relative to moving averages
    df['price_div_sma5'] = df['close'] / df['close'].rolling(window=5).mean() - 1
    df['price_div_sma10'] = df['close'] / df['close'].rolling(window=10).mean() - 1
    df['price_div_sma20'] = df['close'] / df['close'].rolling(window=20).mean() - 1

    # 6. Volume features
    df['vol_rolling_5d'] = df['volume'].rolling(window=5).mean()
    df['vol_rolling_10d'] = df['volume'].rolling(window=10).mean()
    df['vol_rolling_20d'] = df['volume'].rolling(window=20).mean()

    # # 7. Volatility normalized by volume (volume volatility ratio)
    df['vol_vol_ratio_5d'] = df['vol_5d'] / (df['vol_rolling_5d'] + 1e-9)

    # # 8. Price range (High-Low) relative to close
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    # # 9. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_7'] = tr.rolling(window=7).mean()
    df['atr_14'] = tr.rolling(window=14).mean()

    # # 10. Log volume change
    df['log_vol_change_1d'] = np.log(df['volume'] + 1) - np.log(df['volume'].shift(1) + 1)

    # Drop rows with NaNs due to rolling calculations
    df = df.dropna()

    return df.drop(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])

def regime_feat_engineering(final_df):
    """
    Add regime-based features to a DataFrame.

    Includes:
    - Consecutive days in the current regime
    - The last different previous regime
    - Days since each regime last occurred

    Parameters
    ----------
    final_df : DataFrame
        Must contain a 'prev_regime' column of regime labels.
    """

    # 1) Num of consecutive days the current regime is active
    # We'll count forward, resetting when the regime changes
    consecutive_days = []
    count = 0
    prev = None
    for val in final_df['prev_regime']:
        if val == prev:
            count += 1
        else:
            count = 1  # start counting again
        consecutive_days.append(count)
        prev = val

    final_df['consec_days_current_regime'] = consecutive_days

    # 2) Last previous regime different from current
    last_diff_regime = []
    for i, curr_regime in enumerate(final_df['prev_regime']):
        found = np.nan # if no prev regime
        for j in range(i-1, -1, -1):
            if final_df['prev_regime'].iloc[j] != curr_regime:
                found = final_df['prev_regime'].iloc[j]
                break
        last_diff_regime.append(found)

    final_df['last_prev_regime_different'] = last_diff_regime

    # 3) Days since last occurrence for each regime
    for regime_type in range(5):
        col_name = f"days_since_regime_{regime_type}"
        mask = final_df['prev_regime'] == regime_type
        
        last_seen_idx = None
        days_since = []
        
        for i, val in enumerate(final_df['prev_regime']):
            if val == regime_type:
                last_seen_idx = i
                days_since.append(0)
            else:
                if last_seen_idx is None:
                    days_since.append(np.nan)  # NaN for never seen
                else:
                    days_since.append(i - last_seen_idx)
        
        final_df[col_name] = days_since

    final_df.dropna(inplace=True) # drop nan rows that generated with these columns
    final_df.sort_index(inplace=True, ascending=True)

    return final_df

def prepare_and_predict(predict_dates_df, models_dr, model_nn):
    """
    Prepare features and generate predictions using a trained neural network.

    Scales numeric features, encodes categorical ones, combines them, 
    and predicts target values.

    Parameters
    ----------
    predict_dates_df : DataFrame
        Input data containing numeric and categorical features.
    models_dr : str
        Directory path to the saved OneHotEncoder ('ohe.joblib') and scaler ('stscaler_nn.joblib').
    model_nn : keras.Model or similar
        Trained neural network model for prediction.

    Returns
    -------
    ndarray
        Array of predicted values.
    """
    # Define feature groups
    categ_feats = ['symbol', 'strategy', 'prev_regime', 'last_prev_regime_different']
    target_col = 'strat_return'

    # Identify numeric features (exclude target, categorical, and 'signal')
    exclude_cols = categ_feats + [target_col, 'signal']
    numeric_feats = [col for col in predict_dates_df.columns if col not in exclude_cols]

    # --- Numeric features ---
    scaler = joblib.load(os.path.join(models_dr, "stscaler_nn.joblib"))
    X_num_full = predict_dates_df[numeric_feats].values
    X_num_scaled = scaler.transform(X_num_full)

    # --- Categorical features ---
    ohe = joblib.load(os.path.join(models_dr, "ohe_nn.joblib"))
    X_cat_full = ohe.transform(predict_dates_df[categ_feats])

    # --- Combine all features ---
    X = [X_num_scaled, X_cat_full]

    # --- Predict ---
    y_pred = model_nn.predict(X).ravel()

    return y_pred

def prepare_and_predict_by_regime(predict_dates_df, models_dr, fine_tuned_nn_dict):
    """
    Prepare features and generate regime-specific predictions using their fine-tuned models.

    Each model in `fine_tuned_nn` is applied only to rows where `prev_regime` matches
    its regime number (0–4). Predictions are combined into a single array.

    Parameters
    ----------
    predict_dates_df : DataFrame
        Input data containing numeric and categorical features, including 'prev_regime'.
    models_dr : str
        Directory path to the saved OneHotEncoder ('ohe.joblib') and scaler ('stscaler_nn.joblib').
    fine_tuned_nn : dict
        Dictionary mapping regime IDs (e.g. 0–4) to trained Keras models.

    Returns
    -------
    np.ndarray
        Array of predictions aligned with the original DataFrame index.
    """
    categ_feats = ['symbol', 'strategy', 'prev_regime', 'last_prev_regime_different']
    target_col = 'strat_return'

    # Identify numeric features
    exclude_cols = categ_feats + [target_col, 'signal']
    numeric_feats = [col for col in predict_dates_df.columns if col not in exclude_cols]

    # Scale numeric features
    scaler = joblib.load(os.path.join(models_dr, "stscaler_nn.joblib"))
    X_num_full = predict_dates_df[numeric_feats].values
    X_num_scaled = scaler.transform(X_num_full)

    # Load one-hot encoder
    ohe = joblib.load(os.path.join(models_dr, "ohe_nn.joblib"))
    X_cat_full = ohe.transform(predict_dates_df[categ_feats])

    # Combine full feature set
    X_num_df = pd.DataFrame(X_num_scaled, index=predict_dates_df.index)
    X_cat_df = pd.DataFrame(X_cat_full, index=predict_dates_df.index)

    # Initialize prediction array
    y_pred = np.zeros(len(predict_dates_df))

    # Predict per regime
    for name, model in fine_tuned_nn_dict.items():
        regime = int(name[-1]) # "regime_i" -> i
        mask = predict_dates_df['prev_regime'] == regime
        if mask.any():
            X_reg = [X_num_df.loc[mask].values, X_cat_df.loc[mask].values]
            y_pred[mask] = model.predict(X_reg).ravel()

    return y_pred


