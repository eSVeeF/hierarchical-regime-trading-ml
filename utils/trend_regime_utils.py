
import numpy as np
import pandas as pd

from scipy.stats import linregress
from scipy.stats.mstats import winsorize

from collections import Counter

# For Alpaca API
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient

from alpaca.data.requests import (
    StockBarsRequest
)

from alpaca.data.enums import Adjustment


TICKERS_SUBSET = {
    'US_Equity': ['SPY', 'QQQ', 'IWM', 'DIA'],
    'Defensive_Sectors': ['XLP', 'XLV', 'XLU'],
    'Global_Equities': ['EEM', 'EFA', 'FXI', 'EWZ'],
    'Currencies': ['UUP', 'FXY', 'FXF', 'FXE', 'FXA', 'FXC']
}

ALL_TICKERS = sum(TICKERS_SUBSET.values(), [])

def load_trend_data(API_KEY, SECRET_KEY, earliest_date, last_date, all_tickers=ALL_TICKERS):
    """
    Loads historical daily stock data from Alpaca for specified date range and a fixed set of tickers.

    Parameters:
        API_KEY (str): Alpaca API key.
        SECRET_KEY (str): Alpaca secret key.
        earliest_date (str): Start date, datetime object
        last_date (str): End date, datetime object

    Returns:
        pd.DataFrame: Sorted DataFrame of stock bars with timestamp index.
    """
    # Alpaca Data Load
    stock_historical_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # alpaca has no older data than 2016
    req = StockBarsRequest(
        symbol_or_symbols = all_tickers, 
        timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Day), 
        start = earliest_date,  
        end = last_date,                  
        limit = None,    
        adjustment=Adjustment('all') # adjust for splits and dividends                                           
    )
    df_raw = stock_historical_data_client.get_stock_bars(req).df.reset_index().set_index('timestamp')
    df_raw = df_raw.sort_values(by=['symbol', 'timestamp']) # Ensure sorted for correct rolling calcs
    return df_raw


def process_trend_data(df_raw, tickers_subset=TICKERS_SUBSET):
    """
    Processes raw stock data to compute trend-based features and pivot them by symbol. 
    Uses a constant dict of tickers subsets

    Parameters:
        df_raw (pd.DataFrame): Raw stock data with 'symbol' and 'close' columns.

    Returns:
        pd.DataFrame: Pivoted DataFrame of trend features (returns, RSI) by symbol.
    """

    # Trend feature functions
    def compute_trend_features(df):
        df = df.sort_index()
        df['ret_5d'] = df['close'].pct_change(5)
        df['ret_20d'] = df['close'].pct_change(20)
        df['ma_200'] = df['close'].rolling(200).mean()
        df['rsi_14'] = compute_rsi(df['close'], 14)
        return df

    def compute_rsi(series, period):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(com=period - 1, min_periods=period).mean()
        ma_down = down.ewm(com=period - 1, min_periods=period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    # Apply feature extraction
    features_by_subset = {}
    for subset, tickers in tickers_subset.items():
        df_subset = df_raw[df_raw['symbol'].isin(tickers)].copy()
        df_feat = df_subset.groupby('symbol').apply(compute_trend_features)
        df_feat = df_feat.droplevel(0)  # remove groupby index level
        df_feat = df_feat.dropna()
        features_by_subset[subset] = df_feat

    # Combine all subsets
    df_all = pd.concat(features_by_subset.values(), axis=0)
    df_all = df_all.sort_index()

    # Pivot wide
    pivot_cols = ['ret_5d', 'ret_20d', 'rsi_14']
    df_pivot = df_all.pivot_table(index=df_all.index, columns='symbol', values=pivot_cols)
    df_pivot = df_pivot.dropna()

    return df_pivot


def create_advanced_feat(df_pivot):
    """
    Generates advanced macro and momentum features from pivoted trend data.

    Parameters:
        df_pivot (pd.DataFrame): Pivoted DataFrame of trend features by symbol.

    Returns:
        pd.DataFrame: DataFrame of smoothed and winsorized advanced features.
    """

    df_features = pd.DataFrame(index=df_pivot.index)

    # 1. Mean 5-day return across equity indices
    df_features['us_equity_returns_5d_avg'] = df_pivot['ret_5d'][['SPY', 'QQQ', 'IWM', 'DIA']].mean(axis=1)


    # 2. US Equity Indices momentum slope: slope of 10-day returns linear fit
    def slope_of_returns(series):
        x = np.arange(len(series))
        if len(series) < 10 or series.isnull().any():
            return np.nan
        slope, _, _, _, _ = linregress(x, series)
        return slope

    eq_mom_slope = pd.DataFrame(index=df_pivot.index)
    for c in ['SPY', 'QQQ', 'IWM', 'DIA']:
        eq_mom_slope[c] = df_pivot['ret_5d'][c].rolling(window=10).apply(slope_of_returns, raw=False)

    df_features['us_equity_momentum_slope'] = eq_mom_slope.mean(axis=1)

    # 3. Defensive sector RSI difference: mean RSI of defensive sectors minus mean RSI of US and global equity indices
    df_features['equity_defensive_vs_broad_rsi_14d_spread'] = df_pivot['rsi_14'][['XLP', 'XLV', 'XLU']].mean(axis=1) -\
                                    df_pivot['rsi_14'][['SPY', 'QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'FXI', 'EWZ']].mean(axis=1)

    # 4. Global mean RSI across global equities (smooth replacement)
    df_features['global_equity_rsi_14d_avg'] = df_pivot['rsi_14'][['EEM', 'EFA', 'FXI', 'EWZ']].mean(axis=1)

    # 5. Defensive safe-haven currencies against more cyclically-oriented "risk-on" currencies
    df_features['safe_vs_risk_currencies_returns_5d_spread'] = df_pivot['ret_20d'][['UUP', 'FXY', 'FXF']].mean(axis=1) - \
                                        df_pivot['ret_20d'][['FXE', 'FXA', 'FXC']].mean(axis=1)

    # Drop any rows with NaNs introduced by rolling window
    df_features = df_features.dropna()

    # clip 1% and 99% quantiles to reduce outliers
    for col in df_features.columns:
        df_features[col] = winsorize(df_features[col], limits=[0.01, 0.01])

    return df_features


def mayority_vote_cluster_smooth(df_with_clusters, window_smooth=5):
    """
    Applies rolling majority vote smoothing to cluster labels.

    Parameters:
        df_with_clusters (pd.DataFrame): DataFrame containing a 'cluster' column.
        window_smooth (int): Size of the rolling window for smoothing. 
            window=5 days (1 trading week) is best

    Returns:
        pd.Series: Smoothed cluster labels with NaNs filled forward and backward.
    """

    def rolling_mode(series):
        """Returns the most frequent value in the window."""
        counts = Counter(series)
        return counts.most_common(1)[0][0]

    def smooth_clusters(cluster_series, window):
        return cluster_series.rolling(window=window, center=True).apply(rolling_mode, raw=True)

    df_with_clusters['cluster_smooth'] = smooth_clusters(df_with_clusters['cluster'], window=window_smooth)
    # fill NaNs (first and last days are nans because of the center=true window)
    # fill with forward values in the first days, backward in last days. Best solution for post ML models
    df_with_clusters['cluster_smooth'] = df_with_clusters['cluster_smooth'].ffill().bfill().astype(int)

    return df_with_clusters['cluster_smooth']

