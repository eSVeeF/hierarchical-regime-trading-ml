
import numpy as np
import pandas as pd

# For Alpaca API
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient

from alpaca.data.requests import (
    StockBarsRequest
)

from alpaca.data.enums import Adjustment


US_GLOBAL_EQUITIES = ['SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'VEA', 'VWO']

OTHER_BULL_TICKERS = ['TLT', 'GLD']

def load_bull_trend_data(API_KEY, SECRET_KEY, earliest_date, last_date, all_tickers=US_GLOBAL_EQUITIES+OTHER_BULL_TICKERS):
    """
    Loads historical daily stock data from Alpaca for specified date range and a fixed set of tickers.

    Parameters:
        API_KEY (str): Alpaca API key.
        SECRET_KEY (str): Alpaca secret key.
        earliest_date (str): Start date, datetime object
        last_date (str): End date, datetime object
        all_tickers (list): str of tickers to load

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


def create_advanced_bull_feat(df_bull_raw, us_global_equities=US_GLOBAL_EQUITIES, rolling_window=20):
    """
    Generate advanced bullish market features from equity and external asset data.

    This function computes trend and volume-based metrics for a set of U.S. and 
    global equities, along with correlations to selected external assets (TLT, GLD).
    It aggregates equity data, calculates rolling statistical features, 
    and returns a cleaned DataFrame of engineered features for further modeling.

    Parameters
    ----------
    df_bull_raw : pd.DataFrame
        Raw market data containing at least the columns: 'symbol', 'close', 'high', 
        'low', 'volume', indexed by date.
    us_global_equities : list of str, optional
        List of equity symbols to include when computing aggregated market features.
        Defaults to US_GLOBAL_EQUITIES.
    rolling_window : int
        Window size (in periods) for rolling calculations. Is inteded to be 20.

    Returns
    -------
    pd.DataFrame
    """
    df_eq = df_bull_raw[df_bull_raw['symbol'].isin(us_global_equities)].sort_index().copy()
    df_eq_mean = df_eq.groupby(df_eq.index).mean(numeric_only=True)

    # ---Trend Consistency Score ---
    df_eq_mean['trend_consistency'] = ((df_eq_mean['close'] - df_eq_mean['low']) / 
                                    (df_eq_mean['high'] - df_eq_mean['low'] + 1e-6)).rolling(rolling_window).mean()

    # --- Volatility Asymmetry Index ---
    returns = df_eq_mean['close'].pct_change()

    def upside_vol_func(returns_window):
        pos_returns = returns_window[returns_window > 0]
        if len(pos_returns) >= 2:  # needs at least 2 points for std
            return pos_returns.std()
        else:
            return np.nan

    upside_vol = returns.rolling(rolling_window).apply(upside_vol_func, raw=False)
    total_vol = returns.rolling(rolling_window).std()
    df_eq_mean['vol_asymmetry'] = upside_vol / (total_vol + 1e-6)

    # --- Feature Higher High Consistency ---
    higher_high = (df_eq_mean['high'] > df_eq_mean['high'].shift(1)).astype(int)
    df_eq_mean['higher_high_consistency'] = higher_high.rolling(rolling_window).mean()

    # --- Helper function to extract close price of a symbol ---
    def get_close(symbol):
        return df_bull_raw[df_bull_raw['symbol'] == symbol].sort_index()['close']

    # Get closes for external tickers
    tlt = get_close('TLT')
    gld = get_close('GLD')

    external_df = pd.DataFrame(index=df_eq_mean.index)

    # --- Correlations ---
    external_df['long_bond_correlation'] = df_eq_mean['close'].pct_change().rolling(rolling_window).corr(tlt.pct_change())
    external_df['gold_correlation'] = df_eq_mean['close'].pct_change().rolling(rolling_window).corr(gld.pct_change())

    # Combine features and external features
    bull_features_df = pd.concat([df_eq_mean[['trend_consistency', 'vol_asymmetry', 'higher_high_consistency']],
                            external_df], axis=1)

    # Drop rows with missing values
    bull_features_df.dropna(inplace=True)

    return bull_features_df


def merge_clean_final_clusters(bull_trend_spectral_labels, only_bull_features_df, df_with_clusters):
    """
    Merge bullish-trend cluster labels with existing previous smoothed cluster assignments.

    This function adjusts bullish spectral clustering labels (replacing label 0 with 4),
    previous clustering had clusters 0, 1, 2, 3, 1s being bulls, therefore we only need
    to replace the new 0s bull cluster with the 5th cluster 4s.
    aligns them with the corresponding feature DataFrame, and merges them with an
    existing DataFrame of smoothed clusters. The final cluster assignment uses the 
    bullish cluster where available and falls back to the smoothed cluster otherwise.

    Parameters
    ----------
    bull_trend_spectral_labels : array-like
        Cluster labels from bullish spectral clustering.
    only_bull_features_df : pd.DataFrame
        DataFrame of bullish market features with a datetime index matching the labels.
    df_with_clusters : pd.DataFrame
        DataFrame containing at least the column 'cluster_smooth' for non-bull clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column 'final_cluster', containing the merged cluster IDs.
    """
    # since 0 cluster was previously another cluster replace 0s with 5th clusters, 4s
    bull_trend_spectral_labels = np.where(bull_trend_spectral_labels == 0, 4, bull_trend_spectral_labels)

    df_with_bull_clusters = pd.DataFrame(bull_trend_spectral_labels, columns=["cluster_bull"], index=only_bull_features_df.index)

    # compute final clusters
    df_final_clusters = df_with_clusters.join(df_with_bull_clusters, how="outer")
    # use new bull division, and the previous clusers non-bull
    df_final_clusters['final_cluster'] = df_final_clusters['cluster_bull'].fillna(df_final_clusters['cluster_smooth']).astype(int)
    # keep only them
    df_final_clusters = df_final_clusters[['final_cluster']]

    return df_final_clusters