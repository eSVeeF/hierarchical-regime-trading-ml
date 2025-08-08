
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

OTHER_BULL_TICKERS = ['TLT', 'GLD', 'SPHB', 'SPLV']

def load_bull_trend_data(API_KEY, SECRET_KEY, earliest_date, last_date, all_tickers=US_GLOBAL_EQUITIES+OTHER_BULL_TICKERS):
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


def create_advanced_bull_feat(df_bull_raw, us_global_equities=US_GLOBAL_EQUITIES, rolling_window=20):
    """
    """

    df_eq = df_bull_raw[df_bull_raw['symbol'].isin(us_global_equities)].sort_index().copy()
    df_eq_mean = df_eq.groupby(df_eq.index).mean(numeric_only=True)

    # ---Trend Consistency Score ---
    df_eq_mean['glob_eq_trend_consistency'] = ((df_eq_mean['close'] - df_eq_mean['low']) / 
                                    (df_eq_mean['high'] - df_eq_mean['low'] + 1e-6)).rolling(rolling_window).mean()

    # --- Volume Trend Divergence ---
    df_eq_mean['glob_eq_volume_trend_divergence'] = (
        df_eq_mean['close'].pct_change().rolling(rolling_window)
        .corr(df_eq_mean['volume'].rolling(rolling_window).mean())
    )

    # --- Helper function to extract close price of a symbol ---
    def get_close(symbol):
        return df_bull_raw[df_bull_raw['symbol'] == symbol].sort_index()['close']

    # Get closes for external tickers
    tlt = get_close('TLT')
    gld = get_close('GLD')
    sphb = get_close('SPHB')
    splv = get_close('SPLV')

    external_df = pd.DataFrame(index=df_eq_mean.index)

    # --- High Beta vs. Low Volatility Spread ---
    external_df['beta_vol_spread'] = sphb.pct_change().rolling(rolling_window).mean() - \
                                    splv.pct_change().rolling(rolling_window).mean()

    # --- Correlations ---
    external_df['glob_eq_tlt_corr'] = df_eq_mean['close'].pct_change().rolling(rolling_window).corr(tlt.pct_change())
    external_df['glob_eq_gld_corr'] = df_eq_mean['close'].pct_change().rolling(rolling_window).corr(gld.pct_change())

    # Combine features and external features
    bull_features_df = pd.concat([df_eq_mean[['glob_eq_trend_consistency', 'glob_eq_volume_trend_divergence']],
                            external_df], axis=1)

    # Drop rows with missing values
    bull_features_df.dropna(inplace=True)

    return bull_features_df

