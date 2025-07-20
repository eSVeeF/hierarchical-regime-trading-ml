
custom_vol_subset = [
"SPY", # US Broad Market (Large Cap) Benchmark S&P 500, high liquidity, volatility anchor
"IWM", # US Broad Market (Small Cap) Small-cap exposure, higher realized volatility and skew
"XBI", # Biotech (Speculative)	Idiosyncratic volatility, high-vol sector
"TLT", # Long-Term US Treasuries  Bonds often move inversely to equities, and long-term treasuries are highly sensitive to interest rate changes and economic uncertainty
"HYG", # High Yield Corporate Bonds Junk bonds — sensitive to credit risk. A "risk-on/risk-off" indicator within fixed income
"VEA", # Developed Markets ex-US Equity Captures international developed market equity volatility, providing diversification from the US focus.
"VWO", # Emerging Markets Equity. Adds a distinct global equity volatility component.
"VIXY", # Volatility Instrument	Direct VIX futures exposure, useful for spread/skew metrics
]    





# # High Volatility vs. Low Volatility - Expansion vs. Contraction
# custom_vol_subset = [
#   ## For volatility  
#  "VIXY",  # VIX derivatives (direct volatility proxies)
#  "IWM",          # More volatile than SPY, especially small caps & tech. SPY only for comparison.
#  "HYG",                   # Junk bonds — sensitive to credit risk (volatility proxy)
#  "GLD",            # Gold as a safe haven during volatility spikes    
# ]
#  ## For expanding-contracting regimes
# custom_exp_subset = [
# "IWM", # (iShares Russell 2000 ETF): Small-cap companies are often more sensitive to the domestic economic cycle than their large-cap counterparts.
#  "BND", #(Vanguard Total Bond Market ETF): The bond market often behaves differently than the stock market across the business cycle, making it a valuable diversifying indicator.
# "SCHP", # (Schwab US TIPS ETF): Treasury Inflation-Protected Securities provide insight into the market's inflation expectations, a key component of the business cycle.
#  "XLI", # (Industrials Select Sector SPDR Fund): The industrial sector is highly cyclical and typically performs well during economic expansions
#  "CPER", #(United States Copper Index Fund): Copper is a key industrial metal, and its price is often considered a proxy for global economic health, earning it the nickname "Dr. Copper."  
#  "SCHP" #(Schwab US TIPS ETF): Treasury Inflation-Protected Securities provide insight into the market's inflation expectations, a key component of the business cycle.  
# ]

# ## #################################################3 Ver gemmin para explicar porque estos tickers y mucho mas


### List for various market segments
# 1. Broad Market Equity ETFs (USA)
broad_market_equity_etfs = [
    "VOO",  # Vanguard S&P 500 ETF
    "SPY",  # SPDR S&P 500 ETF Trust
    "IVV",  # iShares Core S&P 500 ETF
    "VTI",  # Vanguard Total Stock Market ETF
    "QQQ",  # Invesco QQQ Trust Series I (Nasdaq 100)
    "ITOT", # iShares Core S&P Total U.S. Stock Market ETF
    "SCHB", # Schwab US Broad Market ETF
    "IWM",  # iShares Russell 2000 ETF (Small-Cap)
    "IJH",  # iShares Core S&P Mid-Cap ETF
    "VO",   # Vanguard Mid-Cap ETF
    "VB",   # Vanguard Small Cap ETF
    "SCHD", # Schwab US Dividend Equity ETF
    "VYM",  # Vanguard High Dividend Yield Index ETF
    "VIG",  # Vanguard Dividend Appreciation ETF
    "RSP",  # Invesco S&P 500® Equal Weight ETF
    "VT",   # Vanguard Total World Stock ETF (includes ex-US)
]

# 2. Sector-Specific Equity ETFs (USA)
sector_specific_etfs = [
    "XLC",  # Communication Services Select Sector SPDR Fund
    "XLY",  # Consumer Discretionary Select Sector SPDR Fund
    "XLP",  # Consumer Staples Select Sector SPDR Fund
    "XLE",  # Energy Select Sector SPDR Fund
    "XLF",  # Financial Select Sector SPDR Fund
    "XLV",  # Health Care Select Sector SPDR Fund
    "XLI",  # Industrials Select Sector SPDR Fund
    "XLB",  # Materials Select Sector SPDR Fund
    "XLRE", # Real Estate Select Sector SPDR Fund
    "XLK",  # Technology Select Sector SPDR Fund
    "XLU",  # Utilities Select Sector SPDR Fund
    "KBE",  # SPDR S&P Bank ETF (Industry specific)
    "XBI",  # SPDR S&P Biotech ETF (Industry specific)
    "XOP",  # SPDR S&P Oil & Gas Exploration & Production ETF (Industry specific)
    "XRT",  # SPDR S&P Retail ETF (Industry specific)
    "SOXX", # iShares Semiconductor ETF (though XLK also has tech)
]

# 3. Fixed Income (Bond) ETFs
fixed_income_etfs = [
    "BND",  # Vanguard Total Bond Market ETF (broad aggregate)
    "AGG",  # iShares Core U.S. Aggregate Bond ETF (broad aggregate)
    "TLT",  # iShares 20+ Year Treasury Bond ETF (long-term treasuries)
    "IEF",  # iShares 7-10 Year Treasury Bond ETF (intermediate-term treasuries)
    "VGSH", # Vanguard Short-Term Treasury ETF (short-term treasuries)
    "VCIT", # Vanguard Intermediate-Term Corporate Bond ETF
    "VCLT", # Vanguard Long-Term Corporate Bond ETF
    "VCSH", # Vanguard Short-Term Corporate Bond ETF
    "BNDX", # Vanguard Total International Bond ETF
    "HYG",  # iShares iBoxx $ High Yield Corporate Bond ETF (high yield/junk bonds)
    "SJNK",  # SPDR Bloomberg Short-Term High Yield Bond ETF (high yield/junk bonds)
    "MUB",  # iShares National Muni Bond ETF (municipal bonds)
    "VTEB", # Vanguard Tax-Exempt Bond ETF (municipal bonds)
    "SCHP", # Schwab US TIPS ETF (inflation-protected securities)
    "VTIP", # Vanguard Short-Term Inflation-Protected Securities ETF
    "BSV",  # Vanguard Short-Term Bond ETF
    "MBB",  # iShares MBS ETF (mortgage-backed securities)
]

# 4. Commodity ETFs
commodity_etfs = [
    "GLD",  # SPDR Gold Shares (gold)
    "IAU",  # iShares Gold Trust (gold)
    "SLV",  # iShares Silver Trust (silver)
    "PDBC", # Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF (broad commodities)
    "DBC",  # Invesco DB Commodity Index Tracking Fund (broad commodities)
    "USO",  # United States Oil Fund, LP (crude oil)
    "UNG",  # United States Natural Gas Fund, LP (natural gas)
    "CORN", # Teucrium Corn Fund (corn)
    "WEAT", # Teucrium Wheat Fund (wheat)
    "SOYB", # Teucrium Soybean Fund (soybeans)
    "CPER", # United States Copper Index Fund (copper)
]

# 5. Volatility ETFs (Often complex, involving futures, and not for all investors)
volatility_etfs = [
    "VIXY", # ProShares VIX Short-Term Futures ETF (1x leveraged  VIX futures)
    "UVXY", # ProShares Ultra VIX Short-Term Futures ETF (2x leveraged VIX futures)
    "SVXY", # ProShares Short VIX Short-Term Futures ETF (inverse VIX futures)
    "SVIX", # -1x Short VIX Futures ETF (inverse VIX futures)
    "SVOL", # Simplify Volatility Premium ETF (aims for volatility premium)
]

# 6. International/Global Exposure ETFs
international_global_exposure_etfs = [
    "VEA",  # Vanguard FTSE Developed Markets ETF (developed ex-US)
    "IEFA", # iShares Core MSCI EAFE ETF (developed ex-US)
    "EFA",  # iShares MSCI EAFE ETF (developed ex-US)
    "VWO",  # Vanguard FTSE Emerging Markets ETF (emerging markets)
    "IEMG", # iShares Core MSCI Emerging Markets ETF (emerging markets)
    "ACWI", # iShares MSCI ACWI ETF (All Country World Index - developed + emerging)
    "VXUS", # Vanguard Total International Stock ETF (total international ex-US)
    "SCHF", # Schwab International Equity ETF (developed ex-US)
    "SPDW", # SPDR Portfolio Developed World ex-US ETF
    "EEM",  # iShares MSCI Emerging Markets ETF
    "VT",   # Vanguard Total World Stock ETF (as mentioned in Broad Market, it has global exposure)
]