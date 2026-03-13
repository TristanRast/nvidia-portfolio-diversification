"""
Configuration file for NVIDIA Portfolio Diversification Analysis
Author: Tristan Rast
Date: March 11, 2026
"""

# ============================================================================
# DATE RANGE CONFIGURATION
# ============================================================================
START_DATE = '2021-03-11'  # 5 years of historical data
END_DATE = '2026-03-11'    # Current date

# ============================================================================
# STOCK UNIVERSE - 27 comparison stocks across 7 sectors
# ============================================================================
# Target stock for diversification analysis
TARGET_TICKER = 'NVDA'

# Comparison universe: 27 stocks across diverse sectors for diversification
COMPARISON_TICKERS = {
    'Technology': [
        'AAPL',   # Apple Inc.
        'MSFT',   # Microsoft Corporation
        'GOOGL',  # Alphabet Inc.
        'AMD',    # Advanced Micro Devices (semiconductor peer)
    ],
    'Healthcare': [
        'JNJ',    # Johnson & Johnson
        'UNH',    # UnitedHealth Group
        'PFE',    # Pfizer Inc.
        'ABBV',   # AbbVie Inc.
    ],
    'Financials': [
        'JPM',    # JPMorgan Chase
        'BAC',    # Bank of America
        'GS',     # Goldman Sachs
        'V',      # Visa Inc.
    ],
    'Consumer_Staples': [
        'PG',     # Procter & Gamble
        'KO',     # Coca-Cola Company
        'WMT',    # Walmart Inc.
        'COST',   # Costco Wholesale
    ],
    'Utilities': [
        'NEE',    # NextEra Energy
        'DUK',    # Duke Energy
        'SO',     # Southern Company
        'D',      # Dominion Energy
    ],
    'Energy': [
        'XOM',    # Exxon Mobil
        'CVX',    # Chevron Corporation
        'COP',    # ConocoPhillips
    ],
    'Real_Estate': [
        'AMT',    # American Tower Corp
        'PLD',    # Prologis Inc.
        'SPG',    # Simon Property Group
        'EQIX',   # Equinix Inc.
    ]
}

# ============================================================================
# CORRELATION CLASSIFICATION THRESHOLDS
# ============================================================================
CORRELATION_THRESHOLDS = {
    'non_correlated': 0.25,      # |r| < 0.25 → Non-correlated
    'weakly_correlated': 0.60,   # 0.25 <= |r| < 0.60 → Weakly correlated
    'strongly_correlated': 0.60   # |r| >= 0.60 → Strongly correlated
}

# ============================================================================
# ROLLING CORRELATION WINDOWS (in trading days)
# ============================================================================
ROLLING_WINDOWS = {
    'short_term': 126,   # Approximately 6 months
    'long_term': 252     # Approximately 12 months (1 year)
}

# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================
MIN_DATA_COMPLETENESS = 0.95  # Require 95% of expected trading days

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RANDOM_SEED = 42  # For reproducibility

# File paths
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'results': 'data/results'
}

OUTPUT_PATHS = {
    'figures': 'outputs/figures',
    'reports': 'outputs/reports'
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_SETTINGS = {
    'figsize_default': (12, 8),
    'figsize_wide': (16, 6),
    'figsize_square': (10, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'context': 'notebook',
    'font_scale': 1.2,
    'cmap_correlation': 'RdYlGn_r',  # Red (negative) to Green (positive)
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_tickers():
    """Get combined list of all tickers including target."""
    all_comparison = []
    for sector_stocks in COMPARISON_TICKERS.values():
        all_comparison.extend(sector_stocks)
    return [TARGET_TICKER] + all_comparison

def get_sector_for_ticker(ticker):
    """Get sector name for a given ticker."""
    if ticker == TARGET_TICKER:
        return 'Target'
    for sector, tickers in COMPARISON_TICKERS.items():
        if ticker in tickers:
            return sector
    return 'Unknown'

def classify_correlation(correlation_value):
    """
    Classify correlation value into category.

    Parameters
    ----------
    correlation_value : float
        Correlation coefficient value

    Returns
    -------
    str
        Classification: 'Non-correlated', 'Weakly correlated', or 'Strongly correlated'
    """
    abs_corr = abs(correlation_value)

    if abs_corr < CORRELATION_THRESHOLDS['non_correlated']:
        return 'Non-correlated'
    elif abs_corr < CORRELATION_THRESHOLDS['weakly_correlated']:
        return 'Weakly correlated'
    else:
        return 'Strongly correlated'


if __name__ == '__main__':
    # Print configuration summary
    print("="*80)
    print("NVIDIA PORTFOLIO DIVERSIFICATION - CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nTarget Stock: {TARGET_TICKER}")
    print(f"Analysis Period: {START_DATE} to {END_DATE}")
    print(f"\nComparison Universe: {len(get_all_tickers()) - 1} stocks")
    print("\nSector Breakdown:")
    for sector, tickers in COMPARISON_TICKERS.items():
        print(f"  {sector:20s}: {len(tickers)} stocks - {', '.join(tickers)}")
    print(f"\nTotal Tickers (including NVDA): {len(get_all_tickers())}")
    print(f"\nCorrelation Thresholds:")
    print(f"  Non-correlated:      |r| < {CORRELATION_THRESHOLDS['non_correlated']}")
    print(f"  Weakly correlated:   {CORRELATION_THRESHOLDS['non_correlated']} <= |r| < {CORRELATION_THRESHOLDS['weakly_correlated']}")
    print(f"  Strongly correlated: |r| >= {CORRELATION_THRESHOLDS['strongly_correlated']}")
    print("="*80)
