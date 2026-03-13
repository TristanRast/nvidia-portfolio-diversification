#!/usr/bin/env python3
"""
Script 04: Rolling Correlation Analysis

This script computes time-varying correlations using rolling windows to assess
the stability of correlation patterns over time.

Author: Tristan Rast
Date: March 11, 2026
WGU Capstone Project - BHN1 Task 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from config.config import (
    TARGET_TICKER, DATA_PATHS, ROLLING_WINDOWS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_returns_data(filename='daily_returns.csv'):
    """Load daily returns data."""
    file_path = os.path.join(DATA_PATHS['processed'], filename)
    logger.info(f"Loading returns data from: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logger.info(f"  [OK] Loaded {df.shape[0]} periods × {df.shape[1]} stocks")
    return df


def compute_rolling_correlation(returns, target, window, min_periods=None):
    """
    Compute rolling correlation between target and all other stocks.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    target : str
        Target ticker symbol
    window : int
        Rolling window size in trading days
    min_periods : int, optional
        Minimum observations required

    Returns
    -------
    pd.DataFrame
        Rolling correlations for each stock
    """
    if min_periods is None:
        min_periods = window // 2

    logger.info(f"\nComputing rolling correlation (window={window} days)...")

    target_returns = returns[target]
    rolling_corr = pd.DataFrame(index=returns.index)

    for ticker in returns.columns:
        if ticker == target:
            continue

        # Compute rolling correlation
        rolling_corr[ticker] = returns[ticker].rolling(
            window=window,
            min_periods=min_periods
        ).corr(target_returns)

    # Drop initial NaN rows
    rolling_corr = rolling_corr.dropna(how='all')

    logger.info(f"  [OK] Computed rolling correlations for {rolling_corr.shape[1]} stocks")
    logger.info(f"  [OK] Date range: {rolling_corr.index[0].date()} to {rolling_corr.index[-1].date()}")
    logger.info(f"  [OK] Observations: {len(rolling_corr)}")

    return rolling_corr


def analyze_correlation_stability(rolling_corr, window_name=''):
    """
    Analyze stability of correlations over time.

    Parameters
    ----------
    rolling_corr : pd.DataFrame
        Rolling correlation data
    window_name : str
        Description of the window period

    Returns
    -------
    pd.DataFrame
        Stability metrics
    """
    logger.info(f"\nAnalyzing correlation stability ({window_name})...")

    stability_metrics = pd.DataFrame({
        'Mean_Correlation': rolling_corr.mean(),
        'Std_Correlation': rolling_corr.std(),
        'Min_Correlation': rolling_corr.min(),
        'Max_Correlation': rolling_corr.max(),
        'Range': rolling_corr.max() - rolling_corr.min(),
        'Coefficient_of_Variation': rolling_corr.std() / rolling_corr.mean().abs(),
    })

    # Sort by stability (lower CV = more stable)
    stability_metrics = stability_metrics.sort_values('Coefficient_of_Variation')

    logger.info(f"\nMost Stable Correlations (Top 10 by CV):")
    print(stability_metrics.head(10).round(4))

    logger.info(f"\nLeast Stable Correlations (Bottom 10 by CV):")
    print(stability_metrics.tail(10).round(4))

    return stability_metrics


def identify_consistently_noncorrelated(rolling_corr, threshold=0.25):
    """
    Identify stocks that remain non-correlated across most time periods.

    Parameters
    ----------
    rolling_corr : pd.DataFrame
        Rolling correlation data
    threshold : float
        Correlation threshold for non-correlated classification

    Returns
    -------
    pd.DataFrame
        Stocks with consistent non-correlation
    """
    logger.info(f"\nIdentifying consistently non-correlated stocks (|r| < {threshold})...")

    # Calculate percentage of time each stock is non-correlated
    pct_noncorrelated = (rolling_corr.abs() < threshold).sum() / len(rolling_corr) * 100

    consistency_df = pd.DataFrame({
        'Pct_NonCorrelated': pct_noncorrelated,
        'Mean_Abs_Correlation': rolling_corr.abs().mean(),
        'Max_Abs_Correlation': rolling_corr.abs().max(),
    }).sort_values('Pct_NonCorrelated', ascending=False)

    # Filter stocks that are non-correlated at least 75% of the time
    consistent_stocks = consistency_df[consistency_df['Pct_NonCorrelated'] >= 75.0]

    logger.info(f"\nStocks non-correlated ≥75% of the time: {len(consistent_stocks)}")
    if not consistent_stocks.empty:
        print(consistent_stocks.round(2))
    else:
        logger.info("  No stocks meet the 75% threshold")
        logger.info(f"\nTop 10 by consistency:")
        print(consistency_df.head(10).round(2))

    return consistency_df


def compute_correlation_regime_changes(rolling_corr):
    """
    Detect periods where correlations change significantly.

    Parameters
    ----------
    rolling_corr : pd.DataFrame
        Rolling correlation data

    Returns
    -------
    pd.DataFrame
        Regime change metrics
    """
    logger.info("\nAnalyzing correlation regime changes...")

    # Compute rolling standard deviation of correlations (volatility of correlation)
    corr_volatility = rolling_corr.rolling(window=60, min_periods=30).std()

    regime_metrics = pd.DataFrame({
        'Avg_Correlation_Volatility': corr_volatility.mean(),
        'Max_Correlation_Volatility': corr_volatility.max(),
        'Regime_Instability_Score': corr_volatility.mean() / rolling_corr.abs().mean(),
    }).sort_values('Regime_Instability_Score', ascending=False)

    logger.info(f"\nMost Regime-Unstable Stocks (Top 10):")
    print(regime_metrics.head(10).round(4))

    return regime_metrics


def save_rolling_correlation_results(rolling_corr_short, rolling_corr_long, 
                                      stability_short, stability_long, consistency_long):
    """Save rolling correlation analysis results."""
    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    # Save rolling correlations
    rolling_short_path = os.path.join(DATA_PATHS['results'], 'rolling_corr_6m.csv')
    rolling_long_path = os.path.join(DATA_PATHS['results'], 'rolling_corr_12m.csv')

    rolling_corr_short.to_csv(rolling_short_path)
    rolling_corr_long.to_csv(rolling_long_path)

    # Save stability metrics
    stability_short_path = os.path.join(DATA_PATHS['results'], 'correlation_stability_6m.csv')
    stability_long_path = os.path.join(DATA_PATHS['results'], 'correlation_stability_12m.csv')

    stability_short.to_csv(stability_short_path)
    stability_long.to_csv(stability_long_path)

    # Save consistency analysis
    consistency_path = os.path.join(DATA_PATHS['results'], 'correlation_consistency.csv')
    consistency_long.to_csv(consistency_path)

    logger.info("")
    logger.info("="*80)
    logger.info("SAVED ROLLING CORRELATION RESULTS")
    logger.info("="*80)
    logger.info(f"[OK] Rolling correlation (6m): {rolling_short_path}")
    logger.info(f"[OK] Rolling correlation (12m): {rolling_long_path}")
    logger.info(f"[OK] Stability metrics (6m): {stability_short_path}")
    logger.info(f"[OK] Stability metrics (12m): {stability_long_path}")
    logger.info(f"[OK] Consistency analysis: {consistency_path}")


def main():
    """Main execution function."""
    try:
        logger.info("="*80)
        logger.info("STARTING ROLLING CORRELATION ANALYSIS")
        logger.info("="*80)

        # Load returns data
        returns = load_returns_data()

        # Compute rolling correlations for both windows
        rolling_corr_6m = compute_rolling_correlation(
            returns, 
            TARGET_TICKER, 
            window=ROLLING_WINDOWS['short_term']
        )

        rolling_corr_12m = compute_rolling_correlation(
            returns, 
            TARGET_TICKER, 
            window=ROLLING_WINDOWS['long_term']
        )

        # Analyze correlation stability
        stability_6m = analyze_correlation_stability(rolling_corr_6m, '6-month window')
        stability_12m = analyze_correlation_stability(rolling_corr_12m, '12-month window')

        # Identify consistently non-correlated stocks
        consistency_12m = identify_consistently_noncorrelated(rolling_corr_12m, threshold=0.25)

        # Analyze regime changes
        regime_changes = compute_correlation_regime_changes(rolling_corr_12m)

        # Save results
        save_rolling_correlation_results(
            rolling_corr_6m, rolling_corr_12m,
            stability_6m, stability_12m, consistency_12m
        )

        logger.info("")
        logger.info("[OK] ROLLING CORRELATION ANALYSIS COMPLETE!")
        logger.info(f"Next step: Run 05_portfolio_analysis.py")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()
