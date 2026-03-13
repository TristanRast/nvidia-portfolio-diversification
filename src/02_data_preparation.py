#!/usr/bin/env python3
"""
Script 02: Data Preparation and Cleaning

This script loads raw price data, handles missing values, aligns trading days,
and computes daily returns for correlation analysis.

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
from config.config import DATA_PATHS, MIN_DATA_COMPLETENESS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(filename='raw_prices.csv'):
    """
    Load raw price data from CSV.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    pd.DataFrame
        Raw price data with DatetimeIndex
    """
    file_path = os.path.join(DATA_PATHS['raw'], filename)
    logger.info(f"Loading raw data from: {file_path}")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logger.info(f"  ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def check_data_quality(df):
    """
    Assess data quality and completeness.

    Parameters
    ----------
    df : pd.DataFrame
        Price data

    Returns
    -------
    pd.DataFrame
        Data quality report
    """
    logger.info("")
    logger.info("="*80)
    logger.info("DATA QUALITY ASSESSMENT")
    logger.info("="*80)

    quality_report = pd.DataFrame({
        'Total_Obs': len(df),
        'Missing_Count': df.isnull().sum(),
        'Missing_Pct': (df.isnull().sum() / len(df) * 100).round(2),
        'Completeness': ((1 - df.isnull().sum() / len(df)) * 100).round(2),
        'First_Date': df.apply(lambda x: x.first_valid_index().date() if x.first_valid_index() else None),
        'Last_Date': df.apply(lambda x: x.last_valid_index().date() if x.last_valid_index() else None),
    })

    quality_report['Passes_Threshold'] = quality_report['Completeness'] >= (MIN_DATA_COMPLETENESS * 100)

    logger.info(f"\nData Quality Report:")
    print(quality_report)

    # Check for stocks below threshold
    failing_stocks = quality_report[~quality_report['Passes_Threshold']]
    if not failing_stocks.empty:
        logger.warning(f"\n⚠ {len(failing_stocks)} stock(s) below {MIN_DATA_COMPLETENESS*100}% completeness threshold:")
        for ticker in failing_stocks.index:
            logger.warning(f"  - {ticker}: {quality_report.loc[ticker, 'Completeness']:.2f}%")
    else:
        logger.info(f"\n✓ All stocks meet {MIN_DATA_COMPLETENESS*100}% completeness threshold")

    return quality_report


def handle_missing_data(df, method='forward_fill'):
    """
    Handle missing data using specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with potential missing values
    method : str
        Method to handle missing data: 'forward_fill', 'drop', or 'interpolate'

    Returns
    -------
    pd.DataFrame
        Data with missing values handled
    """
    logger.info("")
    logger.info(f"Handling missing data using method: {method}")

    initial_missing = df.isnull().sum().sum()
    logger.info(f"  Initial missing values: {initial_missing}")

    if method == 'forward_fill':
        # Forward fill then backward fill for any remaining
        df_clean = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'drop':
        # Drop rows with any missing values
        df_clean = df.dropna()
    elif method == 'interpolate':
        # Linear interpolation
        df_clean = df.interpolate(method='linear', limit_direction='both')
    else:
        raise ValueError(f"Unknown method: {method}")

    final_missing = df_clean.isnull().sum().sum()
    logger.info(f"  Final missing values: {final_missing}")
    logger.info(f"  ✓ Cleaned {initial_missing - final_missing} missing values")

    return df_clean


def compute_daily_returns(df, return_type='simple'):
    """
    Compute daily returns from price data.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    return_type : str
        'simple' for simple returns or 'log' for log returns

    Returns
    -------
    pd.DataFrame
        Daily returns
    """
    logger.info("")
    logger.info(f"Computing {return_type} daily returns...")

    if return_type == 'simple':
        returns = df.pct_change()
    elif return_type == 'log':
        returns = np.log(df / df.shift(1))
    else:
        raise ValueError(f"Unknown return type: {return_type}")

    # Drop first row (NaN from pct_change)
    returns = returns.iloc[1:]

    logger.info(f"  ✓ Computed returns: {returns.shape[0]} periods × {returns.shape[1]} stocks")
    logger.info(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    return returns


def compute_descriptive_statistics(returns, prices):
    """
    Compute descriptive statistics for returns and prices.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    prices : pd.DataFrame
        Price data

    Returns
    -------
    pd.DataFrame
        Descriptive statistics
    """
    logger.info("")
    logger.info("Computing descriptive statistics...")

    stats = pd.DataFrame({
        'Mean_Return': returns.mean() * 252,  # Annualized
        'Volatility': returns.std() * np.sqrt(252),  # Annualized
        'Sharpe_Ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'Min_Return': returns.min(),
        'Max_Return': returns.max(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Starting_Price': prices.iloc[0],
        'Ending_Price': prices.iloc[-1],
        'Total_Return': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
    })

    stats = stats.round(4)

    logger.info("  ✓ Statistics computed")
    logger.info(f"\nDescriptive Statistics (Top 10 by Total Return):")
    print(stats.nlargest(10, 'Total_Return'))

    return stats


def save_processed_data(prices, returns, stats):
    """
    Save processed data to CSV files.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned price data
    returns : pd.DataFrame
        Daily returns
    stats : pd.DataFrame
        Descriptive statistics
    """
    os.makedirs(DATA_PATHS['processed'], exist_ok=True)

    # Save files
    prices_path = os.path.join(DATA_PATHS['processed'], 'aligned_prices.csv')
    returns_path = os.path.join(DATA_PATHS['processed'], 'daily_returns.csv')
    stats_path = os.path.join(DATA_PATHS['processed'], 'descriptive_statistics.csv')

    prices.to_csv(prices_path)
    returns.to_csv(returns_path)
    stats.to_csv(stats_path)

    logger.info("")
    logger.info("="*80)
    logger.info("SAVED PROCESSED DATA")
    logger.info("="*80)
    logger.info(f"✓ Aligned prices: {prices_path}")
    logger.info(f"✓ Daily returns: {returns_path}")
    logger.info(f"✓ Descriptive stats: {stats_path}")


def main():
    """Main execution function."""
    try:
        logger.info("="*80)
        logger.info("STARTING DATA PREPARATION")
        logger.info("="*80)

        # Load raw data
        raw_prices = load_raw_data()

        # Assess data quality
        quality_report = check_data_quality(raw_prices)

        # Handle missing data
        clean_prices = handle_missing_data(raw_prices, method='forward_fill')

        # Compute returns
        daily_returns = compute_daily_returns(clean_prices, return_type='simple')

        # Compute descriptive statistics
        desc_stats = compute_descriptive_statistics(daily_returns, clean_prices)

        # Save processed data
        save_processed_data(clean_prices, daily_returns, desc_stats)

        logger.info("")
        logger.info("✅ DATA PREPARATION COMPLETE!")
        logger.info(f"Next step: Run 03_correlation_analysis.py")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()
