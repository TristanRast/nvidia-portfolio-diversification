#!/usr/bin/env python3
"""
Script 01: Data Collection from Yahoo Finance

This script downloads historical adjusted closing prices for NVIDIA and 
comparison stocks using the yfinance library.

Author: Tristan Rast
Date: March 11, 2026
WGU Capstone Project - BHN1 Task 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
from config.config import (
    START_DATE, END_DATE, TARGET_TICKER, COMPARISON_TICKERS,
    DATA_PATHS, get_all_tickers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_stock_data(ticker, start_date, end_date):
    """
    Download historical adjusted closing prices for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns
    -------
    pd.Series
        Time series of adjusted closing prices
    """
    try:
        logger.info(f"Downloading data for {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return None

        # Extract adjusted close prices
        adj_close = hist['Close']
        adj_close.name = ticker

        logger.info(f"  ✓ {ticker}: {len(adj_close)} data points from {adj_close.index[0].date()} to {adj_close.index[-1].date()}")
        return adj_close

    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        return None


def download_all_stocks(tickers, start_date, end_date):
    """
    Download historical data for all tickers.

    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted closing prices for all tickers
    """
    logger.info("="*80)
    logger.info("STARTING DATA COLLECTION")
    logger.info("="*80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info("")

    price_data = {}
    failed_tickers = []

    for ticker in tickers:
        data = download_stock_data(ticker, start_date, end_date)
        if data is not None:
            price_data[ticker] = data
        else:
            failed_tickers.append(ticker)

    if not price_data:
        raise ValueError("No data was successfully downloaded!")

    # Combine into DataFrame
    df = pd.DataFrame(price_data)

    logger.info("")
    logger.info("="*80)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successfully downloaded: {len(price_data)} stocks")
    logger.info(f"Failed downloads: {len(failed_tickers)}")
    if failed_tickers:
        logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    logger.info(f"Total observations per stock: {len(df)}")
    logger.info("")

    return df


def save_raw_data(df, filename='raw_prices.csv'):
    """
    Save raw price data to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    filename : str
        Output filename
    """
    output_path = os.path.join(DATA_PATHS['raw'], filename)
    os.makedirs(DATA_PATHS['raw'], exist_ok=True)

    df.to_csv(output_path)
    logger.info(f"✓ Raw data saved to: {output_path}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {', '.join(df.columns.tolist())}")


def main():
    """Main execution function."""
    try:
        # Get list of all tickers
        all_tickers = get_all_tickers()

        # Download data
        price_df = download_all_stocks(all_tickers, START_DATE, END_DATE)

        # Save raw data
        save_raw_data(price_df)

        # Display basic info
        logger.info("")
        logger.info("="*80)
        logger.info("SAMPLE DATA (First 5 rows)")
        logger.info("="*80)
        print(price_df.head())

        logger.info("")
        logger.info("="*80)
        logger.info("DATA COMPLETENESS CHECK")
        logger.info("="*80)
        missing_pct = (price_df.isnull().sum() / len(price_df) * 100).round(2)
        print(missing_pct[missing_pct > 0])

        logger.info("")
        logger.info("✅ DATA COLLECTION COMPLETE!")
        logger.info(f"Next step: Run 02_data_preparation.py")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()
