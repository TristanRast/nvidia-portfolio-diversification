#!/usr/bin/env python3
"""
Script 03: Correlation Analysis

This script computes Pearson and Spearman correlation coefficients between
NVIDIA and all comparison stocks, then classifies stocks by correlation strength.

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
from scipy.stats import pearsonr, spearmanr
from config.config import (
    TARGET_TICKER, DATA_PATHS, CORRELATION_THRESHOLDS,
    classify_correlation, get_sector_for_ticker
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_returns_data(filename='daily_returns.csv'):
    """
    Load daily returns data.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    pd.DataFrame
        Daily returns data
    """
    file_path = os.path.join(DATA_PATHS['processed'], filename)
    logger.info(f"Loading returns data from: {file_path}")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logger.info(f"  [OK] Loaded {df.shape[0]} periods for {df.shape[1]} stocks")

    return df


def compute_correlation_matrix(returns, method='pearson'):
    """
    Compute correlation matrix using specified method.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    method : str
        'pearson' or 'spearman'

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    logger.info(f"\nComputing {method} correlation matrix...")

    if method == 'pearson':
        corr_matrix = returns.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = returns.corr(method='spearman')
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"  [OK] Correlation matrix computed: {corr_matrix.shape}")

    return corr_matrix


def extract_target_correlations(corr_matrix, target=TARGET_TICKER):
    """
    Extract correlations with target stock (NVIDIA).

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Full correlation matrix
    target : str
        Target ticker symbol

    Returns
    -------
    pd.Series
        Correlations with target stock
    """
    if target not in corr_matrix.columns:
        raise ValueError(f"Target {target} not found in correlation matrix")

    target_corr = corr_matrix[target].copy()
    # Remove self-correlation
    target_corr = target_corr[target_corr.index != target]

    logger.info(f"\nExtracted correlations with {target}")
    logger.info(f"  Number of comparisons: {len(target_corr)}")
    logger.info(f"  Range: [{target_corr.min():.4f}, {target_corr.max():.4f}]")
    logger.info(f"  Mean: {target_corr.mean():.4f}")
    logger.info(f"  Median: {target_corr.median():.4f}")

    return target_corr


def compute_correlation_with_pvalues(returns, target=TARGET_TICKER):
    """
    Compute correlations with statistical significance (p-values).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    target : str
        Target ticker

    Returns
    -------
    pd.DataFrame
        Correlations with p-values
    """
    logger.info(f"\nComputing correlations with p-values for {target}...")

    target_returns = returns[target]
    results = []

    for ticker in returns.columns:
        if ticker == target:
            continue

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(target_returns, returns[ticker])

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(target_returns, returns[ticker])

        results.append({
            'Ticker': ticker,
            'Sector': get_sector_for_ticker(ticker),
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
        })

    results_df = pd.DataFrame(results).set_index('Ticker')
    logger.info(f"  [OK] Computed correlations for {len(results_df)} stocks")

    return results_df


def classify_stocks_by_correlation(corr_results):
    """
    Classify stocks based on correlation strength.

    Parameters
    ----------
    corr_results : pd.DataFrame
        Correlation results with Pearson_r column

    Returns
    -------
    pd.DataFrame
        Classification results
    """
    logger.info("\nClassifying stocks by correlation strength...")

    classification = corr_results.copy()
    classification['Abs_Correlation'] = classification['Pearson_r'].abs()
    classification['Classification'] = classification['Pearson_r'].apply(classify_correlation)

    # Sort by absolute correlation
    classification = classification.sort_values('Abs_Correlation', ascending=True)

    # Count by category
    counts = classification['Classification'].value_counts()
    logger.info(f"\nClassification Summary:")
    for category, count in counts.items():
        logger.info(f"  {category}: {count} stocks")

    # Display thresholds
    logger.info(f"\nThresholds used:")
    logger.info(f"  Non-correlated:      |r| < {CORRELATION_THRESHOLDS['non_correlated']}")
    logger.info(f"  Weakly correlated:   {CORRELATION_THRESHOLDS['non_correlated']} <= |r| < {CORRELATION_THRESHOLDS['weakly_correlated']}")
    logger.info(f"  Strongly correlated: |r| >= {CORRELATION_THRESHOLDS['strongly_correlated']}")

    return classification


def identify_top_noncorrelated_stocks(classification, n=10):
    """
    Identify top N stocks with lowest correlation to NVIDIA.

    Parameters
    ----------
    classification : pd.DataFrame
        Classification results
    n : int
        Number of top stocks to identify

    Returns
    -------
    pd.DataFrame
        Top N non-correlated stocks
    """
    logger.info(f"\nIdentifying top {n} non-correlated stocks...")

    top_stocks = classification.nsmallest(n, 'Abs_Correlation')

    logger.info(f"\nTop {n} Non-Correlated Stocks with NVIDIA:")
    print(top_stocks[['Sector', 'Pearson_r', 'Abs_Correlation', 'Classification', 'Pearson_p']])

    return top_stocks


def save_correlation_results(corr_matrix, classification, top_noncorr):
    """
    Save correlation analysis results.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Full correlation matrix
    classification : pd.DataFrame
        Stock classification results
    top_noncorr : pd.DataFrame
        Top non-correlated stocks
    """
    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    corr_path = os.path.join(DATA_PATHS['results'], 'correlation_matrix.csv')
    class_path = os.path.join(DATA_PATHS['results'], 'stock_classification.csv')
    top_path = os.path.join(DATA_PATHS['results'], 'top_noncorrelated_stocks.csv')

    corr_matrix.to_csv(corr_path)
    classification.to_csv(class_path)
    top_noncorr.to_csv(top_path)

    logger.info("")
    logger.info("="*80)
    logger.info("SAVED CORRELATION RESULTS")
    logger.info("="*80)
    logger.info(f"[OK] Correlation matrix: {corr_path}")
    logger.info(f"[OK] Stock classification: {class_path}")
    logger.info(f"[OK] Top non-correlated: {top_path}")


def main():
    """Main execution function."""
    try:
        logger.info("="*80)
        logger.info("STARTING CORRELATION ANALYSIS")
        logger.info("="*80)

        # Load returns data
        returns = load_returns_data()

        # Compute correlation matrices
        pearson_corr = compute_correlation_matrix(returns, method='pearson')
        spearman_corr = compute_correlation_matrix(returns, method='spearman')

        # Extract NVIDIA correlations
        nvda_corr = extract_target_correlations(pearson_corr)

        # Compute correlations with p-values
        corr_with_pvalues = compute_correlation_with_pvalues(returns)

        # Classify stocks
        classification = classify_stocks_by_correlation(corr_with_pvalues)

        # Identify top non-correlated stocks
        top_noncorrelated = identify_top_noncorrelated_stocks(classification, n=10)

        # Save results
        save_correlation_results(pearson_corr, classification, top_noncorrelated)

        logger.info("")
        logger.info("[OK] CORRELATION ANALYSIS COMPLETE!")
        logger.info(f"Next step: Run 04_rolling_correlation.py")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()
