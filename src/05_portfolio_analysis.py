#!/usr/bin/env python3
"""
Script 05: Portfolio Analysis

This script evaluates the practical significance of diversification by comparing
portfolios with highly correlated vs non-correlated stocks paired with NVIDIA.

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
from config.config import TARGET_TICKER, DATA_PATHS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load necessary data for portfolio analysis."""
    returns_path = os.path.join(DATA_PATHS['processed'], 'daily_returns.csv')
    prices_path = os.path.join(DATA_PATHS['processed'], 'aligned_prices.csv')
    classification_path = os.path.join(DATA_PATHS['results'], 'stock_classification.csv')

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    classification = pd.read_csv(classification_path, index_col=0)

    logger.info("[OK] Loaded returns, prices, and classification data")
    return returns, prices, classification


def create_portfolio(returns, tickers, weights=None):
    """
    Create a portfolio from specified tickers with given weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    tickers : list
        List of ticker symbols
    weights : list, optional
        Portfolio weights (equal weight if None)

    Returns
    -------
    pd.Series
        Portfolio returns
    """
    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)

    portfolio_returns = (returns[tickers] * weights).sum(axis=1)
    return portfolio_returns


def calculate_portfolio_metrics(returns):
    """
    Calculate key portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Portfolio returns

    Returns
    -------
    dict
        Portfolio metrics
    """
    metrics = {
        'Annual_Return': returns.mean() * 252,
        'Annual_Volatility': returns.std() * np.sqrt(252),
        'Sharpe_Ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'Max_Drawdown': calculate_max_drawdown(returns),
        'Calmar_Ratio': (returns.mean() * 252) / abs(calculate_max_drawdown(returns)) if calculate_max_drawdown(returns) != 0 else np.nan,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'VaR_95': returns.quantile(0.05),
        'CVaR_95': returns[returns <= returns.quantile(0.05)].mean(),
    }
    return metrics


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from returns series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns

    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def compare_portfolio_scenarios(returns, classification):
    """
    Compare different portfolio scenarios: NVDA only, NVDA + correlated, NVDA + non-correlated.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    classification : pd.DataFrame
        Stock classification results

    Returns
    -------
    pd.DataFrame
        Comparison of portfolio scenarios
    """
    logger.info("")
    logger.info("="*80)
    logger.info("PORTFOLIO SCENARIO COMPARISON")
    logger.info("="*80)

    scenarios = {}

    # Scenario 1: NVDA only
    logger.info("\nScenario 1: NVIDIA only (100% NVDA)")
    nvda_returns = returns[TARGET_TICKER]
    scenarios['NVDA_Only'] = calculate_portfolio_metrics(nvda_returns)

    # Scenario 2: NVDA + Strongly Correlated Stock
    strongly_corr = classification[classification['Classification'] == 'Strongly correlated']
    if not strongly_corr.empty:
        # Pick most correlated stock
        most_corr_ticker = strongly_corr.nlargest(1, 'Abs_Correlation').index[0]
        logger.info(f"\nScenario 2: NVIDIA + Strongly Correlated ({most_corr_ticker})")
        logger.info(f"  Correlation with NVDA: {strongly_corr.loc[most_corr_ticker, 'Pearson_r']:.4f}")

        portfolio_corr = create_portfolio(returns, [TARGET_TICKER, most_corr_ticker], weights=[0.5, 0.5])
        scenarios['NVDA_Correlated'] = calculate_portfolio_metrics(portfolio_corr)

    # Scenario 3: NVDA + Non-Correlated Stock
    non_corr = classification[classification['Classification'] == 'Non-correlated']
    if not non_corr.empty:
        # Pick least correlated stock
        least_corr_ticker = non_corr.nsmallest(1, 'Abs_Correlation').index[0]
        logger.info(f"\nScenario 3: NVIDIA + Non-Correlated ({least_corr_ticker})")
        logger.info(f"  Correlation with NVDA: {non_corr.loc[least_corr_ticker, 'Pearson_r']:.4f}")

        portfolio_noncorr = create_portfolio(returns, [TARGET_TICKER, least_corr_ticker], weights=[0.5, 0.5])
        scenarios['NVDA_NonCorrelated'] = calculate_portfolio_metrics(portfolio_noncorr)
    else:
        # Use weakly correlated if no non-correlated available
        weak_corr = classification[classification['Classification'] == 'Weakly correlated']
        if not weak_corr.empty:
            least_corr_ticker = weak_corr.nsmallest(1, 'Abs_Correlation').index[0]
            logger.info(f"\nScenario 3: NVIDIA + Weakly Correlated ({least_corr_ticker})")
            logger.info(f"  Correlation with NVDA: {weak_corr.loc[least_corr_ticker, 'Pearson_r']:.4f}")

            portfolio_weak = create_portfolio(returns, [TARGET_TICKER, least_corr_ticker], weights=[0.5, 0.5])
            scenarios['NVDA_NonCorrelated'] = calculate_portfolio_metrics(portfolio_weak)

    # Convert to DataFrame
    comparison_df = pd.DataFrame(scenarios).T

    logger.info("\n" + "="*80)
    logger.info("PORTFOLIO METRICS COMPARISON")
    logger.info("="*80)
    print(comparison_df.round(4))

    # Calculate improvements
    if 'NVDA_NonCorrelated' in scenarios and 'NVDA_Correlated' in scenarios:
        vol_reduction = ((scenarios['NVDA_Correlated']['Annual_Volatility'] - 
                         scenarios['NVDA_NonCorrelated']['Annual_Volatility']) / 
                        scenarios['NVDA_Correlated']['Annual_Volatility'] * 100)

        sharpe_improvement = ((scenarios['NVDA_NonCorrelated']['Sharpe_Ratio'] - 
                              scenarios['NVDA_Correlated']['Sharpe_Ratio']) / 
                             abs(scenarios['NVDA_Correlated']['Sharpe_Ratio']) * 100)

        logger.info("\n" + "="*80)
        logger.info("DIVERSIFICATION BENEFIT")
        logger.info("="*80)
        logger.info(f"Volatility Reduction (Non-corr vs Corr): {vol_reduction:.2f}%")
        logger.info(f"Sharpe Ratio Improvement: {sharpe_improvement:.2f}%")

    return comparison_df


def analyze_diversified_portfolio(returns, classification, n_stocks=5):
    """
    Analyze a diversified portfolio using top non-correlated stocks.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    classification : pd.DataFrame
        Stock classification
    n_stocks : int
        Number of non-correlated stocks to include

    Returns
    -------
    dict
        Diversified portfolio metrics
    """
    logger.info("")
    logger.info("="*80)
    logger.info(f"DIVERSIFIED PORTFOLIO ANALYSIS (NVDA + {n_stocks} non-correlated)")
    logger.info("="*80)

    # Get top N non-correlated stocks
    non_corr = classification.nsmallest(n_stocks, 'Abs_Correlation')

    logger.info(f"\nSelected {n_stocks} stocks with lowest correlation to NVDA:")
    print(non_corr[['Sector', 'Pearson_r', 'Abs_Correlation', 'Classification']])

    # Create diversified portfolio (equal weight)
    tickers = [TARGET_TICKER] + non_corr.index.tolist()
    div_portfolio = create_portfolio(returns, tickers)
    div_metrics = calculate_portfolio_metrics(div_portfolio)

    # Compare to NVDA only
    nvda_metrics = calculate_portfolio_metrics(returns[TARGET_TICKER])

    logger.info("\n" + "="*80)
    logger.info("DIVERSIFIED vs NVDA-ONLY COMPARISON")
    logger.info("="*80)
    comparison = pd.DataFrame({
        'NVDA_Only': nvda_metrics,
        'Diversified_Portfolio': div_metrics
    }).T
    print(comparison.round(4))

    # Calculate benefit
    vol_reduction = ((nvda_metrics['Annual_Volatility'] - div_metrics['Annual_Volatility']) / 
                     nvda_metrics['Annual_Volatility'] * 100)

    logger.info(f"\nVolatility Reduction: {vol_reduction:.2f}%")
    logger.info(f"Risk-Adjusted Return Improvement (Sharpe): {(div_metrics['Sharpe_Ratio'] - nvda_metrics['Sharpe_Ratio']):.4f}")

    return div_metrics


def save_portfolio_results(comparison_df, diversified_metrics):
    """Save portfolio analysis results."""
    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    comparison_path = os.path.join(DATA_PATHS['results'], 'portfolio_comparison.csv')
    diversified_path = os.path.join(DATA_PATHS['results'], 'diversified_portfolio_metrics.csv')

    comparison_df.to_csv(comparison_path)
    pd.DataFrame([diversified_metrics]).to_csv(diversified_path)

    logger.info("")
    logger.info("="*80)
    logger.info("SAVED PORTFOLIO ANALYSIS RESULTS")
    logger.info("="*80)
    logger.info(f"[OK] Portfolio comparison: {comparison_path}")
    logger.info(f"[OK] Diversified metrics: {diversified_path}")


def main():
    """Main execution function."""
    try:
        logger.info("="*80)
        logger.info("STARTING PORTFOLIO ANALYSIS")
        logger.info("="*80)

        # Load data
        returns, prices, classification = load_data()

        # Compare portfolio scenarios
        comparison = compare_portfolio_scenarios(returns, classification)

        # Analyze diversified portfolio
        diversified = analyze_diversified_portfolio(returns, classification, n_stocks=5)

        # Save results
        save_portfolio_results(comparison, diversified)

        logger.info("")
        logger.info("[OK] PORTFOLIO ANALYSIS COMPLETE!")
        logger.info(f"Next step: Run 06_visualization.py")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()
