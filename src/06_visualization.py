#!/usr/bin/env python3
"""
Script 06: Visualization

This script generates all visualizations for the analysis including correlation
heatmaps, time series plots, and portfolio comparisons.

Author: Tristan Rast
Date: March 11, 2026
WGU Capstone Project - BHN1 Task 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config.config import (
    TARGET_TICKER, DATA_PATHS, OUTPUT_PATHS, PLOT_SETTINGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context('notebook', font_scale=1.2)


def load_all_data():
    """Load all necessary data for visualization."""
    logger.info("Loading data for visualization...")

    data = {
        'returns': pd.read_csv(os.path.join(DATA_PATHS['processed'], 'daily_returns.csv'), 
                               index_col=0, parse_dates=True),
        'prices': pd.read_csv(os.path.join(DATA_PATHS['processed'], 'aligned_prices.csv'), 
                              index_col=0, parse_dates=True),
        'correlation_matrix': pd.read_csv(os.path.join(DATA_PATHS['results'], 'correlation_matrix.csv'), 
                                          index_col=0),
        'classification': pd.read_csv(os.path.join(DATA_PATHS['results'], 'stock_classification.csv'), 
                                      index_col=0),
        'rolling_6m': pd.read_csv(os.path.join(DATA_PATHS['results'], 'rolling_corr_6m.csv'), 
                                  index_col=0, parse_dates=True),
        'rolling_12m': pd.read_csv(os.path.join(DATA_PATHS['results'], 'rolling_corr_12m.csv'), 
                                   index_col=0, parse_dates=True),
    }

    logger.info("  [OK] All data loaded successfully")
    return data


def plot_correlation_heatmap(corr_matrix, output_path):
    """
    Create correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    output_path : str
        Output file path
    """
    logger.info("Creating correlation heatmap...")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=False,
                cmap='RdYlGn_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'})

    plt.title(f'Correlation Matrix: {TARGET_TICKER} and Comparison Stocks', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Stock Ticker', fontsize=12)
    plt.ylabel('Stock Ticker', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"  [OK] Saved: {output_path}")


def plot_nvda_correlations_barplot(classification, output_path):
    """
    Create bar plot of NVDA correlations sorted by magnitude.

    Parameters
    ----------
    classification : pd.DataFrame
        Stock classification data
    output_path : str
        Output file path
    """
    logger.info("Creating NVDA correlations bar plot...")

    # Sort by correlation value
    sorted_data = classification.sort_values('Pearson_r')

    # Create color map based on classification
    colors = []
    for idx, row in sorted_data.iterrows():
        if row['Classification'] == 'Non-correlated':
            colors.append('#2ecc71')  # Green
        elif row['Classification'] == 'Weakly correlated':
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(sorted_data))
    ax.barh(y_pos, sorted_data['Pearson_r'], color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_data.index, fontsize=9)
    ax.set_xlabel('Pearson Correlation with NVDA', fontsize=12, fontweight='bold')
    ax.set_title('Stock Correlations with NVIDIA (Sorted)', fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=0.25, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Threshold: 0.25')
    ax.axvline(x=-0.25, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0.60, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Threshold: 0.60')
    ax.axvline(x=-0.60, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Non-correlated'),
        Patch(facecolor='#f39c12', label='Weakly correlated'),
        Patch(facecolor='#e74c3c', label='Strongly correlated')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"  [OK] Saved: {output_path}")


def plot_correlation_distribution(classification, output_path):
    """
    Create histogram of correlation distribution.

    Parameters
    ----------
    classification : pd.DataFrame
        Stock classification data
    output_path : str
        Output file path
    """
    logger.info("Creating correlation distribution plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    ax.hist(classification['Pearson_r'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')

    # Add vertical lines for thresholds
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, label='Zero correlation')
    ax.axvline(x=0.25, color='green', linestyle='--', linewidth=1.5, label='Non-corr threshold (0.25)')
    ax.axvline(x=-0.25, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(x=0.60, color='red', linestyle='--', linewidth=1.5, label='Strong corr threshold (0.60)')
    ax.axvline(x=-0.60, color='red', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Pearson Correlation with NVDA', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Stock Correlations with NVIDIA', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"  [OK] Saved: {output_path}")

def plot_rolling_correlation(rolling_df, window_label, output_path):
    """Plot rolling correlation time series for top/bottom stocks."""
    logger.info(f"Creating rolling correlation plot ({window_label})...")

    # Select top 5 non-correlated and top 3 strongly correlated columns
    mean_corr = rolling_df.mean().sort_values()
    bottom_5 = mean_corr.head(5).index.tolist()
    top_3 = mean_corr.tail(3).index.tolist()
    selected = bottom_5 + top_3

    # Only keep columns that exist
    selected = [c for c in selected if c in rolling_df.columns]

    fig, ax = plt.subplots(figsize=(14, 7))

    for col in selected:
        ax.plot(rolling_df.index, rolling_df[col], linewidth=1.5, label=col, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Non-corr threshold')
    ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Strong corr threshold')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling Pearson Correlation', fontsize=12, fontweight='bold')
    ax.set_title(f'Rolling {window_label} Correlation with NVIDIA', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"  [OK] Saved: {output_path}")


def plot_cumulative_returns(prices, classification, output_path):
    """Plot cumulative returns for NVDA vs top non-correlated and top correlated stocks."""
    logger.info("Creating cumulative returns comparison plot...")

    # Pick NVDA + top 3 non-correlated + top 2 strongly correlated
    non_corr = classification[classification['Classification'] == 'Non-correlated'] \
        .sort_values('Pearson_r').head(3).index.tolist()
    strong_corr = classification[classification['Classification'] == 'Strongly correlated'] \
        .sort_values('Pearson_r', ascending=False).head(2).index.tolist()

    tickers = [TARGET_TICKER] + non_corr + strong_corr
    tickers = [t for t in tickers if t in prices.columns]

    # Compute cumulative returns
    cum_returns = (1 + prices[tickers].pct_change()).cumprod() - 1

    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker in tickers:
        lw = 2.5 if ticker == TARGET_TICKER else 1.5
        ls = '-' if ticker == TARGET_TICKER else '--' if ticker in non_corr else ':'
        ax.plot(cum_returns.index, cum_returns[ticker] * 100,
                linewidth=lw, linestyle=ls, label=ticker, alpha=0.85)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Returns: NVDA vs Selected Stocks (2021–2026)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"  [OK] Saved: {output_path}")


def plot_sector_correlation(classification, output_path):
    """Plot average correlation by sector as a horizontal bar chart."""
    logger.info("Creating sector correlation breakdown plot...")

    if 'Sector' not in classification.columns:
        logger.warning("  [SKIP] No 'Sector' column found in classification data.")
        # Create a placeholder chart so the pipeline doesn't break
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'Sector data not available',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Sector Correlation Breakdown', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
        plt.close()
        return

    sector_avg = classification.groupby('Sector')['Pearson_r'].mean().sort_values()

    colors = ['#2ecc71' if v < 0.25 else '#f39c12' if v < 0.60 else '#e74c3c'
              for v in sector_avg.values]

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(sector_avg))
    ax.barh(y_pos, sector_avg.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_avg.index, fontsize=11)
    ax.axvline(x=0.25, color='green', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Non-corr threshold (0.25)')
    ax.axvline(x=0.60, color='red', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Strong corr threshold (0.60)')
    ax.set_xlabel('Average Pearson Correlation with NVDA', fontsize=12, fontweight='bold')
    ax.set_title('Average Correlation by Sector', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(output_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"  [OK] Saved: {output_path}")


def main():
    """Main execution function."""
    try:
        logger.info("=" * 80)
        logger.info("STARTING VISUALIZATION")
        logger.info("=" * 80)

        # Ensure output directory exists
        os.makedirs(OUTPUT_PATHS['figures'], exist_ok=True)
        fig_dir = OUTPUT_PATHS['figures']

        # Load all data
        data = load_all_data()

        # 1. Correlation heatmap
        plot_correlation_heatmap(
            data['correlation_matrix'],
            os.path.join(fig_dir, 'correlation_heatmap.png')
        )

        # 2. NVDA correlations bar plot
        plot_nvda_correlations_barplot(
            data['classification'],
            os.path.join(fig_dir, 'nvda_correlations_barplot.png')
        )

        # 3. Correlation distribution
        plot_correlation_distribution(
            data['classification'],
            os.path.join(fig_dir, 'correlation_distribution.png')
        )

        # 4. Rolling correlation 6-month
        plot_rolling_correlation(
            data['rolling_6m'],
            window_label='6-Month',
            output_path=os.path.join(fig_dir, 'rolling_correlation_6m.png')
        )

        # 5. Rolling correlation 12-month
        plot_rolling_correlation(
            data['rolling_12m'],
            window_label='12-Month',
            output_path=os.path.join(fig_dir, 'rolling_correlation_12m.png')
        )

        # 6. Cumulative returns comparison
        plot_cumulative_returns(
            data['prices'],
            data['classification'],
            output_path=os.path.join(fig_dir, 'cumulative_returns_comparison.png')
        )

        # 7. Sector correlation breakdown
        plot_sector_correlation(
            data['classification'],
            output_path=os.path.join(fig_dir, 'sector_correlation_breakdown.png')
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("[OK] ALL VISUALIZATIONS COMPLETE!")
        logger.info(f"Figures saved to: {fig_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()
