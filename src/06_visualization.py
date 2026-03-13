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

