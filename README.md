# NVIDIA Portfolio Diversification Analysis

**Author:** Tristan Rast  
**Project:** WGU Data Analytics Capstone  
**Date:** March 11, 2026  
**GitHub Repository:** https://github.com/TristanRast/nvidia-portfolio-diversification

---

## Project Overview

This project analyzes historical stock correlations to identify stocks that can effectively diversify portfolio risk for investors holding NVIDIA (NVDA). Using 5 years of daily price data (2021-2026), the analysis computes Pearson and Spearman correlations, classifies stocks by correlation strength, and evaluates diversification benefits through portfolio simulations.

### Research Question

**Which publicly traded stocks exhibit low or negative return correlation with NVIDIA over a defined historical period, and how can these stocks be identified and classified using historical price data?**

### Key Objectives

1. Collect 5 years of daily adjusted closing prices for NVIDIA and 27 comparison stocks
2. Compute correlation coefficients and classify stocks (non-correlated, weakly correlated, strongly correlated)
3. Analyze correlation stability over time using rolling windows
4. Evaluate practical diversification benefits through portfolio metrics
5. Generate professional visualizations to communicate findings

---

## Business Value

- **Risk Management:** Identify stocks that don't move in lockstep with NVIDIA
- **Portfolio Construction:** Build diversified portfolios with reduced concentration risk
- **Decision Support:** Provide quantitative evidence for allocation decisions
- **Ongoing Monitoring:** Framework can be reused with updated data

---

## Dataset

- **Source:** Yahoo Finance API (via yfinance library)
- **Target Stock:** NVDA (NVIDIA Corporation)
- **Comparison Universe:** 27 stocks across 7 sectors
  - Technology (4 stocks): AAPL, MSFT, GOOGL, AMD
  - Healthcare (4 stocks): JNJ, UNH, PFE, ABBV
  - Financials (4 stocks): JPM, BAC, GS, V
  - Consumer Staples (4 stocks): PG, KO, WMT, COST
  - Utilities (4 stocks): NEE, DUK, SO, D
  - Energy (3 stocks): XOM, CVX, COP
  - Real Estate (4 stocks): AMT, PLD, SPG, EQIX
- **Time Period:** March 11, 2021 - March 11, 2026 (5 years)
- **Frequency:** Daily adjusted closing prices

---

## Tech Stack

- **Language:** Python 3.8+
- **IDE:** Visual Studio Code
- **Version Control:** Git & GitHub
- **Key Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `yfinance` - Financial data retrieval
  - `matplotlib` & `seaborn` - Visualization
  - `scipy` - Statistical analysis
  - `scikit-learn` - Additional metrics

---

## Project Structure

```
nvidia-portfolio-diversification/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_analysis.py                    # Master script to run entire pipeline
├── .gitignore                         # Git ignore rules
├── config/
│   └── config.py                      # Configuration (tickers, dates, thresholds)
├── src/
│   ├── 01_data_collection.py         # Download stock data from Yahoo Finance
│   ├── 02_data_preparation.py        # Clean and prepare data
│   ├── 03_correlation_analysis.py    # Compute correlations
│   ├── 04_rolling_correlation.py     # Time-based correlation analysis
│   ├── 05_portfolio_analysis.py      # Evaluate diversification benefits
│   └── 06_visualization.py           # Generate all charts
├── data/
│   ├── raw/                           # Downloaded price data (auto-generated)
│   ├── processed/                     # Cleaned returns data (auto-generated)
│   └── results/                       # Correlation matrices (auto-generated)
├── outputs/
│   ├── figures/                       # Visualizations (auto-generated)
│   └── reports/                       # Summary tables (auto-generated)
├── tests/
│   └── test_data_quality.py          # Data validation tests
└── docs/
    └── methodology.md                 # Detailed methodology documentation
```

---



## Output Files

### Processed Data (`data/processed/`)
- `aligned_prices.csv` - Cleaned daily prices for all stocks
- `daily_returns.csv` - Daily simple returns
- `descriptive_statistics.csv` - Summary statistics (mean, volatility, Sharpe ratio, etc.)

### Analysis Results (`data/results/`)
- `correlation_matrix.csv` - Full correlation matrix (all stocks)
- `stock_classification.csv` - Classification by correlation strength
- `top_noncorrelated_stocks.csv` - Top 10 lowest correlation with NVDA
- `rolling_corr_6m.csv` - 6-month rolling correlations
- `rolling_corr_12m.csv` - 12-month rolling correlations
- `correlation_stability_6m.csv` - Stability metrics (6-month)
- `correlation_stability_12m.csv` - Stability metrics (12-month)
- `correlation_consistency.csv` - Consistency analysis
- `portfolio_comparison.csv` - Portfolio scenario comparison
- `diversified_portfolio_metrics.csv` - Diversified portfolio metrics

### Visualizations (`outputs/figures/`)
1. `correlation_heatmap.png` - Full correlation matrix heatmap
2. `nvda_correlations_barplot.png` - NVDA correlations (sorted bar chart)
3. `correlation_distribution.png` - Distribution histogram
4. `rolling_correlation_6m.png` - 6-month rolling correlations
5. `rolling_correlation_12m.png` - 12-month rolling correlations
6. `cumulative_returns_comparison.png` - NVDA vs non-correlated stocks
7. `sector_correlation_breakdown.png` - Correlation by sector (box plot)
8. `portfolio_performance_comparison.png` - NVDA-only vs diversified

---

## Methodology

### Correlation Classification Thresholds

- **Non-correlated:** |r| < 0.25
- **Weakly correlated:** 0.25 ≤ |r| < 0.60
- **Strongly correlated:** |r| ≥ 0.60

### Analytical Methods

1. **Pearson Correlation** - Measures linear relationship between returns
2. **Spearman Correlation** - Rank-based correlation (robustness check)
3. **Rolling Correlation** - 6-month and 12-month windows to assess stability
4. **Portfolio Metrics** - Volatility, Sharpe ratio, maximum drawdown, VaR, CVaR

---

## Contributing

This is a capstone project for academic purposes. If you'd like to suggest improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is created for educational purposes as part of my degree plan, but I have attached the MIT License.

---

## Contact

**Tristan Rast**  
Email: tristanarast [at] gmail [dot] com  
GitHub: [@tristanrast](https://github.com/tristanrast)  
LinkedIn: https://www.linkedin.com/in/tristan-rast/

---

## Acknowledgments

- **WGU Faculty** - Guidance and support
- **Yahoo Finance** - Historical market data
- **Open Source Community** - Python libraries (pandas, yfinance, matplotlib, seaborn)
- **Academic Sources** - Portfolio theory and correlation analysis methodology

---

## References

Final report for complete APA-formatted references include:
- Investopedia articles on correlation and MPT
- Saxo Group guides on diversification
- Research on semiconductor stocks and portfolio construction
- Yahoo Finance API documentation
- Python library documentation

---

**Last Updated:** March 13, 2026
