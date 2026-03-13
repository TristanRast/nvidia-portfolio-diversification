# Methodology Documentation

## NVIDIA Portfolio Diversification Analysis - Detailed Methodology

**Author:** Tristan Rast  
**Date:** March 11, 2026  
**Project:** WGU Data Analytics Capstone (BHN1 Task 3)

---

## 1. Research Design

### 1.1 Framework

This analysis employs **Modern Portfolio Theory (MPT)** as the theoretical foundation for identifying diversification opportunities. MPT posits that investors can reduce portfolio risk by combining assets with low or negative correlations, thereby achieving superior risk-adjusted returns along the efficient frontier.

### 1.2 CRISP-DM Methodology

The project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM):

1. **Business Understanding** - Define concentration risk problem and diversification needs
2. **Data Understanding** - Explore Yahoo Finance historical price data
3. **Data Preparation** - Clean, align, and compute returns
4. **Modeling** - Calculate correlations and rolling windows
5. **Evaluation** - Assess practical significance through portfolio metrics
6. **Deployment** - Deliver actionable results and reproducible pipeline

---

## 2. Data Collection

### 2.1 Data Source

- **Provider:** Yahoo Finance
- **Access Method:** Python `yfinance` library
- **Data Type:** Historical adjusted closing prices
- **Justification:** Yahoo Finance is widely used in academic and practitioner research, provides reliable data for large-cap US stocks, and offers free programmatic access

### 2.2 Stock Universe Selection

**Target Stock:**
- **NVDA** (NVIDIA Corporation) - Selected due to high market capitalization, significant index weight, and concentration risk concerns for investors

**Comparison Universe (27 stocks across 7 sectors):**

| Sector | Stocks | Rationale |
|--------|--------|-----------|
| Technology | AAPL, MSFT, GOOGL, AMD | Includes semiconductor peer (AMD) and major tech companies |
| Healthcare | JNJ, UNH, PFE, ABBV | Defensive sector, historically lower tech correlation |
| Financials | JPM, BAC, GS, V | Interest rate sensitive, different drivers than tech |
| Consumer Staples | PG, KO, WMT, COST | Defensive, stable earnings, low tech correlation |
| Utilities | NEE, DUK, SO, D | Most defensive sector, bond-like characteristics |
| Energy | XOM, CVX, COP | Commodity-driven, historically low tech correlation |
| Real Estate | AMT, PLD, SPG, EQIX | REITs with different market dynamics |

**Selection Criteria:**
- Large-cap stocks (market cap > $50B as of 2026)
- High liquidity (average daily volume > 1M shares)
- Complete 5-year trading history
- Represent diverse economic sectors

### 2.3 Time Period

- **Start Date:** March 11, 2021
- **End Date:** March 11, 2026
- **Duration:** 5 years (approximately 1,260 trading days)

**Justification:**
- Captures multiple market regimes (COVID recovery, inflation surge, rate hikes, AI boom)
- Sufficient observations for statistical significance (n > 1,000)
- Recent enough to reflect current market structure
- Includes NVIDIA's AI-driven rally (2023-2026)

---

## 3. Data Preparation

### 3.1 Data Cleaning

**Steps:**
1. Load raw adjusted closing prices
2. Align all time series to common trading dates
3. Identify missing values (non-trading days, data gaps)
4. Apply forward-fill method for minor gaps (<5 consecutive days)
5. Exclude stocks with >5% missing data (none in this analysis)

**Justification:**
- Forward-fill preserves last known price for short gaps (holidays, exchange closures)
- 95% completeness threshold ensures data quality
- Adjusted closing prices account for splits and dividends

### 3.2 Return Calculation

**Formula (Simple Returns):**
```
r_t = (P_t - P_{t-1}) / P_{t-1}
```

Where:
- `r_t` = return on day t
- `P_t` = adjusted closing price on day t

**Alternative (Log Returns - computed for robustness):**
```
r_t = ln(P_t / P_{t-1})
```

**Justification:**
- Simple returns used for correlation analysis (standard in portfolio theory)
- Log returns computed as robustness check
- Daily frequency captures short-term co-movements

### 3.3 Descriptive Statistics

For each stock, compute:
- **Mean Return** (annualized): `μ = mean(r_t) × 252`
- **Volatility** (annualized): `σ = std(r_t) × √252`
- **Sharpe Ratio** (assuming Rf=0): `SR = μ / σ`
- **Skewness**: Measure of return distribution asymmetry
- **Kurtosis**: Measure of tail risk
- **Total Return**: `(P_end / P_start - 1) × 100`

---

## 4. Correlation Analysis

### 4.1 Pearson Correlation (Primary Metric)

**Formula:**
```
ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
```

Where:
- `Cov(X,Y)` = Covariance between returns of X and Y
- `σ_X`, `σ_Y` = Standard deviations of X and Y returns

**Range:** -1.0 to +1.0
- +1.0 = Perfect positive correlation
- 0 = No linear relationship
- -1.0 = Perfect negative correlation

**Interpretation:**
- Measures linear relationship between return series
- Used in MPT for portfolio optimization
- Standard metric in finance literature

### 4.2 Spearman Rank Correlation (Robustness Check)

**Formula:**
```
ρ_s = 1 - (6 Σd_i²) / (n(n²-1))
```

Where:
- `d_i` = Difference between ranks
- `n` = Number of observations

**Purpose:**
- Non-parametric alternative to Pearson
- Detects monotonic relationships (not just linear)
- Robust to outliers

### 4.3 Statistical Significance (p-values)

**Hypothesis Test:**
- **H₀:** ρ = 0 (no correlation)
- **H₁:** ρ ≠ 0 (correlation exists)
- **Significance level:** α = 0.05

**Interpretation:**
- p < 0.05: Reject H₀, correlation is statistically significant
- p ≥ 0.05: Fail to reject H₀, correlation not significant

### 4.4 Classification Thresholds

Based on correlation magnitude (`|ρ|`):

| Classification | Threshold | Interpretation | Portfolio Implication |
|----------------|-----------|----------------|---------------------|
| **Non-correlated** | \|ρ\| < 0.25 | Weak to no linear relationship | Strong diversifier |
| **Weakly correlated** | 0.25 ≤ \|ρ\| < 0.60 | Moderate relationship | Moderate diversifier |
| **Strongly correlated** | \|ρ\| ≥ 0.60 | Strong relationship | Poor diversifier |

**Justification:**
- Thresholds based on finance literature conventions
- |ρ| < 0.25: Goetzmann & Kumar (2008) threshold for "low correlation"
- |ρ| ≥ 0.60: Commonly used threshold for "high correlation" in MPT

---

## 5. Rolling Correlation Analysis

### 5.1 Purpose

- Assess correlation stability over time
- Identify regime changes
- Validate diversification persistence

### 5.2 Window Sizes

**6-Month Window (126 trading days):**
- Short-term correlation dynamics
- Captures recent market behavior
- More responsive to regime changes

**12-Month Window (252 trading days):**
- Medium-term correlation patterns
- Standard annual window in finance
- Balances responsiveness and stability

### 5.3 Calculation Method

For each stock and each date t:
```
ρ_t = corr(NVDA_returns[t-window:t], Stock_returns[t-window:t])
```

### 5.4 Stability Metrics

**Coefficient of Variation (CV):**
```
CV = σ(ρ_rolling) / |μ(ρ_rolling)|
```

**Interpretation:**
- Low CV → Stable correlation (reliable diversifier)
- High CV → Unstable correlation (regime-dependent)

**Consistency Percentage:**
```
Pct_NonCorr = (# periods where |ρ_t| < 0.25) / Total_periods × 100
```

**Threshold:** Stock is "consistently non-correlated" if Pct_NonCorr ≥ 75%

---

## 6. Portfolio Analysis

### 6.1 Portfolio Construction

**Three Scenarios:**

1. **NVDA Only (Baseline)**
   - 100% NVDA
   - Represents concentrated position risk

2. **NVDA + Strongly Correlated**
   - 50% NVDA + 50% most correlated stock
   - Tests ineffective diversification

3. **NVDA + Non-Correlated**
   - 50% NVDA + 50% least correlated stock
   - Tests effective diversification

4. **Diversified Portfolio**
   - Equal weight: NVDA + top 5 non-correlated stocks
   - Realistic diversification strategy

### 6.2 Portfolio Metrics

**Annual Return:**
```
μ_p = Σ(w_i × μ_i)
```

**Annual Volatility:**
```
σ_p = √(w' Σ w)
```
Where Σ is the covariance matrix

**Sharpe Ratio (Rf=0):**
```
SR_p = μ_p / σ_p
```

**Maximum Drawdown:**
```
MDD = min((V_t - V_max) / V_max)
```

**Value at Risk (95%):**
```
VaR_95 = Percentile(returns, 5%)
```

**Conditional VaR (Expected Shortfall):**
```
CVaR_95 = Mean(returns where returns ≤ VaR_95)
```

### 6.3 Diversification Benefit Calculation

**Volatility Reduction:**
```
Vol_Reduction = (σ_baseline - σ_diversified) / σ_baseline × 100%
```

**Sharpe Improvement:**
```
Sharpe_Improvement = (SR_diversified - SR_baseline) / SR_baseline × 100%
```

---

## 7. Evaluation Methods

### 7.1 Accuracy Metrics

**Correlation Confidence:**
- All correlations computed with n > 1,000 observations
- Standard error: `SE = √((1-ρ²)/(n-2))`
- 95% Confidence interval: `ρ ± 1.96 × SE`

**Statistical Power:**
- With n=1,260, power > 0.99 for detecting |ρ| > 0.10 at α=0.05

### 7.2 Practical Significance

**Criteria:**
1. **Diversification Effect:** Volatility reduction ≥ 10% considered practically significant
2. **Risk-Adjusted Return:** Sharpe ratio improvement ≥ 0.10 considered meaningful
3. **Stability:** Correlation CV < 0.50 indicates stable relationship

### 7.3 Validation

**Out-of-Sample Approach:**
- Primary analysis uses full 5-year period
- Rolling correlation validates stability across subperiods
- Can be extended with train-test split (e.g., 2021-2024 train, 2024-2026 test)

---

## 8. Limitations and Assumptions

### 8.1 Limitations

1. **Historical Data:** Past correlations may not predict future relationships
2. **Market Regime:** Analysis period dominated by specific regimes (AI boom, rate hikes)
3. **Linear Relationships:** Pearson correlation captures only linear dependencies
4. **Static Weights:** Portfolio analysis uses equal weights (no optimization)
5. **Transaction Costs:** Not included in portfolio metrics
6. **Survivorship Bias:** All stocks existed for full period (no delisted stocks)

### 8.2 Assumptions

1. **Stationarity:** Return distributions assumed relatively stable within analysis period
2. **Normality:** Not assumed (robust statistics and non-parametric tests used)
3. **Market Efficiency:** Prices reflect available information
4. **Risk-Free Rate:** Set to zero for Sharpe ratio (simplification)

---

## 9. Tools and Implementation

### 9.1 Software Stack

- **Python 3.8+**: Core language
- **pandas 2.0+**: Data manipulation
- **numpy 1.24+**: Numerical computations
- **yfinance 0.2.28+**: Data retrieval
- **scipy 1.10+**: Statistical functions
- **matplotlib 3.7+ & seaborn 0.12+**: Visualization

### 9.2 Computational Efficiency

- Vectorized operations via NumPy for speed
- Memory-efficient data structures (DataFrames)
- Modular code design for maintainability
- Logging for debugging and audit trail

### 9.3 Reproducibility

- Fixed random seed (42) for any stochastic operations
- Version-controlled code (Git)
- Documented configuration (`config.py`)
- Requirements file with exact library versions

---

## 10. Ethical and Legal Considerations

### 10.1 Data Governance

- Yahoo Finance data used for educational purposes only
- Attribution provided in all documentation
- No proprietary or confidential data used
- Data quality checks documented

### 10.2 Academic Integrity

- All sources properly cited (APA format)
- Original analysis and code
- Third-party libraries acknowledged
- Collaboration limited to instructor guidance

## References

1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
2. Goetzmann, W. N., & Kumar, A. (2008). Equity Portfolio Diversification. *Review of Finance*, 12(3), 433-463.
3. Investopedia. (2022). How is correlation used in modern portfolio theory?
4. Saxo Group. (2024). How correlation impacts diversification: A guide to smarter investing.

---

**Document Version:** 1.1  
**Last Updated:** March 13, 2026
