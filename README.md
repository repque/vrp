# VRP Trading System

A volatility risk premium (VRP) trading system that generates signals by predicting market volatility states using Markov chain modeling.

## Quick Start

```bash
pip install pandas numpy pydantic pydantic-settings
python vrp.py --backtest
```

## How It Works

### Volatility Risk Premium (VRP)

VRP measures the difference between implied and realized volatility:

```
VRP = Implied Volatility ÷ Realized Volatility
```

When VRP is high, options are expensive relative to actual market movement. When VRP is low, options are cheap relative to actual volatility.

### The Trading Strategy

The system follows a **predictive VRP momentum** approach:

1. **When VRP trending higher**: Model predicts movement toward overvalued states → **Buy volatility** to ride the upward trend
2. **When VRP trending lower**: Model predicts movement toward undervalued states → **Sell volatility** to ride the downward trend
3. **When VRP trend unclear**: Balanced probabilities → Hold position

**Key Insight**: Our Markov model detects VRP **trends and momentum** over 60-day windows. Rather than assuming mean reversion, we trade **with** the predicted VRP direction.

### Trading Timeframes and Execution

**Signal Generation Window:**
- Uses 60-day rolling windows to build transition matrices
- Predicts next-day VRP state probabilities
- Generates daily signals (BUY_VOL, SELL_VOL, HOLD)

**Position Management:**
- **Entry**: Take positions based on daily signal generation
- **Holding Period**: Positions held until signal changes (typically 1-5 days)
- **Exit**: Close position when new signal conflicts with current position
- **No explicit profit targets**: System relies on signal changes for exits

**Data Requirements:**
- **IV Source**: Use at-the-money (ATM) options with 30-day expiration
- **IV Timing**: End-of-day implied volatility closing values
- **Update Frequency**: Daily signal generation after market close
- **Minimum History**: 90 days (30 for volatility calculation + 60 for Markov modeling)

### System Components

#### 1. State Classification

The system categorizes VRP into five states using adaptive quantiles:

- **EXTREME_LOW**: ≤ 10th percentile (severely undervalued)
- **FAIR_VALUE**: 10th-30th percentile (undervalued) 
- **NORMAL_PREMIUM**: 30th-70th percentile (normal range)
- **ELEVATED_PREMIUM**: 70th-90th percentile (overvalued)
- **EXTREME_HIGH**: > 90th percentile (severely overvalued)

The quantile boundaries are calculated from a rolling 252-day window, so they adapt to changing market conditions rather than using fixed thresholds.

#### 2. Markov Chain Prediction

The system builds a transition matrix showing how VRP states change over time:

```
Current State → Next State Probabilities
EXTREME_LOW  → [0.6, 0.3, 0.1, 0.0, 0.0]  (likely stays low or moves to fair)
FAIR_VALUE   → [0.2, 0.4, 0.3, 0.1, 0.0]  (distributed across states)
NORMAL       → [0.0, 0.2, 0.5, 0.2, 0.1]  (tends to stay normal)
ELEVATED     → [0.0, 0.0, 0.2, 0.4, 0.4]  (likely stays high)
EXTREME_HIGH → [0.0, 0.0, 0.0, 0.2, 0.8]  (usually persists)
```

The matrix is updated using a 60-day rolling window with Laplace smoothing for stability.

#### 3. Signal Generation

Based on the predicted state probabilities:

- **BUY_VOL**: If model predicts >60% chance of moving to overvalued states (ride the VRP uptrend)
- **SELL_VOL**: If model predicts >60% chance of moving to undervalued states (ride the VRP downtrend)
- **HOLD**: If probabilities are balanced or uncertain

The system also considers trend reversals - if currently in an extreme state but model predicts low persistence, it generates a reversal signal.

#### 4. Position Sizing

Position sizes are calculated based on:
- Signal strength (probability of directional move)
- Model confidence (entropy-based measure)
- Maximum position limits (risk controls)

## Data Requirements

CSV file with these columns:
```csv
date,open,high,low,close,volume,iv
2024-01-15,480.50,482.75,479.25,481.80,85000000,0.1625
```

- **OHLCV**: Standard price and volume data for underlying asset
- **IV**: Implied volatility as decimal (0.16 = 16%) from ATM 30-day options
- **Frequency**: Daily end-of-day values (no intraday data required)
- **Source**: Use VIX for SPY, or calculate IV from ATM options for other assets
- **Minimum**: 90 days of data (30 for volatility calculation + 60 for modeling)

## Usage

### Python API
```python
from vrp import VRPTrader

trader = VRPTrader()
trader.load_data('data.csv')
signal = trader.get_signal()  # Returns "BUY_VOL", "SELL_VOL", or "HOLD"
```

### Command Line
```bash
python cli.py signal data.csv     # Get current signal
python cli.py backtest data.csv   # Run historical backtest
```

### Example Output
```
Generated predictive signal: BUY_VOL | 
Current: ELEVATED_PREMIUM | 
Predicted: ELEVATED_PREMIUM | 
Confidence: 0.832 | 
Reason: Model predicts 80.0% probability of overvalued VRP states - buy vol to ride uptrend
```

### Example Trade Lifecycle
```
Day 1 (Monday 4:00 PM): System generates BUY_VOL signal (confidence: 0.75)
Day 2 (Tuesday 9:30 AM): Enter long VIX calls or UVXY position 
Day 2 (Tuesday 4:00 PM): System generates HOLD signal - maintain position
Day 3 (Wednesday 4:00 PM): System generates HOLD signal - maintain position  
Day 4 (Thursday 4:00 PM): System generates SELL_VOL signal - exit long position
Day 5 (Friday 9:30 AM): Enter short volatility position (SVXY or VIX puts)

Total hold time: 3 days
Profit/Loss: Determined by volatility movement during hold period
```

## Implementation Details

### Temporal Validation

The system prevents forward-looking bias by ensuring each trading decision only uses data available at that point in time. Every backtest decision is validated to prevent unrealistic results.

### Error Handling

Comprehensive validation covers:
- Data quality checks (missing values, outliers)
- OHLC price relationship validation  
- Sufficient data requirements
- Model convergence monitoring

### Architecture

```
services/
├── vrp_calculator.py      # Calculates realized volatility and VRP ratios
├── signal_generator.py    # Generates signals from Markov predictions
└── backtest_engine.py     # Backtests with temporal validation

src/services/
├── markov_chain_model.py  # Builds transition matrices and predicts states
└── vrp_classifier.py      # Adaptive quantile-based state classification

src/models/
├── data_models.py         # Pydantic data validation models
└── constants.py           # System configuration constants
```

## Trading Implementation

### BUY_VOL Signals
- **Options**: Buy VIX calls (30-45 DTE for time decay management)
- **ETFs**: Long volatility ETFs (UVXY, VXX, VIXY)
- **Inverse ETFs**: Short inverse volatility ETFs (SVXY, XIV if available)
- **Timing**: Enter positions after market close when signal generated

### SELL_VOL Signals  
- **Options**: Sell VIX calls or buy VIX puts (30-45 DTE)
- **ETFs**: Short volatility ETFs (UVXY, VXX) or long inverse ETFs (SVXY)
- **Spreads**: VIX call credit spreads for defined risk
- **Timing**: Enter positions after market close when signal generated

### Position Management Timeline
- **Signal Check**: Daily after market close (4:00 PM ET)
- **Entry Window**: Next trading day market open (9:30 AM ET)
- **Position Review**: Daily basis - hold until signal changes
- **Average Hold Time**: 2-4 trading days based on backtest analysis
- **Exit Trigger**: New conflicting signal or position reaches maximum hold period (7 days)

### Risk Management
- **Position Sizing**: Based on signal confidence (higher confidence = larger position)
- **Maximum Position**: 5% of portfolio per signal (from configuration)
- **Stop Loss**: None - system relies on signal changes for exits
- **Time Decay Protection**: Use 30-45 DTE options to minimize theta impact
- **Correlation Limits**: Maximum 3 concurrent volatility positions

## System Properties

- **Adaptive**: Quantile boundaries adjust to market regimes
- **Predictive**: Uses Markov chains rather than reactive thresholds
- **Validated**: Prevents forward-looking bias in backtests
- **Generic**: Works with any OHLCV+IV data, not limited to specific assets
- **Robust**: Comprehensive error handling and data validation

## Requirements

- Python 3.8+
- pandas, numpy, pydantic
- CSV data with OHLCV+IV columns
- Minimum 90 days of historical data

## Disclaimer

Educational and research use only. Not investment advice. Trading involves substantial risk of loss. Consult financial advisors before making investment decisions.