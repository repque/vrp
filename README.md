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

The system follows a mean reversion approach:

1. **When VRP is high (overvalued volatility)**: Expect volatility to decrease → Buy volatility to profit when it reverts down
2. **When VRP is low (undervalued volatility)**: Expect volatility to increase → Sell volatility to profit when it reverts up
3. **When VRP is normal**: No clear directional edge → Hold position

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

- **BUY_VOL**: If model predicts >60% chance of moving to overvalued states
- **SELL_VOL**: If model predicts >60% chance of moving to undervalued states  
- **HOLD**: If probabilities are balanced or uncertain

The system also considers mean reversion patterns - if currently in an extreme state but model predicts low persistence, it generates a contrarian signal.

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

- **OHLCV**: Standard price and volume data
- **IV**: Implied volatility as decimal (0.16 = 16%)
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
Reason: Model predicts 80.0% probability of overvalued VRP states - buy for mean reversion
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
- Buy VIX call options
- Long volatility ETFs (UVXY, VXX)  
- Short inverse volatility ETFs (SVXY)

### SELL_VOL Signals  
- Sell VIX calls or buy VIX puts
- Short volatility ETFs
- Long inverse volatility ETFs

### Risk Management
- Position sizing based on signal confidence
- Maximum position limits
- Drawdown monitoring and alerts

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