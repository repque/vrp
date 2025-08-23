# VRP Trading System

A volatility risk premium (VRP) trading system that uses adaptive Markov chain state transitions to generate predictive trading signals. The system processes CSV market data and provides buy/sell/hold recommendations based on quantile-based VRP state classification.

## Quick Start

### Install Dependencies
```bash
pip install pandas numpy pydantic pydantic-settings
```

### Run the System
```bash
# Generate sample data and show example signal
python vrp.py

# Run backtest
python vrp.py --backtest

# Interactive CLI
python cli.py
```

## What is VRP?

**VRP (Volatility Risk Premium)** = Implied Volatility ÷ Realized Volatility

The system uses adaptive quantile-based classification to categorize VRP states:
- **EXTREME_LOW**: ≤ 10th percentile (undervalued volatility)
- **FAIR_VALUE**: 10th-30th percentile (slightly undervalued)
- **NORMAL_PREMIUM**: 30th-70th percentile (normal range)
- **ELEVATED_PREMIUM**: 70th-90th percentile (overvalued)
- **EXTREME_HIGH**: > 90th percentile (extremely overvalued)

## CSV Data Format

Required columns:
```csv
date,open,high,low,close,volume,iv
2024-01-15,480.50,482.75,479.25,481.80,85000000,0.1625
2024-01-16,481.90,483.40,480.10,482.60,78000000,0.1590
```

Note: The system uses generic OHLCV+IV format (not asset-specific). IV should be provided as decimal (0.16 = 16%).

## Basic Usage

### Python API
```python
from vrp import VRPTrader

trader = VRPTrader()
trader.load_data('your_data.csv')
signal = trader.get_signal()  # Returns "BUY_VOL", "SELL_VOL", or "HOLD"
```

### Command Line
```bash
# Get current signal
python cli.py signal data.csv

# Run backtest
python cli.py backtest data.csv
```

### DataFrame Input
```python
import pandas as pd
from vrp import VRPTrader

df = pd.read_csv('market_data.csv')
trader = VRPTrader()
trader.load_data(df)
signal = trader.get_signal()
```

## Example Output

```
Generated predictive signal: BUY_VOL | 
Current: ELEVATED_PREMIUM | 
Predicted: ELEVATED_PREMIUM | 
Confidence: 0.832 | 
Reason: Model predicts 80.0% probability of overvalued VRP states - buy for mean reversion
```

The system provides:
- Current VRP state classification using adaptive quantiles
- Predicted next state using Markov chain transitions
- Model confidence score (0.0-1.0)
- Clear reasoning for signal decisions

## Detailed Signal Information

```python
signal_details = trader.get_trading_signal_details()

print(f"Signal: {signal_details.signal_type}")
print(f"Current State: {signal_details.current_state.name}")
print(f"Predicted State: {signal_details.predicted_state.name}")
print(f"Confidence: {signal_details.confidence_score:.3f}")
print(f"Position Size: {signal_details.risk_adjusted_size:.1%}")
print(f"Reasoning: {signal_details.reason}")
```

## Backtesting

```python
# Run backtest on entire dataset
results = trader.backtest()

print(f"Win Rate: {results['win_rate']:.1%}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Architecture

```
vrp/
├── vrp.py                           # Main trading system entry point
├── cli.py                           # Command-line interface
├── services/                        # Core trading services
│   ├── vrp_calculator.py           # VRP calculation and volatility data generation
│   ├── signal_generator.py         # Predictive signal generation
│   └── backtest_engine.py          # Strategy backtesting with temporal validation
├── src/
│   ├── services/
│   │   ├── markov_chain_model.py   # Transition matrix and state predictions
│   │   └── vrp_classifier.py       # Adaptive quantile-based state classification
│   ├── models/                     # Data models and validation
│   │   ├── data_models.py          # Pydantic models with business rules
│   │   ├── constants.py            # System constants and configuration
│   │   └── validators.py           # Reusable validation logic
│   ├── interfaces/
│   │   └── contracts.py            # Service interface definitions
│   └── utils/
│       ├── exceptions.py           # Custom exception classes
│       └── temporal_validation.py  # Forward-looking bias prevention
└── analysis/
    └── signal_quality_analyzer.py  # Prediction accuracy analysis tools
```

## How the System Works

### 1. Adaptive State Classification
- Uses rolling 252-day window to calculate quantile boundaries
- Boundaries adapt to changing market conditions
- No fixed thresholds - fully data-driven classification

### 2. Markov Chain Modeling
- Builds transition matrix from 60-day rolling windows
- Applies Laplace smoothing for robust predictions
- Calculates next-state probabilities from current state

### 3. Signal Generation
- **BUY_VOL**: Model predicts movement to overvalued states (profit from mean reversion)
- **SELL_VOL**: Model predicts movement to undervalued states (profit from mean reversion)
- **HOLD**: Balanced or uncertain state probabilities

### 4. Confidence Scoring
- Entropy-based prediction confidence
- Data quality assessment
- Model stability metrics
- Combined into overall confidence score

### 5. Risk Management
- Position sizing based on signal strength and confidence
- Maximum position limits
- Drawdown monitoring

## Performance Metrics

The system tracks comprehensive performance metrics:
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of profits to losses
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Return**: Mean return per trade
- **Prediction Accuracy**: State transition prediction hit rate

## Key Features

- **Threshold-Free Operation**: Uses adaptive quantile-based classification
- **Forward-Looking Bias Prevention**: Temporal validation ensures realistic backtests
- **High Prediction Accuracy**: State transition prediction with confidence scoring
- **Risk-Adjusted Sizing**: Position sizes based on model confidence
- **Comprehensive Validation**: Extensive data validation and error handling
- **Asset Agnostic**: Works with any OHLCV+IV data (not limited to SPY/VIX)

## Trading Implementation

### BUY_VOL Signal
- Buy VIX call options
- Long volatility ETFs (UVXY, VXX)
- Short inverse volatility ETFs (SVXY)

### SELL_VOL Signal
- Sell VIX calls / Buy VIX puts
- Short volatility ETFs (UVXY, VXX)
- Long inverse volatility ETFs (SVXY)

### HOLD Signal
- No position changes
- Wait for clearer directional signals

## System Requirements

- Python 3.8+
- pandas, numpy, pydantic
- Minimum 90 days of historical data (30 for volatility calculation + 60 for Markov modeling)
- CSV format with OHLCV+IV columns

## Disclaimer

This software is for educational and research purposes only. It does not constitute investment advice. Trading involves substantial risk of loss. Users should conduct their own research and consult with financial advisors before making investment decisions.