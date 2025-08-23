# VRP Trading System

A volatility risk premium (VRP) trading system that uses Markov chain state transitions to generate predictive buy/sell signals from CSV market data.

## 🚀 Quick Start (2 Steps)

### 1. Install Dependencies
```bash
pip install pandas numpy pydantic pydantic-settings
```

### 2. Run the System
```bash
python vrp.py     # Creates sample data and shows example
python cli.py     # Interactive mode
```

## 📊 What is VRP?

**VRP (Volatility Risk Premium)** = Implied Volatility ÷ Realized Volatility

The system uses Markov chain models to predict VRP state transitions:

- **VRP States**: EXTREME_LOW, FAIR_VALUE, NORMAL_PREMIUM, ELEVATED_PREMIUM, EXTREME_HIGH
- **Signal Generation**: Based on predicted state probabilities and confidence scoring
- **Position Sizing**: Risk-adjusted based on model confidence

## 📄 CSV Data Format

Your CSV file needs these columns:
```csv
date,spy_open,spy_high,spy_low,spy_close,spy_volume,vix_close
2024-01-15,480.50,482.75,479.25,481.80,85000000,16.25
2024-01-16,481.90,483.40,480.10,482.60,78000000,15.90
```

## 💻 Simple Usage

### Python API (3 lines)
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

# Interactive mode
python cli.py
```

### DataFrame Input
```python
import pandas as pd
from vrp import VRPTrader

# Load your data however you want
df = pd.read_csv('market_data.csv')
# or df = get_data_from_database()
# or df = download_from_api()

trader = VRPTrader()
trader.load_data(df)  # Pass DataFrame directly
signal = trader.get_signal()
```

## 📈 Example Output

```
Generated predictive signal: SELL_VOL | 
Current: EXTREME_HIGH | 
Predicted: EXTREME_HIGH | 
Confidence: 88.0% | 
Reason: Model predicts 95.3% probability of overvalued VRP states
```

The system provides:
- Current VRP state classification
- Predicted next state using Markov chain transitions
- Model confidence score (0-100%)
- Clear reasoning for signal decisions

## 📈 Advanced Signal Details

```python
# Get detailed signal information
signal_details = trader.get_trading_signal_details()

print(f"Signal: {signal_details.signal_type}")
print(f"Current State: {signal_details.current_state.name}")
print(f"Predicted State: {signal_details.predicted_state.name}")
print(f"Confidence: {signal_details.confidence_score:.1%}")
print(f"Position Size: {signal_details.risk_adjusted_size:.1%}")
print(f"Reasoning: {signal_details.reason}")
```

## 📈 Backtesting

```python
# Backtest entire dataset
results = trader.backtest()

# Backtest specific period
results = trader.backtest(
    start_date='2023-06-01',
    end_date='2023-12-31'
)

print(f"Win Rate: {results['win_rate']:.1%}")
print(f"Total Return: {results['total_return']:.2%}")
```

*Note: Backtest functionality is temporarily disabled pending API updates.*

## 🛠 Getting Your Data

The system is designed to work with CSV files for maximum flexibility and reliability. Here are your data options:

### Option 1: Use the sample generator
```python
from vrp import create_sample_data
create_sample_data('my_data.csv')  # Creates realistic sample data
```

### Option 2: Download data yourself (optional)
If you want to download live data, you can use any data provider:

```python
# Example with any data source
import pandas as pd

# Get data from your preferred source
# (Yahoo Finance, Alpha Vantage, Bloomberg API, etc.)
spy_data = get_spy_data()  # Your data source
vix_data = get_vix_data()  # Your data source

# Format and save
data = pd.DataFrame({
    'date': spy_data.index,
    'spy_open': spy_data['Open'],
    'spy_high': spy_data['High'], 
    'spy_low': spy_data['Low'],
    'spy_close': spy_data['Close'],
    'spy_volume': spy_data['Volume'],
    'vix_close': vix_data['Close']
})
data.to_csv('market_data.csv', index=False)
```

### Option 3: Use your own data source
As long as you have the required columns in CSV format, you can use any data source.

## 📋 Architecture

The system uses predictive modeling with Markov chain state transitions:

```
vrp/
├── vrp.py                           # Main VRP trading system (simple interface)
├── cli.py                           # Command-line interface  
├── services/                        # Core trading services
│   ├── vrp_calculator.py           # VolatilityData generation for Markov chain
│   ├── signal_generator.py         # Predictive signal generation 
│   └── backtest_engine.py          # Strategy backtesting
├── src/
│   ├── services/
│   │   └── markov_chain_model.py   # Transition matrix and state predictions
│   ├── models/                     # Data models and validation
│   │   ├── data_models.py          # Pydantic models with business rules
│   │   ├── constants.py            # Centralized configuration
│   │   └── validators.py           # Reusable validation logic
│   └── interfaces/
│       └── contracts.py            # Service interface definitions
└── README.md                       # This file
```

**Key Features:**
- Markov chain state transition modeling for prediction
- Multi-factor confidence scoring (entropy + data quality + stability)  
- Risk-adjusted position sizing based on model confidence
- Comprehensive error handling and data validation
- Simple 3-line API maintained for ease of use

## 🧠 How Prediction Works

1. **State Classification**: VRP ratios are classified into 5 discrete states
2. **Transition Matrix**: Built from historical state transitions using 60-day rolling windows
3. **Laplace Smoothing**: Applied to handle sparse transition data
4. **State Prediction**: Next state probabilities calculated from current state
5. **Signal Generation**: Signals generated based on predicted state probabilities:
   - **BUY_VOL**: >60% probability of undervalued states
   - **SELL_VOL**: >60% probability of overvalued states  
   - **HOLD**: Balanced or uncertain probabilities

## 🎯 Trading Implementation

When you get a signal, here's how to trade it:

### BUY_VOL Signal
- Buy VIX call options
- Long UVXY, VXX (volatility ETFs)
- Short SVXY, XIV (inverse vol ETFs)

### SELL_VOL Signal
- Sell VIX calls / Buy VIX puts
- Short UVXY, VXX 
- Long SVXY, XIV

## ⚠️ Disclaimer

Educational use only. Do your own research and manage your risk appropriately.

---

**Production-ready predictive VRP trading system** 📈