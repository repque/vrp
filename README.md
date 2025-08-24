# VRP Trading System

A **production-ready volatility risk premium (VRP) trading system** that uses sophisticated Markov chain modeling to predict market volatility states and generate trading signals. The system features **state persistence**, **confidence-based position management**, and **comprehensive testing** (465 tests) for reliable daily operations.

## Quick Start

### Basic Usage
```bash
pip install pandas numpy pydantic pydantic-settings
python vrp.py --backtest
```

### Daily Production Operations
```bash
# Process today's market data and generate signals
python daily_cli.py process market_data.csv

# Check system status and positions
python daily_cli.py status

# Validate database consistency
python daily_cli.py validate
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

### System Architecture

The system is built with **production-grade engineering practices**:

**Core Components:**
```
src/models/                 # Strongly-typed Pydantic data models
├── data_models.py         # VRPState, TradingSignal, TransitionMatrix
├── constants.py           # System configuration parameters
└── validators.py          # Validation mixins for data integrity

src/services/              # Machine learning and calculation engines  
├── markov_chain_model.py  # 60-day rolling Markov chain with Laplace smoothing
├── vrp_classifier.py      # Adaptive quantile-based state classification
└── volatility_calculator.py # Realized volatility calculations

services/                  # High-level orchestration services
├── signal_generator.py    # Predictive signal generation with confidence scoring
├── vrp_calculator.py      # VRP ratio calculations and state management
└── backtest_engine.py     # Temporal validation and backtesting

src/production/            # Production deployment infrastructure
├── daily_trader.py        # Enhanced trader with state persistence
└── position_manager.py    # Position tracking across trading sessions

src/persistence/           # State management and database layer
├── database.py            # SQLite persistence with ACID guarantees
└── state_manager.py       # Complete system state orchestration
```

**Key Features:**
- **Type Safety**: Strongly-typed Pydantic models throughout
- **Financial Precision**: Decimal arithmetic for accurate calculations  
- **State Persistence**: SQLite database maintains state across daily runs
- **Error Recovery**: Comprehensive error handling and backup mechanisms
- **Testing**: 465 tests covering core functionality, integration, and behavioral scenarios
- **Configuration**: Environment-based configuration with validation
- **Monitoring**: Structured logging and performance monitoring

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

## Advanced Features

### State Persistence for Production
The system maintains complete state across daily operations:

**Persisted State:**
- **Model State**: Markov transition matrices and learning parameters
- **Position Tracking**: Current positions (FLAT/LONG_VOL/SHORT_VOL) and sizes
- **Signal History**: Complete trading signal history with confidence scores  
- **System Metrics**: Processing dates, performance statistics

**Daily Workflow:**
```bash
# Morning: Check system status
python daily_cli.py status

# Process new data
python daily_cli.py process latest_data.csv

# Output: DAILY SIGNAL: BUY_VOL | Confidence: 0.832 | Position: 0.045
```

### Confidence-Based Position Management
Recent enhancement implements **three-tier confidence thresholds**:

- **Entry Threshold**: 0.65 - Minimum confidence to enter new positions
- **Exit Threshold**: 0.40 - Minimum confidence to maintain positions  
- **Flip Threshold**: 0.75 - Minimum confidence to flip positions directly

This reduces whipsaws and improves risk-adjusted returns by staying flat during uncertain periods.

### Testing & Validation
**Comprehensive test suite** with 449 tests:

- **Core Logic Tests**: Markov chains, signal generation, VRP calculations
- **Integration Tests**: End-to-end trading workflows
- **Behavioral Tests**: Flat periods, risk management scenarios
- **Mathematical Property Tests**: Statistical model validation
- **Production Infrastructure Tests**: State persistence, error recovery

Run tests: `python -m pytest tests/ -v`

## Configuration & Environment

### Modern Settings System

The system uses a **unified Pydantic Settings architecture** with nested configuration sections for type safety and validation.

### Environment Variables
```bash
# .env file configuration (with nested support)
VRP_MODEL__VRP_THRESHOLDS=[0.9,1.1,1.3,1.5]
VRP_MODEL__TRANSITION_WINDOW_DAYS=60
VRP_TRADING__BASE_POSITION_SIZE_PCT=0.02
VRP_TRADING__MAX_POSITION_SIZE_PCT=0.05
VRP_DATABASE__DATABASE_PATH=vrp_model.db
VRP_LOGGING__LOG_LEVEL=INFO
```

### Configuration Sections

#### **Model Configuration** (`settings.model`)
- `vrp_thresholds`: VRP state boundary thresholds [0.9, 1.1, 1.3, 1.5]
- `transition_window_days`: Rolling window for Markov chains (60 days)
- `realized_vol_window_days`: Volatility calculation window (30 days)
- `laplace_smoothing_alpha`: Smoothing parameter for sparse data (1.0)
- `min_confidence_for_signal`: Minimum confidence threshold (0.6)
- `volatility_annualization_factor`: Trading days per year (252)

#### **Trading Configuration** (`settings.trading`)
- `base_position_size_pct`: Base position size (2% of portfolio)
- `max_position_size_pct`: Maximum position size (5% of portfolio)
- `extreme_low_confidence_threshold`: BUY_VOL threshold (0.3)
- `extreme_high_confidence_threshold`: SELL_VOL threshold (0.6)
- `transaction_cost_bps`: Transaction costs (10 basis points)

#### **Data Configuration** (`settings.data`)
- `min_data_years`: Minimum data requirement (3 years)
- `preferred_data_years`: Optimal data amount (5 years)
- `max_missing_days_pct`: Maximum missing data tolerance (2%)

#### **Database Configuration** (`settings.database`)
- `database_url`: Connection string
- `database_path`: SQLite file path
- `enable_wal_mode`: Enable WAL mode for concurrency
- `backup_retention_days`: Backup retention period (7 days)

#### **Logging Configuration** (`settings.logging`)
- `log_level`: Logging verbosity
- `log_file_path`: Log file location
- `max_log_file_mb`: Log rotation size

### Programmatic Configuration

```python
from src.config.settings import get_settings, Settings

# Get default settings
settings = get_settings()

# Access nested configuration
vrp_thresholds = settings.model.vrp_thresholds
position_size = settings.trading.base_position_size_pct

# Custom settings for testing
custom_settings = Settings()
custom_settings.model.vrp_thresholds = [0.85, 1.1, 1.3, 1.5]
trader = VRPTrader(settings=custom_settings)
```

## System Properties

### Technical Characteristics
- **Adaptive**: Quantile boundaries adjust to market regimes automatically
- **Predictive**: Uses Markov chains rather than reactive threshold rules
- **Statistically Robust**: Laplace smoothing handles sparse data conditions
- **Temporally Validated**: Prevents forward-looking bias in all calculations
- **Production Ready**: State persistence, error recovery, comprehensive logging

### Performance Expectations
- **Processing Time**: 1-3 seconds for signal generation (252+ days of data)
- **Memory Usage**: 50-100MB typical, 200MB peak with large datasets
- **Database Growth**: ~10KB per day, ~4MB annually
- **Test Coverage**: 449 tests across all system components

## Requirements

**System Requirements:**
- Python 3.8+ 
- Dependencies: `pandas>=1.3.0`, `numpy>=1.21.0`, `pydantic>=2.0.0`, `pydantic-settings>=2.0.0`

**Data Requirements:**  
- CSV format: `date,open,high,low,close,volume,iv`
- Minimum: 90 days of daily data (30 for volatility + 60 for Markov modeling)
- Optimal: 252+ days for statistical robustness

**Installation:**
```bash
git clone [repository]
cd vrp  
pip install -r requirements.txt
python vrp.py --backtest  # Test with sample data
```

## Disclaimer

**This software is for educational and research purposes only.** It is not investment advice and trading involves substantial risk of loss. Past performance does not guarantee future results. Consult qualified financial advisors before making investment decisions. The authors assume no responsibility for trading losses.