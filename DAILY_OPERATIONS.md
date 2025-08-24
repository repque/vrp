# VRP Daily Operations Guide

This guide covers using the VRP trading system for daily production operations with state persistence.

## Quick Start for Daily Operations

### 1. Daily Signal Generation
```bash
# Process today's market data  
python daily_cli.py process market_data.csv

# Process specific date
python daily_cli.py process market_data.csv --date 2024-01-15
```

### 2. Check System Status
```bash
# View current positions and model state
python daily_cli.py status

# Validate database consistency  
python daily_cli.py validate
```

### 3. Backup Management
```bash
# Create backup before important operations
python daily_cli.py backup
```

## State Persistence

The system automatically maintains state across daily runs:

### What's Persisted
- **Model State**: Markov transition matrix and learning parameters
- **Positions**: Current position state (FLAT, LONG_VOL, SHORT_VOL) and size
- **Signal History**: All generated signals with timestamps
- **System State**: Last processed date and operational metrics

### Database Location
- Default: `vrp_model.db` in current directory
- Specify custom path: `--database /path/to/database.db`

## Daily Workflow

### Morning Startup
1. **Check Status**: `python daily_cli.py status`
2. **Validate State**: `python daily_cli.py validate` 
3. **Create Backup**: `python daily_cli.py backup` (optional)

### Process New Data
```bash
python daily_cli.py process latest_market_data.csv
```

**Expected Output:**
```
DAILY SIGNAL: BUY_VOL
Date: 2024-01-15
Current State: ELEVATED_PREMIUM
Predicted State: EXTREME_HIGH  
Confidence: 0.832
Position Size: 0.045
Reason: Model predicts 80.0% probability of overvalued VRP states
```

### Evening Review
1. **Check Final Status**: `python daily_cli.py status`
2. **Review Logs**: Check `logs/vrp_daily.log`

## File Structure

```
vrp/
├── daily_cli.py              # Production CLI interface
├── vrp_model.db              # SQLite state database
├── logs/
│   └── vrp_daily.log         # Daily operation logs
└── src/
    ├── persistence/          # State persistence layer
    │   ├── database.py       # SQLite implementation
    │   └── state_manager.py  # State orchestration
    └── production/           # Production components
        └── daily_trader.py   # Enhanced trader with persistence
```

## Position Management

The system tracks position states across sessions:

### Position States
- **FLAT**: No volatility exposure
- **LONG_VOL**: Long volatility position (expecting VIX to rise)
- **SHORT_VOL**: Short volatility position (expecting VIX to fall)

### Position Transitions
```
Current: FLAT + Signal: BUY_VOL → New: LONG_VOL
Current: LONG_VOL + Signal: SELL_VOL → New: SHORT_VOL  
Current: SHORT_VOL + Signal: HOLD → Keep: SHORT_VOL
```

## Error Handling

### Common Issues

**"Date already processed"**
- System skips dates already processed
- Use `--date` to reprocess specific date

**"No previous model state found"**
- Normal on first run
- System will initialize with defaults

**"State consistency issues"**
- Run `python daily_cli.py validate`
- Check logs for specific problems
- Create backup before fixing

### Recovery Procedures

**Database Corruption:**
1. Stop all operations
2. Restore from backup: `cp vrp_model_backup_*.db vrp_model.db`
3. Validate: `python daily_cli.py validate`

**Missing Data:**
- System will log warnings but continue with available data
- Check data file format matches requirements

## Monitoring

### Log Files
- **Location**: `logs/vrp_daily.log`
- **Rotation**: Append-only (manual cleanup)
- **Levels**: INFO (production), DEBUG (troubleshooting)

### Key Metrics to Monitor
- Signal confidence scores (target: >0.6)
- Position state transitions
- Model learning progress (total_observations)
- Data quality warnings

## Integration with Trading Systems

### Signal Output Format
The CLI outputs structured signals suitable for automated trading:

```bash
# Capture signal in script
SIGNAL=$(python daily_cli.py process data.csv | grep "DAILY SIGNAL:" | cut -d' ' -f3)
if [ "$SIGNAL" = "BUY_VOL" ]; then
    # Execute buy volatility trade
    # ... trading system integration
fi
```

### Position Size Scaling
- `risk_adjusted_size`: Recommended position size (0.0-1.0)
- Scale by account size: `position_value = account_size * risk_adjusted_size`

## Advanced Usage

### Python API
```python
import asyncio
from src.production.daily_trader import run_daily_processing
from datetime import date

# Programmatic daily processing
signal = await run_daily_processing(
    target_date=date.today(),
    data_source='market_data.csv',
    database_path='vrp_model.db'
)

if signal:
    print(f"Signal: {signal.signal_type}, Confidence: {signal.confidence_score}")
```

### Custom Configuration
```python
from src.config.settings import get_settings
from src.production.daily_trader import DailyVRPTrader

# Custom settings
settings = get_settings()
settings.trading.base_position_size_pct = 0.03  # 3% base position

trader = DailyVRPTrader(settings=settings)
```

## Troubleshooting

### Debug Mode
```bash
python daily_cli.py process data.csv --debug
```

### State Inspection
```python
from src.persistence.state_manager import StateManager

state_manager = StateManager("vrp_model.db")
system_state = await state_manager.load_complete_state()
print(f"Last processed: {system_state.last_processed_date}")
print(f"Position state: {system_state.current_position_state}")
```

### Manual State Reset (Emergency)
```bash
# Backup first!
cp vrp_model.db vrp_model_emergency_backup.db

# Remove corrupted database (will reinitialize)
rm vrp_model.db
python daily_cli.py validate  # Creates new database
```

## Performance Expectations

### Processing Times
- **Data Loading**: <1 second for daily CSV
- **Signal Generation**: 1-3 seconds for 252+ days of data
- **State Persistence**: <1 second for complete state save

### Memory Usage
- **Typical**: 50-100MB during processing
- **Peak**: 200MB with large datasets (2+ years daily data)

### Database Growth
- **Daily Growth**: ~10KB per day (signal + state updates)
- **Annual Size**: ~4MB for full year of operations