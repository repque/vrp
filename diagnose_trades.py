#!/usr/bin/env python3
"""
Diagnose VRP trading system to understand trade execution and P&L patterns.
"""

import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from vrp import VRPTrader

def main():
    # Configure logging to show trades
    logging.basicConfig(level=logging.WARNING)
    
    # Initialize trader
    trader = VRPTrader()
    
    if not trader.load_data("sample_data.csv"):
        print("ERROR: Failed to load data")
        return
    
    print(f"Loaded {len(trader.data)} data points")
    
    # Run backtest and get detailed results
    print("\nRunning detailed backtest analysis...")
    results = trader.backtest()
    
    if not results:
        print("ERROR: Backtest failed")
        return
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRADE EXECUTION DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Total Return:     {results['total_return']:.6f}")
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Winning Trades:   {results['winning_trades']}")
    print(f"Losing Trades:    {results['losing_trades']}")
    print(f"Zero P&L Trades:  {results['total_trades'] - results['winning_trades'] - results['losing_trades']}")
    print(f"Win Rate:         {results['win_rate']:.2%}")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    
    # Let's create a simple backtest to see actual trade details
    print(f"\nDIGGING INTO TRADE DETAILS...")
    
    # Manually run a few steps to see what's happening
    from services import VRPCalculator, SignalGenerator, BacktestEngine
    from src.models.constants import DefaultConfiguration
    
    config = DefaultConfiguration()
    calculator = VRPCalculator(config)
    signal_generator = SignalGenerator(config)
    backtest_engine = BacktestEngine(calculator, signal_generator)
    
    # Get volatility data
    volatility_data = calculator.generate_volatility_data(trader.data)
    print(f"Generated {len(volatility_data)} volatility points")
    
    # Look at the first few trading decisions
    print(f"\nFIRST 10 TRADING DECISIONS:")
    print(f"{'Date':<12} {'Signal':<8} {'VRP':<8} {'State':<15} {'Reason'}")
    print("-" * 80)
    
    # Simulate a few trading decisions manually
    for i in range(90, min(100, len(trader.data))):  # Start after warmup period
        try:
            # Get historical data up to this point
            historical_data = trader.data[:i+1]
            vol_data = calculator.generate_volatility_data(historical_data)
            
            if len(vol_data) < 60:
                continue
                
            # Generate signal
            trading_signal = signal_generator.generate_signal(vol_data)
            
            current_vrp = float(vol_data[-1].vrp)
            
            print(f"{trader.data[i].date} {trading_signal.signal_type:<8} {current_vrp:<8.4f} {trading_signal.current_state.name:<15} {trading_signal.reason[:50]}...")
            
        except Exception as e:
            print(f"{trader.data[i].date} ERROR    -------- --------------- {str(e)[:50]}...")
    
    print(f"\nDIAGNOSIS:")
    zero_pnl_trades = results['total_trades'] - results['winning_trades'] - results['losing_trades']
    if zero_pnl_trades > results['total_trades'] * 0.8:
        print(f"⚠  Most trades ({zero_pnl_trades}/{results['total_trades']}) have zero P&L")
        print(f"   This suggests P&L calculation or position tracking issues")
    
    if results['winning_trades'] + results['losing_trades'] < 20:
        print(f"⚠  Very few trades with actual P&L ({results['winning_trades'] + results['losing_trades']})")
        print(f"   This suggests issues with trade execution logic")
    
    if abs(results['total_return']) < 0.01:
        print(f"⚠  Very small total return ({results['total_return']:.6f})")
        print(f"   This might be expected for IV-based P&L or might indicate scaling issues")

if __name__ == "__main__":
    main()