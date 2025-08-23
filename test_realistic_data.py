#!/usr/bin/env python3
"""
Test VRP system with realistic data that has proper mean reversion characteristics.
"""

import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from vrp import VRPTrader

def main():
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    print("VRP Trading System - Realistic Data Test")
    print("=" * 50)
    
    # Test with realistic VRP data
    trader = VRPTrader()
    
    if not trader.load_data("realistic_vrp_data.csv"):
        print("ERROR: Failed to load realistic data")
        return
    
    print(f"Loaded {len(trader.data)} data points")
    
    # Run backtest
    print("Running backtest on realistic VRP data...")
    results = trader.backtest()
    
    if not results:
        print("ERROR: Backtest failed")
        return
    
    # Display results
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS - REALISTIC VRP DATA")
    print(f"{'='*60}")
    print(f"Total Return:        {results['total_return']:.4f}")
    print(f"Total Trades:        {results['total_trades']}")
    print(f"Win Rate:           {results['win_rate']:.1%}")
    print(f"Winning Trades:     {results['winning_trades']}")
    print(f"Losing Trades:      {results['losing_trades']}")
    print(f"Average Win:        {results['avg_win']:.4f}")
    print(f"Average Loss:       {results['avg_loss']:.4f}")
    print(f"Profit Factor:      {results['profit_factor']:.2f}")
    print(f"Max Drawdown:       {results['max_drawdown']:.4f}")
    print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    
    # Assessment
    print(f"\n{'-'*40}")
    print(f"PERFORMANCE ASSESSMENT")
    print(f"{'-'*40}")
    
    win_rate = results['win_rate']
    sharpe = results['sharpe_ratio']
    max_dd = results['max_drawdown']
    total_return = results['total_return']
    
    # Check benchmarks
    win_rate_ok = win_rate >= 0.45  # >45%
    sharpe_ok = sharpe >= 0.5       # >0.5
    drawdown_ok = max_dd <= 0.10    # <10%
    return_ok = total_return > 0    # Positive
    
    print(f"Win Rate >45%:      {'✓' if win_rate_ok else '✗'} ({win_rate:.1%})")
    print(f"Sharpe >0.5:        {'✓' if sharpe_ok else '✗'} ({sharpe:.2f})")
    print(f"Max DD <10%:        {'✓' if drawdown_ok else '✗'} ({max_dd:.2%})")
    print(f"Positive Return:    {'✓' if return_ok else '✗'} ({total_return:.4f})")
    
    # Overall assessment
    benchmarks_met = sum([win_rate_ok, sharpe_ok, drawdown_ok, return_ok])
    print(f"\nBenchmarks Met:     {benchmarks_met}/4")
    
    if benchmarks_met >= 3:
        print("Status:             ✓ MEETS PRODUCTION CRITERIA")
        return True
    elif benchmarks_met >= 2:
        print("Status:             ⚠ MARGINAL - NEEDS REVIEW")
        return False
    else:
        print("Status:             ✗ BELOW PRODUCTION STANDARDS")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nFINAL RESULT: {'PASS' if success else 'FAIL'}")