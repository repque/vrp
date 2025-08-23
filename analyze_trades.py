#!/usr/bin/env python3
"""
Analyze VRP trading logic and recent trades to identify performance issues.
"""

import logging
import sys
import os
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from vrp import VRPTrader, create_sample_data

def analyze_signal_logic():
    """Analyze the signal generation logic."""
    print("VRP MEAN REVERSION STRATEGY ANALYSIS")
    print("=" * 50)
    
    print("\nVRP MEAN REVERSION THEORY:")
    print("- VRP = IV / RV (Implied Vol / Realized Vol)")
    print("- When VRP is HIGH (IV > RV): Volatility is OVERPRICED → SELL volatility")
    print("- When VRP is LOW (IV < RV): Volatility is UNDERPRICED → BUY volatility")
    print("- Mean reversion: extreme values tend to revert toward average")
    
    print("\nCURRENT SIGNAL LOGIC:")
    print("From signal_generator.py:")
    print("- When VRP will be LOW (undervalued): SELL_VOL")
    print("- When VRP will be HIGH (overvalued): BUY_VOL")
    print("- This appears CORRECT for mean reversion")
    
    print("\nLet's verify with actual data...")

def main():
    # Configure logging to show more detail
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    analyze_signal_logic()
    
    # Initialize trader
    trader = VRPTrader()
    
    # Use existing sample data
    if not trader.load_data("sample_data.csv"):
        print("ERROR: Failed to load data")
        return
    
    print(f"\nLoaded {len(trader.data)} data points")
    
    # Get the latest signal with details
    print("\nGETTING CURRENT SIGNAL DETAILS:")
    print("-" * 40)
    
    signal_details = trader.get_trading_signal_details()
    if signal_details:
        print(f"Signal: {signal_details.signal_type}")
        print(f"Current State: {signal_details.current_state.name}")
        print(f"Predicted State: {signal_details.predicted_state.name}")
        print(f"Confidence: {signal_details.confidence_score:.3f}")
        print(f"Signal Strength: {signal_details.signal_strength}")
        print(f"Reason: {signal_details.reason}")
    
    # Generate some historical VRP data to analyze patterns
    try:
        volatility_data = trader._generate_volatility_data()
        print(f"\nVOLATILITY DATA ANALYSIS:")
        print(f"Generated {len(volatility_data)} volatility points")
        
        # Show recent VRP values and states
        print("\nRECENT VRP VALUES:")
        for i, vdata in enumerate(volatility_data[-10:]):
            print(f"{vdata.date}: VRP={float(vdata.vrp):.4f}, State={vdata.vrp_state.name}")
        
        # Analyze state distribution
        state_counts = {}
        for vdata in volatility_data:
            state = vdata.vrp_state.name
            state_counts[state] = state_counts.get(state, 0) + 1
        
        print(f"\nSTATE DISTRIBUTION:")
        for state, count in state_counts.items():
            pct = (count / len(volatility_data)) * 100
            print(f"{state}: {count} ({pct:.1f}%)")
            
    except Exception as e:
        print(f"Error analyzing volatility data: {e}")
    
    # Run a quick backtest with more details
    print(f"\nRUNNING QUICK BACKTEST ANALYSIS...")
    results = trader.backtest()
    
    if results:
        print(f"Total Return: {results['total_return']:.4f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        
        # Check if the issue is with trade execution or signal logic
        if results['total_trades'] > 0:
            avg_trade = results['total_return'] / results['total_trades']
            print(f"Average P&L per trade: {avg_trade:.6f}")
            
            if abs(avg_trade) < 0.001:
                print("WARNING: Very small P&L per trade - possible calculation issue")
        
        print("\nPOSSIBLE ISSUES:")
        if results['win_rate'] < 0.3:
            print("- Win rate is very low (<30%) - signal logic may be inverted")
        if abs(results['total_return']) < 0.01:
            print("- Total return is very small - P&L calculation may be wrong")
        if results['total_trades'] > 100:
            print("- Very frequent trading - signal may be too noisy")

if __name__ == "__main__":
    main()