#!/usr/bin/env python3
"""
Basic VRP Trading Examples

This file shows the simplest ways to use the VRP trading system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vrp_trader import SimpleVRPTrader


def example_1_basic_signal():
    """Example 1: Get a simple trading signal"""
    print("=" * 50)
    print("EXAMPLE 1: Basic Trading Signal")
    print("=" * 50)
    
    # Create trader and get signal
    trader = SimpleVRPTrader()
    trader.get_data(days=100)  # Get 100 days of data
    signal = trader.get_signal()
    
    # Act on the signal
    if signal == "BUY_VOL":
        print("💡 Action: Consider buying volatility (VIX calls, short VIX ETFs)")
    elif signal == "SELL_VOL":
        print("💡 Action: Consider selling volatility (VIX puts, long VIX ETFs)")
    else:
        print("💡 Action: Hold current positions")


def example_2_custom_settings():
    """Example 2: Customize the trading thresholds"""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Custom Settings")
    print("=" * 50)
    
    trader = SimpleVRPTrader()
    
    # Make it more aggressive (trade more often)
    trader.customize_settings(
        low_threshold=0.95,   # Buy when VRP < 0.95 (instead of 0.9)
        high_threshold=1.3,   # Sell when VRP > 1.3 (instead of 1.4)
        position_size=0.01    # Smaller position size (1% instead of 2%)
    )
    
    trader.get_data()
    signal = trader.get_signal()


def example_3_backtest():
    """Example 3: Test the strategy on historical data"""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Strategy Backtest")
    print("=" * 50)
    
    trader = SimpleVRPTrader()
    trader.get_data(days=500)  # Get more data for backtesting
    results = trader.backtest(years=1)
    
    if results:
        print(f"\n📊 Strategy would have made {len(results.get('signals_generated', 0))} trades")
        if results['total_return'] > 0:
            print("🎉 Strategy was profitable!")
        else:
            print("📉 Strategy lost money - consider adjusting thresholds")


def example_4_monitoring():
    """Example 4: Daily monitoring setup"""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Daily Monitoring")
    print("=" * 50)
    
    trader = SimpleVRPTrader()
    trader.get_data()
    
    # Get VRP value directly
    vrp = trader.calculate_vrp()
    if vrp:
        print(f"📅 Today's VRP: {vrp:.2f}")
        
        # Set up alerts
        if vrp < 0.8:
            print("🚨 ALERT: VRP extremely low - strong buy signal!")
        elif vrp > 1.6:
            print("🚨 ALERT: VRP extremely high - strong sell signal!")
        else:
            print("😴 VRP in normal range - no urgent action needed")


def run_all_examples():
    """Run all examples in sequence"""
    try:
        example_1_basic_signal()
        example_2_custom_settings()
        example_3_backtest()
        example_4_monitoring()
        
        print("\n" + "🎯" * 20)
        print("All examples completed!")
        print("🎯" * 20)
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have internet connection for data download")


if __name__ == "__main__":
    run_all_examples()