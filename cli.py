#!/usr/bin/env python3
"""
VRP Trading CLI

Usage:
    python cli.py                    # Interactive mode
    python cli.py signal data.csv    # Get signal from CSV
    python cli.py backtest data.csv  # Run backtest
"""

import sys
import argparse
import os
import logging
from vrp import VRPTrader, create_sample_data


def interactive_mode():
    """Interactive command-line interface"""
    print("🎯 VRP Trading System")
    print("=" * 30)
    
    trader = VRPTrader()
    data_loaded = False
    
    while True:
        print(f"\nData loaded: {'✅' if data_loaded else '❌'}")
        print("\nWhat would you like to do?")
        print("1. Load CSV data")
        print("2. Get trading signal")
        print("3. Run backtest")
        print("4. Customize thresholds")
        print("5. Create sample data")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                filename = input("Enter CSV filename: ").strip()
                if not filename:
                    filename = "sample_data.csv"
                
                if os.path.exists(filename):
                    data_loaded = trader.load_data(filename)
                    if data_loaded:
                        print(f"✅ Successfully loaded {filename}")
                    else:
                        print(f"❌ Failed to load {filename}")
                else:
                    print(f"❌ File '{filename}' not found")
                    
            elif choice == "2":
                if not data_loaded:
                    print("❌ Please load data first (option 1)")
                    continue
                signal = trader.get_signal()
                if signal:
                    vrp = trader.current_vrp
                    state = trader.current_state
                    print(f"\n🎯 SIGNAL: {signal}")
                    print(f"   VRP Ratio: {vrp:.2f}")
                    print(f"   VRP State: {state.name}")
                    print(f"   Position Size: {trader.position_size * 100}%")
                else:
                    print("❌ Unable to generate signal")
                
            elif choice == "3":
                if not data_loaded:
                    print("❌ Please load data first (option 1)")
                    continue
                    
                start_date = input("Start date (YYYY-MM-DD, or Enter for all): ").strip()
                end_date = input("End date (YYYY-MM-DD, or Enter for all): ").strip()
                
                results = trader.backtest(
                    start_date=start_date if start_date else None,
                    end_date=end_date if end_date else None
                )
                
                if results:
                    print(f"\n📈 Backtest Results:")
                    print(f"   Total Return: {results['total_return']:.2%}")
                    print(f"   Win Rate: {results['win_rate']:.1%}")
                    print(f"   Total Trades: {results['total_trades']}")
                    print(f"   Winning Trades: {results['winning_trades']}")
                    print(f"   Losing Trades: {results['losing_trades']}")
                else:
                    print("❌ Backtest failed")
                
            elif choice == "4":
                print(f"\nCurrent thresholds:")
                print(f"Buy: {trader.buy_threshold}, Sell: {trader.sell_threshold}, Size: {trader.position_size}")
                
                try:
                    buy = input("Buy threshold (default 0.9): ").strip()
                    sell = input("Sell threshold (default 1.4): ").strip()
                    size = input("Position size (default 0.02): ").strip()
                    
                    trader.set_thresholds(
                        buy_threshold=float(buy) if buy else None,
                        sell_threshold=float(sell) if sell else None,
                        position_size=float(size) if size else None
                    )
                except ValueError:
                    print("❌ Invalid input, keeping current settings")
                
            elif choice == "5":
                filename = input("Sample data filename (default: sample_data.csv): ").strip()
                if not filename:
                    filename = "sample_data.csv"
                create_sample_data(filename)
                print(f"✅ Created {filename}. You can now load this file with option 1")
                
            elif choice == "6":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice, try again")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def get_signal(csv_file: str):
    """Get trading signal from CSV file"""
    trader = VRPTrader()
    
    if trader.load_data(csv_file):
        signal = trader.get_signal()
        if signal:
            vrp = trader.current_vrp
            state = trader.current_state
            print(f"\n🎯 SIGNAL: {signal}")
            print(f"   VRP Ratio: {vrp:.2f}")
            print(f"   VRP State: {state.name}")
            print(f"   Position Size: {trader.position_size * 100}%")
        return signal
    return None


def run_backtest(csv_file: str):
    """Run backtest on CSV file"""
    trader = VRPTrader()
    
    if trader.load_data(csv_file):
        results = trader.backtest()
        if results:
            print(f"\n📈 Backtest Results:")
            print(f"   Total Return: {results['total_return']:.2%}")
            print(f"   Win Rate: {results['win_rate']:.1%}")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   Winning Trades: {results['winning_trades']}")
            print(f"   Losing Trades: {results['losing_trades']}")
        return results
    return None


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="VRP Trading System")
    parser.add_argument('command', nargs='?', choices=['signal', 'backtest'], 
                       help='Command to run')
    parser.add_argument('csv_file', nargs='?', 
                       help='CSV file with market data')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'signal':
            if not args.csv_file:
                print("❌ Please specify CSV file: python cli.py signal data.csv")
                return
            get_signal(args.csv_file)
            
        elif args.command == 'backtest':
            if not args.csv_file:
                print("❌ Please specify CSV file: python cli.py backtest data.csv")
                return
            run_backtest(args.csv_file)
            
        else:
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()