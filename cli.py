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

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def interactive_mode():
    """Interactive command-line interface"""
    logger.info("VRP Trading System")
    logger.info("=" * 30)
    
    trader = VRPTrader()
    data_loaded = False
    
    while True:
        logger.info(f"\nData loaded: {'✅' if data_loaded else '❌'}")
        logger.info("\nWhat would you like to do?")
        logger.info("1. Load CSV data")
        logger.info("2. Get trading signal")
        logger.info("3. Run backtest")
        logger.info("4. Customize thresholds")
        logger.info("5. Create sample data")
        logger.info("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                filename = input("Enter CSV filename: ").strip()
                if not filename:
                    filename = "sample_data.csv"
                
                if os.path.exists(filename):
                    data_loaded = trader.load_data(filename)
                    if data_loaded:
                        logger.info(f"Successfully loaded {filename}")
                    else:
                        logger.error(f"Failed to load {filename}")
                else:
                    logger.error(f"File '{filename}' not found")
                    
            elif choice == "2":
                if not data_loaded:
                    logger.error("Please load data first (option 1)")
                    continue
                signal = trader.get_signal()
                if signal:
                    vrp = trader.current_vrp
                    state = trader.current_state
                    logger.info(f"\nSIGNAL: {signal}")
                    logger.info(f"   VRP Ratio: {vrp:.2f}")
                    logger.info(f"   VRP State: {state.name}")
                    logger.info(f"   Position Size: {trader.position_size * 100}%")
                else:
                    logger.error("Unable to generate signal")
                
            elif choice == "3":
                if not data_loaded:
                    logger.error("Please load data first (option 1)")
                    continue
                    
                start_date = input("Start date (YYYY-MM-DD, or Enter for all): ").strip()
                end_date = input("End date (YYYY-MM-DD, or Enter for all): ").strip()
                
                results = trader.backtest(
                    start_date=start_date if start_date else None,
                    end_date=end_date if end_date else None
                )
                
                if results:
                    logger.info(f"\nBacktest Results:")
                    logger.info(f"   Total Return: {results['total_return']:.2%}")
                    logger.info(f"   Win Rate: {results['win_rate']:.1%}")
                    logger.info(f"   Total Trades: {results['total_trades']}")
                    logger.info(f"   Winning Trades: {results['winning_trades']}")
                    logger.info(f"   Losing Trades: {results['losing_trades']}")
                else:
                    logger.error("Backtest failed")
                
            elif choice == "4":
                logger.info(f"\nCurrent thresholds:")
                logger.info(f"Buy: {trader.buy_threshold}, Sell: {trader.sell_threshold}, Size: {trader.position_size}")
                
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
                    logger.error("Invalid input, keeping current settings")
                
            elif choice == "5":
                filename = input("Sample data filename (default: sample_data.csv): ").strip()
                if not filename:
                    filename = "sample_data.csv"
                create_sample_data(filename)
                logger.info(f"Created {filename}. You can now load this file with option 1")
                
            elif choice == "6":
                logger.info("Goodbye!")
                break
                
            else:
                logger.error("Invalid choice, try again")
                
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def get_signal(csv_file: str):
    """Get trading signal from CSV file"""
    trader = VRPTrader()
    
    if trader.load_data(csv_file):
        signal = trader.get_signal()
        if signal:
            vrp = trader.current_vrp
            state = trader.current_state
            logger.info(f"\nSIGNAL: {signal}")
            logger.info(f"   VRP Ratio: {vrp:.2f}")
            logger.info(f"   VRP State: {state.name}")
            logger.info(f"   Position Size: {trader.position_size * 100}%")
        return signal
    return None


def run_backtest(csv_file: str):
    """Run backtest on CSV file"""
    trader = VRPTrader()
    
    if trader.load_data(csv_file):
        results = trader.backtest()
        if results:
            logger.info(f"\nBacktest Results:")
            logger.info(f"   Total Return: {results['total_return']:.2%}")
            logger.info(f"   Win Rate: {results['win_rate']:.1%}")
            logger.info(f"   Total Trades: {results['total_trades']}")
            logger.info(f"   Winning Trades: {results['winning_trades']}")
            logger.info(f"   Losing Trades: {results['losing_trades']}")
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
                logger.error("Please specify CSV file: python cli.py signal data.csv")
                return
            get_signal(args.csv_file)
            
        elif args.command == 'backtest':
            if not args.csv_file:
                logger.error("Please specify CSV file: python cli.py backtest data.csv")
                return
            run_backtest(args.csv_file)
            
        else:
            interactive_mode()
            
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()