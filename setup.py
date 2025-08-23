#!/usr/bin/env python3
"""
Simple VRP Trading System Setup

This script helps you get started quickly.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Install required packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pandas", "numpy", "pydantic", "pydantic-settings"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("💡 Try manually: pip install pandas numpy pydantic pydantic-settings")
        return False

def test_installation():
    """Test that everything works"""
    print("🧪 Testing installation...")
    
    try:
        from vrp import VRPTrader
        trader = VRPTrader()
        print("✅ VRP Trader working correctly!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🚀 VRP Trading System Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Run: python cli.py")
    print("2. Or try: python examples/basic_usage.py")
    print("3. Or read: README.md")
    
    print("\n💡 Quick test:")
    print("python -c \"from vrp import VRPTrader; t=VRPTrader()\"")

if __name__ == "__main__":
    main()