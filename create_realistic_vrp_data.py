#!/usr/bin/env python3
"""
Create more realistic VRP sample data with proper mean reversion characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_realistic_vrp_data(filename: str = "realistic_vrp_data.csv", days: int = 365):
    """
    Create realistic VRP data with proper mean reversion patterns.
    
    VRP typically:
    - Has mean around 0.8-1.2 
    - Shows mean reversion (extreme values tend to revert)
    - Has periods of elevated volatility premium during market stress
    - Has correlation between IV spikes and negative market returns
    """
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    data = []
    
    # Base parameters
    base_price = 420.0
    base_rv = 15.0  # Base realized volatility
    vrp_mean = 1.0  # Long-term VRP mean
    
    current_price = base_price
    current_rv = base_rv
    current_vrp = vrp_mean
    
    for i, date in enumerate(dates):
        # Create market regime periods
        regime_cycle = np.sin(2 * np.pi * i / 120) * 0.3  # 120-day cycle
        stress_event = 1.0 if np.random.random() < 0.02 else 0.0  # 2% chance of stress event
        
        # Generate realistic daily return with occasional stress
        if stress_event:
            daily_return = np.random.normal(-0.03, 0.04)  # Stress: -3% mean, 4% vol
            iv_multiplier = 1.5 + np.random.uniform(0.5, 1.0)  # IV spikes 50-100%
        else:
            daily_return = np.random.normal(0.0005 + regime_cycle * 0.01, 0.015)  # Normal: slight up bias
            iv_multiplier = 1.0 + np.random.normal(0, 0.1)  # Normal IV variation
        
        # Update price
        current_price = current_price * (1 + daily_return)
        
        # Update realized volatility (trailing)
        if i < 20:
            current_rv = base_rv + np.random.normal(0, 2)
        else:
            # Calculate actual RV from recent returns (simplified)
            recent_vol = abs(daily_return) * np.sqrt(252) * 100
            current_rv = 0.9 * current_rv + 0.1 * recent_vol  # Smooth update
        
        # Calculate IV with mean reversion and stress response
        iv_base = current_rv * vrp_mean  # Base IV from RV
        iv_stress = current_rv * iv_multiplier  # Stress-adjusted IV
        
        # VRP mean reversion: extreme values revert toward mean
        vrp_deviation = current_vrp - vrp_mean
        mean_reversion_force = -0.1 * vrp_deviation + np.random.normal(0, 0.05)
        
        # Update VRP with mean reversion
        target_vrp = vrp_mean + regime_cycle * 0.2 + stress_event * 0.4
        current_vrp = 0.95 * current_vrp + 0.05 * target_vrp + mean_reversion_force
        current_vrp = max(0.3, min(2.0, current_vrp))  # Bound VRP
        
        # Calculate IV from VRP and RV
        current_iv = current_rv * current_vrp
        current_iv = max(5.0, min(80.0, current_iv))  # Reasonable IV bounds
        
        # Create OHLC
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        close_price = current_price
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(np.random.uniform(80_000_000, 150_000_000)),
            'iv': round(current_iv, 2)
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    # Print VRP statistics for verification
    vrp_values = df['iv'] / 15.0  # Approximate VRP (using base RV)
    print(f"Created {filename}")
    print(f"VRP Statistics:")
    print(f"  Mean: {vrp_values.mean():.2f}")
    print(f"  Std:  {vrp_values.std():.2f}")
    print(f"  Min:  {vrp_values.min():.2f}")
    print(f"  Max:  {vrp_values.max():.2f}")
    print(f"  <0.8: {(vrp_values < 0.8).sum()} days ({(vrp_values < 0.8).mean():.1%})")
    print(f"  >1.2: {(vrp_values > 1.2).sum()} days ({(vrp_values > 1.2).mean():.1%})")
    
    return filename

if __name__ == "__main__":
    create_realistic_vrp_data()