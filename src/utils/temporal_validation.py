"""
Temporal Validation Utilities

This module provides utilities to prevent forward-looking bias in trading systems.
All functions validate that data used for decisions is strictly available at the time
of the decision, preventing unrealistic backtesting results.
"""

import logging
from datetime import date
from typing import Any, List

from src.models.data_models import MarketData, VolatilityData

logger = logging.getLogger(__name__)


class ForwardLookingError(Exception):
    """Raised when forward-looking bias is detected in trading logic."""
    pass


def validate_no_forward_looking(
    decision_date: date,
    data_used: List[Any],
    field_name: str = "date"
) -> bool:
    """
    Validate that no data from future dates is used for current decision.

    Args:
        decision_date: Date when trading decision is made
        data_used: List of data points used in decision
        field_name: Name of date field in data objects

    Returns:
        True if validation passes

    Raises:
        ForwardLookingError: If future data is detected
    """
    future_data = []

    for item in data_used:
        item_date = getattr(item, field_name, None)
        if item_date and item_date > decision_date:
            future_data.append({
                'item_date': item_date,
                'decision_date': decision_date,
                'item': str(item)[:100]  # Truncate for logging
            })

    if future_data:
        error_msg = f"Forward-looking bias detected! Using {len(future_data)} future data points:"
        for fd in future_data[:5]:  # Show first 5 violations
            error_msg += f"\n  - Item date {fd['item_date']} > Decision date {fd['decision_date']}"

        logger.error(error_msg)
        raise ForwardLookingError(error_msg)

    logger.debug(f"✅ No forward-looking bias detected for decision date {decision_date}")
    return True


def validate_trading_decision_temporal_integrity(
    current_date: date,
    market_data: List[MarketData],
    volatility_data: List[VolatilityData]
) -> bool:
    """
    Comprehensive validation for trading decisions to prevent forward-looking bias.

    Args:
        current_date: Date when trading decision is being made
        market_data: Market data used for decision
        volatility_data: Volatility data used for decision

    Returns:
        True if all temporal constraints are satisfied

    Raises:
        ForwardLookingError: If any forward-looking bias is detected
    """
    # Validate market data
    validate_no_forward_looking(current_date, market_data, "date")

    # Validate volatility data
    validate_no_forward_looking(current_date, volatility_data, "date")

    # Additional check: ensure current day data is the latest available
    if market_data:
        latest_market_date = max(d.date for d in market_data)
        if latest_market_date != current_date:
            logger.warning(
                f"Latest market data is {latest_market_date}, decision date is {current_date}")

    if volatility_data:
        latest_vol_date = max(d.date for d in volatility_data)
        if latest_vol_date > current_date:
            raise ForwardLookingError(
                f"Volatility data contains future date {latest_vol_date} > {current_date}"
            )

    logger.info(f"✅ Trading decision temporal integrity validated for {current_date}")
    return True


def create_temporal_safe_window(
    data: List[Any],
    current_date: date,
    window_size: int,
    date_field: str = "date"
) -> List[Any]:
    """
    Create a temporal-safe data window that excludes any future data.

    Args:
        data: Full dataset
        current_date: Current decision date
        window_size: Desired window size
        date_field: Name of date field

    Returns:
        Safe data window with no forward-looking bias
    """
    # Filter out any future data
    safe_data = [
        item for item in data
        if getattr(item, date_field) <= current_date
    ]

    # Sort by date to ensure chronological order
    safe_data.sort(key=lambda x: getattr(x, date_field))

    # Take the last window_size items (most recent historical data)
    window = safe_data[-window_size:] if len(safe_data) >= window_size else safe_data

    logger.debug(f"Created temporal-safe window: {len(window)} items ending {current_date}")
    return window
