"""
Interface contracts for VRP Markov Chain Trading Model.

This module defines abstract interfaces that establish clear contracts
between components, enabling dependency injection and testability.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from src.models.data_models import (
    BacktestResult,
    ConfidenceMetrics,
    MarketData,
    ModelState,
    PerformanceMetrics,
    TradingSignal,
    TransitionMatrix,
    VolatilityData,
    VRPState,
)


class IDataFetcher(ABC):
    """Interface for fetching market data from external sources."""

    @abstractmethod
    async def fetch_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """
        Fetch historical market data for a symbol.

        Args:
            symbol: Market symbol (e.g., 'SPY', '^VIX')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            List of MarketData objects sorted by date

        Raises:
            DataFetchError: If data retrieval fails
            ValidationError: If data validation fails
        """
        pass

    @abstractmethod
    async def fetch_latest_data(self, symbol: str) -> MarketData:
        """
        Fetch the most recent market data for a symbol.

        Args:
            symbol: Market symbol

        Returns:
            Latest MarketData object

        Raises:
            DataFetchError: If data retrieval fails
        """
        pass

    @abstractmethod
    def validate_data_quality(self, data: List[MarketData]) -> bool:
        """
        Validate the quality of fetched data.

        Args:
            data: List of market data to validate

        Returns:
            True if data quality is acceptable

        Raises:
            DataQualityError: If data quality is unacceptable
        """
        pass


class IVolatilityCalculator(ABC):
    """Interface for volatility calculations."""

    @abstractmethod
    def calculate_realized_volatility(
        self,
        market_data: List[MarketData],
        window_days: int = 30
    ) -> Dict[datetime, float]:
        """
        Calculate rolling realized volatility.

        Args:
            market_data: Historical market data
            window_days: Rolling window size in days

        Returns:
            Dictionary mapping dates to annualized volatility values

        Raises:
            CalculationError: If calculation fails
        """
        pass

    @abstractmethod
    def calculate_vrp_ratio(
        self,
        implied_vol: float,
        realized_vol: float
    ) -> float:
        """
        Calculate volatility risk premium ratio.

        Args:
            implied_vol: Implied volatility (VIX/100)
            realized_vol: Realized volatility (annualized)

        Returns:
            VRP ratio (implied/realized)

        Raises:
            CalculationError: If calculation fails
        """
        pass

    @abstractmethod
    def generate_volatility_data(
        self,
        market_data: List[MarketData]
    ) -> List[VolatilityData]:
        """
        Generate complete volatility analysis for market data.

        Args:
            market_data: Input market data

        Returns:
            List of VolatilityData objects with VRP calculations
        """
        pass


class IVRPClassifier(ABC):
    """Interface for VRP state classification."""

    @abstractmethod
    def classify_vrp_state(self, vrp_ratio: float) -> VRPState:
        """
        Classify VRP ratio into discrete state.

        Args:
            vrp_ratio: Volatility risk premium ratio

        Returns:
            VRP state classification
        """
        pass

    @abstractmethod
    def get_state_boundaries(self) -> List[float]:
        """
        Get the threshold boundaries for state classification.

        Returns:
            List of threshold values
        """
        pass

    @abstractmethod
    def validate_state_transition(
        self,
        from_state: VRPState,
        to_state: VRPState
    ) -> bool:
        """
        Validate if a state transition is reasonable.

        Args:
            from_state: Starting state
            to_state: Ending state

        Returns:
            True if transition is valid
        """
        pass


class IMarkovChainModel(ABC):
    """Interface for Markov chain model operations."""

    @abstractmethod
    def update_transition_matrix(
        self,
        volatility_data: List[VolatilityData],
        window_days: int = 60
    ) -> TransitionMatrix:
        """
        Update transition matrix with new data using rolling window.

        Args:
            volatility_data: Historical volatility data with states
            window_days: Rolling window size for transition counting

        Returns:
            Updated transition matrix
        """
        pass

    @abstractmethod
    def predict_next_state(
        self,
        current_state: VRPState,
        transition_matrix: TransitionMatrix
    ) -> Dict[VRPState, float]:
        """
        Predict next state probabilities.

        Args:
            current_state: Current VRP state
            transition_matrix: Current transition matrix

        Returns:
            Dictionary mapping states to transition probabilities
        """
        pass

    @abstractmethod
    def apply_laplace_smoothing(
        self,
        transition_counts: List[List[int]],
        alpha: float = 1.0
    ) -> List[List[float]]:
        """
        Apply Laplace smoothing to transition counts.

        Args:
            transition_counts: Raw transition count matrix
            alpha: Smoothing parameter

        Returns:
            Smoothed probability matrix
        """
        pass


class IConfidenceCalculator(ABC):
    """Interface for confidence scoring calculations."""

    @abstractmethod
    def calculate_entropy_score(
        self,
        state_probabilities: Dict[VRPState, float]
    ) -> float:
        """
        Calculate entropy-based confidence score.

        Args:
            state_probabilities: Probability distribution over states

        Returns:
            Entropy score (1 = high confidence, 0 = low confidence)
        """
        pass

    @abstractmethod
    def calculate_data_quality_score(
        self,
        observation_count: int,
        min_observations: int = 30
    ) -> float:
        """
        Calculate data quality score based on observation count.

        Args:
            observation_count: Number of observations used
            min_observations: Minimum required observations

        Returns:
            Data quality score (0-1)
        """
        pass

    @abstractmethod
    def calculate_stability_score(
        self,
        current_matrix: TransitionMatrix,
        previous_matrix: Optional[TransitionMatrix]
    ) -> float:
        """
        Calculate model stability score.

        Args:
            current_matrix: Current transition matrix
            previous_matrix: Previous transition matrix

        Returns:
            Stability score (1 = stable, 0 = unstable)
        """
        pass

    @abstractmethod
    def calculate_overall_confidence(
        self,
        entropy_score: float,
        data_quality_score: float,
        stability_score: float
    ) -> ConfidenceMetrics:
        """
        Calculate overall confidence metrics.

        Args:
            entropy_score: Entropy-based confidence
            data_quality_score: Data quality score
            stability_score: Model stability score

        Returns:
            Combined confidence metrics
        """
        pass


class ISignalGenerator(ABC):
    """Interface for trading signal generation."""

    @abstractmethod
    def generate_signal(
        self,
        current_vrp_data: VolatilityData,
        state_probabilities: Dict[VRPState, float],
        confidence_metrics: ConfidenceMetrics
    ) -> TradingSignal:
        """
        Generate trading signal based on VRP state and confidence.

        Args:
            current_vrp_data: Current VRP data
            state_probabilities: Next state probabilities
            confidence_metrics: Model confidence metrics

        Returns:
            Trading signal with position sizing
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        signal_confidence: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size based on confidence and portfolio value.

        Args:
            signal_confidence: Signal confidence level
            portfolio_value: Current portfolio value

        Returns:
            Position size as percentage of portfolio
        """
        pass

    @abstractmethod
    def validate_signal_logic(self, signal: TradingSignal) -> bool:
        """
        Validate that signal follows business logic rules.

        Args:
            signal: Generated trading signal

        Returns:
            True if signal is valid
        """
        pass


class IModelPersistence(ABC):
    """Interface for model state persistence."""

    @abstractmethod
    async def save_model_state(self, model_state: ModelState) -> None:
        """
        Save current model state to persistent storage.

        Args:
            model_state: Complete model state to save

        Raises:
            PersistenceError: If save operation fails
        """
        pass

    @abstractmethod
    async def load_model_state(self) -> Optional[ModelState]:
        """
        Load model state from persistent storage.

        Returns:
            Loaded model state or None if not found

        Raises:
            PersistenceError: If load operation fails
        """
        pass

    @abstractmethod
    async def save_trading_signal(self, signal: TradingSignal) -> None:
        """
        Save trading signal to storage.

        Args:
            signal: Trading signal to save
        """
        pass

    @abstractmethod
    async def get_recent_signals(
        self,
        days: int = 30
    ) -> List[TradingSignal]:
        """
        Get recent trading signals from storage.

        Args:
            days: Number of days to look back

        Returns:
            List of recent trading signals
        """
        pass


class IBacktester(ABC):
    """Interface for model backtesting."""

    @abstractmethod
    def run_walk_forward_test(
        self,
        historical_data: List[VolatilityData],
        retrain_days: int = 20,
        min_train_days: int = 252
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtesting.

        Args:
            historical_data: Historical volatility data
            retrain_days: Days between model retraining
            min_train_days: Minimum training data required

        Returns:
            List of backtest results
        """
        pass

    @abstractmethod
    def calculate_performance_metrics(
        self,
        backtest_results: List[BacktestResult]
    ) -> PerformanceMetrics:
        """
        Calculate aggregated performance metrics.

        Args:
            backtest_results: Individual backtest results

        Returns:
            Aggregated performance metrics
        """
        pass

    @abstractmethod
    def validate_success_criteria(
        self,
        performance: PerformanceMetrics
    ) -> bool:
        """
        Validate if model meets success criteria.

        Args:
            performance: Performance metrics to validate

        Returns:
            True if success criteria are met
        """
        pass


class IMonitoringService(ABC):
    """Interface for model monitoring and health checks."""

    @abstractmethod
    async def perform_health_check(self) -> Dict[str, bool]:
        """
        Perform comprehensive model health check.

        Returns:
            Dictionary of health check results
        """
        pass

    @abstractmethod
    async def check_data_quality(
        self,
        recent_data: List[MarketData]
    ) -> bool:
        """
        Check recent data quality for anomalies.

        Args:
            recent_data: Recent market data to check

        Returns:
            True if data quality is acceptable
        """
        pass

    @abstractmethod
    async def monitor_model_performance(self) -> PerformanceMetrics:
        """
        Monitor recent model performance.

        Returns:
            Recent performance metrics
        """
        pass

    @abstractmethod
    async def send_alert(self, message: str, severity: str = "INFO") -> None:
        """
        Send monitoring alert.

        Args:
            message: Alert message
            severity: Alert severity level
        """
        pass


class IProductionOrchestrator(ABC):
    """Interface for daily production operations."""

    @abstractmethod
    async def run_daily_update(self) -> TradingSignal:
        """
        Run complete daily update cycle.

        Returns:
            Generated trading signal for the day

        Raises:
            OperationError: If daily update fails
        """
        pass

    @abstractmethod
    async def validate_system_health(self) -> bool:
        """
        Validate overall system health before processing.

        Returns:
            True if system is healthy
        """
        pass

    @abstractmethod
    async def process_new_market_data(
        self,
        market_data: MarketData
    ) -> VolatilityData:
        """
        Process new market data through complete pipeline.

        Args:
            market_data: New market data

        Returns:
            Processed volatility data
        """
        pass
