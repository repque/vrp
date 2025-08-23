"""
Signal Quality Analysis Tool

Analyzes the accuracy of Markov chain predictions vs actual state transitions
to identify root causes of poor win rates in the VRP trading system.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vrp_calculator import VRPCalculator
from services.signal_generator import SignalGenerator
from src.services.markov_chain_model import MarkovChainModel
from src.models.data_models import MarketData, VRPState, TradingSignal, VolatilityData
from src.utils.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


class SignalQualityAnalyzer:
    """
    Analyzes prediction accuracy and signal quality for the VRP trading system.
    
    Provides detailed metrics on:
    - State prediction accuracy
    - Signal timing effectiveness
    - Confidence score reliability
    - Transition pattern analysis
    """
    
    def __init__(self):
        """Initialize signal quality analyzer."""
        self.vrp_calculator = VRPCalculator()
        self.signal_generator = SignalGenerator()
        self.markov_model = MarkovChainModel()
        
    def analyze_prediction_accuracy(
        self, 
        data: List[MarketData],
        window_days: int = 60
    ) -> Dict:
        """
        Analyze prediction accuracy vs actual state transitions.
        
        Args:
            data: Historical market data
            window_days: Rolling window for analysis
            
        Returns:
            Dictionary with detailed accuracy metrics
        """
        if len(data) < window_days + 30:  # Need buffer for volatility calc
            raise InsufficientDataError(
                required=window_days + 30,
                available=len(data)
            )
        
        # Generate volatility data
        volatility_data = self.vrp_calculator.generate_volatility_data(data)
        
        if len(volatility_data) < window_days:
            raise InsufficientDataError(
                required=window_days,
                available=len(volatility_data)
            )
        
        # Track predictions vs actual outcomes
        predictions = []
        actual_outcomes = []
        confidence_scores = []
        signal_types = []
        dates = []
        
        # Analyze each day where we can make predictions
        for i in range(window_days, len(volatility_data) - 1):
            try:
                # Get current slice for prediction
                current_slice = volatility_data[:i+1]
                
                # Generate signal and prediction
                signal = self.signal_generator.generate_signal(current_slice, window_days)
                
                # Get actual next state
                actual_next_state = volatility_data[i + 1].vrp_state
                
                # Record prediction vs reality
                predictions.append(signal.predicted_state)
                actual_outcomes.append(actual_next_state)
                confidence_scores.append(float(signal.confidence_score))
                signal_types.append(signal.signal_type)
                dates.append(signal.date)
                
            except Exception as e:
                logger.warning(f"Skipping analysis for day {i}: {str(e)}")
                continue
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(
            predictions, actual_outcomes, confidence_scores, signal_types
        )
        
        # Add temporal analysis
        accuracy_metrics['temporal_analysis'] = self._analyze_temporal_patterns(
            dates, predictions, actual_outcomes, confidence_scores
        )
        
        # Add state transition analysis
        accuracy_metrics['transition_analysis'] = self._analyze_transition_patterns(
            predictions, actual_outcomes
        )
        
        return accuracy_metrics
    
    def _calculate_accuracy_metrics(
        self,
        predictions: List[VRPState],
        actual_outcomes: List[VRPState],
        confidence_scores: List[float],
        signal_types: List[str]
    ) -> Dict:
        """Calculate detailed accuracy metrics."""
        
        total_predictions = len(predictions)
        if total_predictions == 0:
            return {"error": "No predictions to analyze"}
        
        # Overall accuracy
        correct_predictions = sum(1 for p, a in zip(predictions, actual_outcomes) if p == a)
        overall_accuracy = correct_predictions / total_predictions
        
        # Accuracy by confidence level
        confidence_buckets = {
            'high': (0.7, 1.0),
            'medium': (0.5, 0.7),
            'low': (0.0, 0.5)
        }
        
        accuracy_by_confidence = {}
        for bucket_name, (min_conf, max_conf) in confidence_buckets.items():
            indices = [i for i, conf in enumerate(confidence_scores) 
                      if min_conf <= conf <= max_conf]
            
            if indices:
                bucket_correct = sum(1 for i in indices 
                                   if predictions[i] == actual_outcomes[i])
                bucket_accuracy = bucket_correct / len(indices)
                accuracy_by_confidence[bucket_name] = {
                    'accuracy': bucket_accuracy,
                    'count': len(indices),
                    'avg_confidence': sum(confidence_scores[i] for i in indices) / len(indices)
                }
            else:
                accuracy_by_confidence[bucket_name] = {
                    'accuracy': 0.0,
                    'count': 0,
                    'avg_confidence': 0.0
                }
        
        # Accuracy by signal type
        accuracy_by_signal = {}
        for signal_type in set(signal_types):
            indices = [i for i, s in enumerate(signal_types) if s == signal_type]
            if indices:
                signal_correct = sum(1 for i in indices 
                                   if predictions[i] == actual_outcomes[i])
                accuracy_by_signal[signal_type] = {
                    'accuracy': signal_correct / len(indices),
                    'count': len(indices),
                    'avg_confidence': sum(confidence_scores[i] for i in indices) / len(indices)
                }
        
        # Accuracy by state
        accuracy_by_state = {}
        for state in VRPState:
            indices = [i for i, p in enumerate(predictions) if p == state]
            if indices:
                state_correct = sum(1 for i in indices 
                                  if predictions[i] == actual_outcomes[i])
                accuracy_by_state[state.name] = {
                    'accuracy': state_correct / len(indices),
                    'count': len(indices)
                }
        
        # Directional accuracy (up, down, same)
        directional_accuracy = self._calculate_directional_accuracy(
            predictions, actual_outcomes
        )
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy_by_confidence': accuracy_by_confidence,
            'accuracy_by_signal_type': accuracy_by_signal,
            'accuracy_by_predicted_state': accuracy_by_state,
            'directional_accuracy': directional_accuracy,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        }
    
    def _calculate_directional_accuracy(
        self,
        predictions: List[VRPState],
        actual_outcomes: List[VRPState]
    ) -> Dict:
        """Calculate directional accuracy (whether movement direction was correct)."""
        
        # Map states to ordinal values
        state_order = {
            VRPState.EXTREME_LOW: 0,
            VRPState.FAIR_VALUE: 1,
            VRPState.NORMAL_PREMIUM: 2,
            VRPState.ELEVATED_PREMIUM: 3,
            VRPState.EXTREME_HIGH: 4
        }
        
        directional_correct = 0
        directional_total = 0
        
        for i in range(len(predictions) - 1):
            current_state = predictions[i]
            pred_next_state = predictions[i]
            actual_next_state = actual_outcomes[i]
            
            # Skip first prediction (no previous state)
            if i == 0:
                continue
                
            current_ordinal = state_order[current_state]
            pred_ordinal = state_order[pred_next_state]
            actual_ordinal = state_order[actual_next_state]
            
            pred_direction = 0 if pred_ordinal == current_ordinal else (1 if pred_ordinal > current_ordinal else -1)
            actual_direction = 0 if actual_ordinal == current_ordinal else (1 if actual_ordinal > current_ordinal else -1)
            
            if pred_direction == actual_direction:
                directional_correct += 1
            directional_total += 1
        
        directional_accuracy = directional_correct / directional_total if directional_total > 0 else 0
        
        return {
            'directional_accuracy': directional_accuracy,
            'directional_correct': directional_correct,
            'directional_total': directional_total
        }
    
    def _analyze_temporal_patterns(
        self,
        dates: List,
        predictions: List[VRPState],
        actual_outcomes: List[VRPState],
        confidence_scores: List[float]
    ) -> Dict:
        """Analyze temporal patterns in prediction accuracy."""
        
        # Group by time periods
        df = pd.DataFrame({
            'date': dates,
            'prediction': predictions,
            'actual': actual_outcomes,
            'confidence': confidence_scores,
            'correct': [p == a for p, a in zip(predictions, actual_outcomes)]
        })
        
        # Convert dates for analysis
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Monthly accuracy
        monthly_accuracy = df.groupby('month')['correct'].agg(['mean', 'count']).to_dict('index')
        
        # Day of week accuracy
        dow_accuracy = df.groupby('day_of_week')['correct'].agg(['mean', 'count']).to_dict('index')
        
        return {
            'monthly_accuracy': monthly_accuracy,
            'day_of_week_accuracy': dow_accuracy,
            'total_days_analyzed': len(df)
        }
    
    def _analyze_transition_patterns(
        self,
        predictions: List[VRPState],
        actual_outcomes: List[VRPState]
    ) -> Dict:
        """Analyze state transition prediction patterns."""
        
        # Count prediction vs actual transition pairs
        transition_matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, actual in zip(predictions, actual_outcomes):
            transition_matrix[pred.name][actual.name] += 1
        
        # Calculate most common prediction errors
        error_patterns = []
        for pred_state, actual_dict in transition_matrix.items():
            for actual_state, count in actual_dict.items():
                if pred_state != actual_state:  # Only errors
                    error_patterns.append({
                        'predicted': pred_state,
                        'actual': actual_state,
                        'count': count
                    })
        
        # Sort by frequency
        error_patterns.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'transition_matrix': dict(transition_matrix),
            'top_prediction_errors': error_patterns[:10],
            'most_accurate_predictions': self._find_most_accurate_predictions(transition_matrix)
        }
    
    def _find_most_accurate_predictions(self, transition_matrix: Dict) -> List[Dict]:
        """Find states that are predicted most accurately."""
        
        accurate_predictions = []
        
        for pred_state, actual_dict in transition_matrix.items():
            total_predictions = sum(actual_dict.values())
            correct_predictions = actual_dict.get(pred_state, 0)
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                accurate_predictions.append({
                    'state': pred_state,
                    'accuracy': accuracy,
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions
                })
        
        accurate_predictions.sort(key=lambda x: x['accuracy'], reverse=True)
        return accurate_predictions
    
    def generate_quality_report(self, data: List[MarketData]) -> str:
        """Generate a comprehensive signal quality report."""
        
        try:
            analysis = self.analyze_prediction_accuracy(data)
            
            report = [
                "=" * 80,
                "VRP SIGNAL QUALITY ANALYSIS REPORT",
                "=" * 80,
                "",
                f"Overall Prediction Accuracy: {analysis['overall_accuracy']:.3f} ({analysis['correct_predictions']}/{analysis['total_predictions']})",
                f"Average Confidence Score: {analysis['avg_confidence']:.3f}",
                "",
                "ACCURACY BY CONFIDENCE LEVEL:",
                "-" * 40
            ]
            
            for level, metrics in analysis['accuracy_by_confidence'].items():
                report.append(f"{level.upper():>8}: {metrics['accuracy']:.3f} ({metrics['count']} predictions, avg conf: {metrics['avg_confidence']:.3f})")
            
            report.extend([
                "",
                "ACCURACY BY SIGNAL TYPE:",
                "-" * 40
            ])
            
            for signal_type, metrics in analysis['accuracy_by_signal_type'].items():
                report.append(f"{signal_type:>10}: {metrics['accuracy']:.3f} ({metrics['count']} signals)")
            
            report.extend([
                "",
                "DIRECTIONAL ACCURACY:",
                "-" * 40,
                f"Direction Correct: {analysis['directional_accuracy']['directional_accuracy']:.3f} ({analysis['directional_accuracy']['directional_correct']}/{analysis['directional_accuracy']['directional_total']})",
                "",
                "TOP PREDICTION ERRORS:",
                "-" * 40
            ])
            
            for i, error in enumerate(analysis['transition_analysis']['top_prediction_errors'][:5], 1):
                report.append(f"{i}. Predicted {error['predicted']} → Actually {error['actual']} ({error['count']} times)")
            
            report.extend([
                "",
                "MOST ACCURATE STATE PREDICTIONS:",
                "-" * 40
            ])
            
            for pred in analysis['transition_analysis']['most_accurate_predictions'][:5]:
                report.append(f"{pred['state']:>15}: {pred['accuracy']:.3f} ({pred['correct_predictions']}/{pred['total_predictions']})")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error generating quality report: {str(e)}"


def main():
    """Run signal quality analysis on sample data."""
    
    import csv
    from datetime import date
    from decimal import Decimal
    
    # Load sample data
    data = []
    with open('sample_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(MarketData(
                date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
                open=Decimal(row['open']),
                high=Decimal(row['high']),
                low=Decimal(row['low']),
                close=Decimal(row['close']),
                volume=int(row['volume']),
                iv=Decimal(row['iv'])
            ))
    
    # Analyze signal quality
    analyzer = SignalQualityAnalyzer()
    report = analyzer.generate_quality_report(data)
    
    print(report)
    
    # Save detailed analysis
    analysis = analyzer.analyze_prediction_accuracy(data)
    
    # Create summary for team lead
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY FOR TECH LEAD")
    print("=" * 80)
    print(f"PREDICTION ACCURACY: {analysis['overall_accuracy']:.1%}")
    print(f"TOTAL PREDICTIONS: {analysis['total_predictions']}")
    print(f"HIGH CONFIDENCE ACCURACY: {analysis['accuracy_by_confidence']['high']['accuracy']:.1%}")
    print(f"DIRECTIONAL ACCURACY: {analysis['directional_accuracy']['directional_accuracy']:.1%}")
    
    if analysis['overall_accuracy'] < 0.4:
        print("\nCRITICAL: Prediction accuracy below 40% - Model needs immediate attention")
    elif analysis['overall_accuracy'] < 0.5:
        print("\nWARNING: Prediction accuracy below 50% - Model needs tuning")
    else:
        print("\nACCEPTABLE: Prediction accuracy above 50%")


if __name__ == "__main__":
    main()