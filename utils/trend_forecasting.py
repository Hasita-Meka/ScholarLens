"""
Trend forecasting for ScholarLens
Time-series analysis of method/dataset popularity with predictions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict


class TrendForecaster:
    """
    Simple trend forecasting using linear regression and moving averages
    """
    
    def __init__(self):
        self.data = {}
    
    def fit(self, years: List[int], counts: List[int]):
        """
        Fit trend model to historical data
        """
        if len(years) < 2:
            self.slope = 0
            self.intercept = counts[0] if counts else 0
            return
        
        x = np.array(years)
        y = np.array(counts)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            self.slope = 0
            self.intercept = y_mean
        else:
            self.slope = numerator / denominator
            self.intercept = y_mean - self.slope * x_mean
    
    def predict(self, future_years: List[int]) -> List[float]:
        """
        Predict values for future years
        """
        predictions = []
        for year in future_years:
            pred = self.slope * year + self.intercept
            predictions.append(max(0, pred))
        return predictions
    
    def get_trend_direction(self) -> str:
        """
        Get trend direction (rising, falling, stable)
        """
        if self.slope > 0.5:
            return "rising"
        elif self.slope < -0.5:
            return "falling"
        else:
            return "stable"
    
    def get_growth_rate(self) -> float:
        """
        Get annual growth rate percentage
        """
        if self.intercept == 0:
            return 0.0
        return (self.slope / self.intercept) * 100


def analyze_method_trends(method_data: List[Dict]) -> Dict:
    """
    Analyze trends for methods over time
    
    Args:
        method_data: List of dicts with 'method', 'year', 'count' keys
    
    Returns:
        Dictionary with trend analysis results
    """
    if not method_data:
        return {}
    
    method_series = defaultdict(lambda: defaultdict(int))
    for item in method_data:
        method = item.get('method', 'Unknown')
        year = item.get('year')
        count = item.get('count', 0)
        if year:
            method_series[method][year] += count
    
    trends = {}
    current_year = datetime.now().year
    future_years = [current_year + 1, current_year + 2, current_year + 3]
    
    for method, year_counts in method_series.items():
        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]
        
        if len(years) < 2:
            continue
        
        forecaster = TrendForecaster()
        forecaster.fit(years, counts)
        
        predictions = forecaster.predict(future_years)
        
        recent_years = years[-3:] if len(years) >= 3 else years
        recent_counts = [year_counts[y] for y in recent_years]
        
        if len(recent_counts) >= 2:
            recent_change = recent_counts[-1] - recent_counts[0]
            recent_growth = (recent_change / recent_counts[0] * 100) if recent_counts[0] > 0 else 0
        else:
            recent_growth = 0
        
        total_count = sum(counts)
        peak_year = years[counts.index(max(counts))]
        
        trends[method] = {
            'historical': {year: count for year, count in zip(years, counts)},
            'predictions': {year: pred for year, pred in zip(future_years, predictions)},
            'trend_direction': forecaster.get_trend_direction(),
            'growth_rate': forecaster.get_growth_rate(),
            'recent_growth': recent_growth,
            'total_count': total_count,
            'peak_year': peak_year,
            'peak_count': max(counts),
            'first_appeared': min(years),
            'latest_year': max(years)
        }
    
    return trends


def identify_emerging_methods(trends: Dict, min_recent_growth: float = 20.0) -> List[Dict]:
    """
    Identify methods that are emerging (high recent growth)
    """
    emerging = []
    
    for method, data in trends.items():
        if data['recent_growth'] >= min_recent_growth and data['trend_direction'] == 'rising':
            emerging.append({
                'method': method,
                'recent_growth': data['recent_growth'],
                'growth_rate': data['growth_rate'],
                'predictions': data['predictions']
            })
    
    emerging.sort(key=lambda x: x['recent_growth'], reverse=True)
    return emerging


def identify_declining_methods(trends: Dict, min_decline: float = -20.0) -> List[Dict]:
    """
    Identify methods that are declining
    """
    declining = []
    
    for method, data in trends.items():
        if data['recent_growth'] <= min_decline and data['trend_direction'] == 'falling':
            declining.append({
                'method': method,
                'recent_growth': data['recent_growth'],
                'peak_year': data['peak_year'],
                'peak_count': data['peak_count']
            })
    
    declining.sort(key=lambda x: x['recent_growth'])
    return declining


def calculate_moving_average(data: List[Tuple[int, int]], window: int = 3) -> List[Tuple[int, float]]:
    """
    Calculate moving average for time series
    """
    if len(data) < window:
        return [(year, float(count)) for year, count in data]
    
    sorted_data = sorted(data, key=lambda x: x[0])
    years = [d[0] for d in sorted_data]
    counts = [d[1] for d in sorted_data]
    
    result = []
    for i in range(len(counts)):
        start_idx = max(0, i - window + 1)
        window_vals = counts[start_idx:i + 1]
        avg = sum(window_vals) / len(window_vals)
        result.append((years[i], avg))
    
    return result


def compare_method_trajectories(method_trends: Dict, methods: List[str]) -> Dict:
    """
    Compare trajectories of multiple methods
    """
    comparison = {}
    
    for method in methods:
        if method not in method_trends:
            continue
        
        data = method_trends[method]
        comparison[method] = {
            'historical': data['historical'],
            'predictions': data['predictions'],
            'trend': data['trend_direction'],
            'growth': data['growth_rate']
        }
    
    return comparison


def generate_forecast_summary(trends: Dict) -> str:
    """
    Generate a natural language summary of trends
    """
    if not trends:
        return "No trend data available."
    
    rising = [m for m, d in trends.items() if d['trend_direction'] == 'rising']
    falling = [m for m, d in trends.items() if d['trend_direction'] == 'falling']
    stable = [m for m, d in trends.items() if d['trend_direction'] == 'stable']
    
    summary_parts = []
    
    if rising:
        top_rising = sorted(rising, key=lambda m: trends[m]['recent_growth'], reverse=True)[:3]
        summary_parts.append(f"**Rising trends:** {', '.join(top_rising)}")
    
    if falling:
        top_falling = sorted(falling, key=lambda m: trends[m]['recent_growth'])[:3]
        summary_parts.append(f"**Declining:** {', '.join(top_falling)}")
    
    if stable:
        summary_parts.append(f"**Stable methods:** {len(stable)} methods showing steady usage")
    
    all_methods = list(trends.items())
    if all_methods:
        newest = max(all_methods, key=lambda x: x[1]['first_appeared'])
        summary_parts.append(f"**Newest:** {newest[0]} (first appeared {newest[1]['first_appeared']})")
    
    return "\n\n".join(summary_parts)


def create_timeline_data(trends: Dict, methods: List[str] = None) -> Dict:
    """
    Create data for timeline visualization
    """
    if methods is None:
        methods = list(trends.keys())[:10]
    
    timeline_data = {
        'years': set(),
        'series': {}
    }
    
    for method in methods:
        if method not in trends:
            continue
        
        data = trends[method]
        all_years = list(data['historical'].keys()) + list(data['predictions'].keys())
        timeline_data['years'].update(all_years)
        
        timeline_data['series'][method] = {
            'historical': data['historical'],
            'predicted': data['predictions']
        }
    
    timeline_data['years'] = sorted(timeline_data['years'])
    
    return timeline_data
