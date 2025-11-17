import time
from functools import wraps
from typing import Dict, List, Callable
from collections import defaultdict
import numpy as np


class PerformanceTracker:
    """Track timing and performance metrics"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.query_results = []
        self.enabled = True
    
    def time_it(self, component_name: str):
        """Context manager to track execution time"""
        class Timer:
            def __init__(self, tracker, name):
                self.tracker = tracker
                self.name = name
                self.start = None
            
            def __enter__(self):
                if self.tracker.enabled:
                    self.start = time.time()
                return self
            
            def __exit__(self, *args):
                if self.tracker.enabled and self.start:
                    elapsed = (time.time() - self.start) * 1000
                    self.tracker.timings[self.name].append(elapsed)
        
        return Timer(self, component_name)
    
    def track_function(self, component_name: str):
        """Decorator to track function execution time"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.enabled:
                    with self.time_it(component_name):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def record_query(self, query: str, response: Dict, success: bool, 
                     total_time_ms: float):
        """Record a complete query execution"""
        if self.enabled:
            self.query_results.append({
                'query': query,
                'success': success,
                'total_time_ms': total_time_ms,
                'timestamp': time.time(),
                'num_sources': len(response.get('sources', [])),
                'confidence': response.get('confidence', 'unknown'),
                'chunks_found': response.get('chunks_found', 0)
            })
    
    def get_stats(self) -> Dict:
        """Get aggregated statistics"""
        stats = {}
        
        # Component timings
        for component, timings in self.timings.items():
            if timings:
                stats[component] = {
                    'mean_ms': float(np.mean(timings)),
                    'median_ms': float(np.median(timings)),
                    'p95_ms': float(np.percentile(timings, 95)),
                    'p99_ms': float(np.percentile(timings, 99)),
                    'min_ms': float(np.min(timings)),
                    'max_ms': float(np.max(timings)),
                    'std_ms': float(np.std(timings)),
                    'count': len(timings)
                }
        
        # Query-level stats
        if self.query_results:
            successful = [r for r in self.query_results if r['success']]
            
            stats['overall'] = {
                'total_queries': len(self.query_results),
                'successful': len(successful),
                'failed': len(self.query_results) - len(successful),
                'success_rate': len(successful) / len(self.query_results) * 100,
            }
            
            if successful:
                stats['overall'].update({
                    'avg_sources': float(np.mean([r['num_sources'] for r in successful])),
                    'avg_chunks': float(np.mean([r['chunks_found'] for r in successful])),
                    'avg_response_time_ms': float(np.mean([r['total_time_ms'] for r in successful]))
                })
        
        return stats
    
    def reset(self):
        """Reset all tracked metrics"""
        self.timings.clear()
        self.query_results.clear()
    
    def enable(self):
        """Enable tracking"""
        self.enabled = True
    
    def disable(self):
        """Disable tracking"""
        self.enabled = False


# Global tracker instance
tracker = PerformanceTracker()