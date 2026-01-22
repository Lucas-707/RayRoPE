import time
import torch
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, Optional
import threading

class TimingStats:
    """Thread-safe timing statistics collector."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(list)
        self._current_iteration_stats = defaultdict(float)
        
    def record_time(self, name: str, duration: float):
        """Record a timing measurement."""
        with self._lock:
            self._current_iteration_stats[name] += duration
    
    def end_iteration(self) -> Dict[str, float]:
        """End current iteration and return aggregated stats."""
        with self._lock:
            stats = dict(self._current_iteration_stats)
            for name, total_time in stats.items():
                self._stats[name].append(total_time)
            self._current_iteration_stats.clear()
            return stats
    
    def get_recent_stats(self, n: int = 1) -> Dict[str, float]:
        """Get average stats from last n iterations."""
        with self._lock:
            result = {}
            for name, times in self._stats.items():
                if times:
                    recent_times = times[-n:] if len(times) >= n else times
                    result[name] = sum(recent_times) / len(recent_times)
            return result
    
    def clear(self):
        """Clear all statistics."""
        with self._lock:
            self._stats.clear()
            self._current_iteration_stats.clear()

# Global timing stats instance
_timing_stats = TimingStats()

def get_timing_stats() -> TimingStats:
    """Get the global timing stats instance."""
    return _timing_stats

@contextmanager
def time_block(name: str, enabled: bool = True):
    """Context manager to time a block of code."""
    if not enabled:
        yield
        return
        
    torch.cuda.synchronize()  # Ensure GPU operations are complete
    start_time = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()  # Ensure GPU operations are complete
        end_time = time.perf_counter()
        duration = end_time - start_time
        _timing_stats.record_time(name, duration)