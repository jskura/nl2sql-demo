import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)

def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log the execution time of a function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The wrapped function with execution time logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__
        
        logger.info(
            f"Function {module_name}.{func_name} executed in {duration:.2f}ms",
            extra={"duration": duration, "function": f"{module_name}.{func_name}"}
        )
        
        return result
    
    return wrapper

def async_log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log the execution time of an async function.
    
    Args:
        func: The async function to be decorated
        
    Returns:
        The wrapped async function with execution time logging
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__
        
        logger.info(
            f"Async function {module_name}.{func_name} executed in {duration:.2f}ms",
            extra={"duration": duration, "function": f"{module_name}.{func_name}"}
        )
        
        return result
    
    return wrapper

class PerformanceTracker:
    """
    Context manager for tracking performance of code blocks.
    
    Example:
        with PerformanceTracker("database_query"):
            result = db.execute_query(...)
    """
    
    def __init__(self, operation_name: str):
        """
        Initialize the performance tracker.
        
        Args:
            operation_name: A descriptive name for the operation being tracked
        """
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.start_time:
            return
            
        duration = (time.time() - self.start_time) * 1000  # Convert to milliseconds
        
        if exc_type:
            # If an exception occurred
            logger.warning(
                f"Operation '{self.operation_name}' failed after {duration:.2f}ms",
                extra={
                    "duration": duration,
                    "operation": self.operation_name,
                    "error": str(exc_val)
                }
            )
        else:
            logger.info(
                f"Operation '{self.operation_name}' completed in {duration:.2f}ms",
                extra={"duration": duration, "operation": self.operation_name}
            ) 