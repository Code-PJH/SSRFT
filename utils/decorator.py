import time
import logging
from functools import wraps

def retry_evaluation(max_retries=5, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_result = (None, "Max retries exceeded", None)
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    # Check if evaluation was successful (score is not None)
                    if result[0] is not None:
                        return result
                    last_result = result
                except Exception as e:
                    logging.error(f"Error in {func.__name__} (Attempt {attempt+1}/{max_retries}): {e}")
                    last_result = (None, f"Error: {e}", None)
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
            return last_result
        return wrapper
    return decorator