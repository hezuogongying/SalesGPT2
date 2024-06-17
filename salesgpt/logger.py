import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

stream_handler = logging.StreamHandler()
log_filename = "output.log"
file_handler = logging.FileHandler(filename=log_filename)
handlers = [stream_handler, file_handler]


class TimeFilter(logging.Filter):
    def filter(self, record):
        return "Running" in record.getMessage()


logger.addFilter(TimeFilter())

# 配置日志记录模块
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(asctime)s - %(levelname)s - %(message)s",
    handlers=handlers,
)


def time_logger(func):
    """
    装饰器函数用于记录任何函数所花费的时间。

    该装饰器记录被装饰函数的执行时间。它记录函数之前的开始时间
    执行，函数执行后的结束时间，并计算执行时间。函数名称和
    然后将执行时间记录在 INFO 级别。

    Args:
        func (Callable): 要装饰的功能。

    Returns:
        Callable: 装饰函数.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function execution
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time after function execution
        execution_time = end_time - start_time  # Calculate execution time
        logger.info(f"Running {func.__name__}: --- {execution_time} seconds ---")
        return result

    return wrapper
