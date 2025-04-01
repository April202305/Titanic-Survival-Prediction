# ==== 新增依赖 ====
import time
import psutil
import os
from pympler import asizeof




def measure_performance(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        print(f"\nPerformance Metrics:")
        print(f"Time: {time.time() - start_time:.2f}s")
        print(f"Memory: {(process.memory_info().rss - start_mem)/1024**2:.2f}MB")
        return result
    return wrapper

# 使用方式
@measure_performance
def train_model():
    grid_search.fit(X_train, y_train)
