import time
import datetime

def runtime(func):
    def measure_time(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed to completion of {func.__name__}: {datetime.timedelta(seconds=end-start)}")
        return output
    return measure_time
