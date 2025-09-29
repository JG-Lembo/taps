import sys
import re

def count_retries(file):
    count = 0
    runtime_seconds = None
    
    for line in file:
        count += line.count("retry_sleep")
        
        if "Finished app" in line:
            match = re.search(r'runtime=(\d+\.\d+)s', line)
            if match:
                runtime_seconds = float(match.group(1))
    
    return count, runtime_seconds

if __name__ == "__main__":
    count, runtime_seconds = count_retries(sys.stdin)
    
    print(f"retry_sleep count: {count}")
    if runtime_seconds is not None:
        print(f"Runtime in seconds: {runtime_seconds}")
    else:
        print("No runtime found after 'Finished app'")
