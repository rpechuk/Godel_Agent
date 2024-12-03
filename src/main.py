# from agent_module import Agent, self_evolving_agent

import time
import psutil
import threading

from agent_modules import *
import agent_modules.Agent as Agent

def run_agent():
    for _ in range(1):
        self_evolving_agent = Agent.Agent(goal_prompt_path="goal_prompt2.md")
        self_evolving_agent.reinit()
        self_evolving_agent.evolve()
        
def monitor_process(pid, metrics, stop_event, duration=0.1):
    """Monitor memory and CPU usage of a process."""
    process = psutil.Process(pid)
    while not stop_event.is_set():  # Stop monitoring when the event is set
        try:
            # Capture memory and CPU usage
            metrics["cpu_usage"].append(process.cpu_percent(interval=None))
            metrics["memory_usage"].append(process.memory_info().rss / (1024 * 1024))  # Convert bytes to MB
            time.sleep(duration)  # Sampling interval
        except psutil.NoSuchProcess:
            break

def benchmark_agent():
    """Benchmark memory, CPU, and latency."""
    # Initialize metrics
    metrics = {"cpu_usage": [], "memory_usage": []}

    # Event to signal the monitor to stop
    stop_event = threading.Event()

    # Start the agent in a separate thread
    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()

    # Monitor the process
    process = psutil.Process()  # Get the current Python process
    monitor_thread = threading.Thread(target=monitor_process, args=(process.pid, metrics, stop_event))
    monitor_thread.start()

    # Start time for latency
    start_time = time.time()

    # Wait for the agent to complete
    agent_thread.join()  # Wait for the agent thread to finish

    # End time for latency
    end_time = time.time()

    # Signal the monitoring thread to stop
    stop_event.set()
    monitor_thread.join()

    # Collect results
    peak_memory = max(metrics["memory_usage"]) if metrics["memory_usage"] else 0
    avg_cpu = sum(metrics["cpu_usage"]) / len(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0
    latency = end_time - start_time

    # Output the benchmark results
    print(f"Benchmark Results:")
    print(f"------------------")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    print(f"Average CPU Usage: {avg_cpu:.2f}%")
    print(f"Total Latency (Execution Time): {latency:.2f} seconds")

    return peak_memory, avg_cpu, latency

if __name__ == "__main__":
    # run_agent() # will just run agent without benchmarking
    peak_memory, avg_cpu, latency = benchmark_agent() 