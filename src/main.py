import time
import psutil
import threading
import requests
import json
from datetime import datetime
from openai import OpenAI

from agent_module import *
import agent_module.Agent as Agent


class LatencyTrackingClient(OpenAI):
    """A subclass of OpenAI to track the latency of real requests."""
    
    def __init__(self, *args, metrics, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def request(self, method, url, **kwargs):
        start_time = time.time()
        try:
            response = super().request(method, url, **kwargs)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics["latency"].append(latency)
            return response
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.metrics["latency"].append(None)  # Track failed requests
            print(f"Request failed: {e}")
            raise


def find_dynamic_process(process_name):
    """Find a process by its name. Returns the PID if found, else None."""
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if process_name in proc.info['name']:
            return proc.info['pid']
    return None


def run_agent_with_latency_tracking(metrics):
    """Run the agent with a latency tracking client."""
    for _ in range(1):
        self_evolving_agent = Agent.Agent(goal_prompt_path="goal_prompt.md")
        self_evolving_agent.client = LatencyTrackingClient(
            api_key="ollama", base_url="http://localhost:11434/v1", metrics=metrics
        )
        self_evolving_agent.reinit()
        self_evolving_agent.evolve()


def monitor_process(pid, metrics, stop_event, duration=0.1):
    """Monitor memory and CPU usage of a process."""
    process = psutil.Process(pid)
    while not stop_event.is_set():
        try:
            metrics["cpu_usage"].append(process.cpu_percent(interval=None))
            metrics["memory_usage"].append(process.memory_info().rss / (1024 * 1024))  # MB
            time.sleep(duration)
        except psutil.NoSuchProcess:
            break


def benchmark_agent(main_ollama_pid, log_file="resource_usage_log.json"):
    """Benchmark resource usage and latency."""
    # Initialize metrics
    metrics = {
        "main_ollama": {"cpu_usage": [], "memory_usage": []},
        "dynamic_ollama": {"cpu_usage": [], "memory_usage": []},
        "agent": {"cpu_usage": [], "memory_usage": []},
        "latency": [],
    }

    # Event to signal monitoring threads to stop
    stop_event = threading.Event()

    # Start the agent in a separate thread
    agent_thread = threading.Thread(target=run_agent_with_latency_tracking, args=(metrics,))
    agent_thread.start()

    # Monitor the agent process
    agent_process = psutil.Process()  # Current Python process
    monitor_agent_thread = threading.Thread(
        target=monitor_process,
        args=(agent_process.pid, metrics["agent"], stop_event),
    )
    monitor_agent_thread.start()

    # Monitor the main Ollama process
    monitor_main_ollama_thread = threading.Thread(
        target=monitor_process,
        args=(main_ollama_pid, metrics["main_ollama"], stop_event),
    )
    monitor_main_ollama_thread.start()

    # Dynamically monitor the ollama_llama_server process
    def monitor_dynamic_process(metrics, stop_event):
        dynamic_pid = None
        while not stop_event.is_set():
            if dynamic_pid is None:  # Look for the dynamic process
                dynamic_pid = find_dynamic_process("ollama_llama_server")
                if dynamic_pid:
                    print(f"Found dynamic Ollama process: {dynamic_pid}")
            if dynamic_pid:
                try:
                    monitor_process(dynamic_pid, metrics["dynamic_ollama"], stop_event)
                except psutil.NoSuchProcess:
                    dynamic_pid = None  # Reset if the process stops
            time.sleep(0.1)

    monitor_dynamic_thread = threading.Thread(target=monitor_dynamic_process, args=(metrics, stop_event))
    monitor_dynamic_thread.start()

    # Start time for the benchmark
    start_time = datetime.now()

    # Wait for the agent to complete
    agent_thread.join()

    # End time for the benchmark
    end_time = datetime.now()

    # Signal monitoring threads to stop
    stop_event.set()
    monitor_agent_thread.join()
    monitor_main_ollama_thread.join()
    monitor_dynamic_thread.join()

    # Collect peak memory and average CPU usage
    filtered_latency = list(filter(None, metrics["latency"]))  # Convert filter result to a list
    results = {
        "summary": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": {
                "value": (end_time - start_time).total_seconds(),
                "unit": "seconds",
            },
        },
        "details": {
            "agent": {
                "peak_memory": {"value": max(metrics["agent"]["memory_usage"], default=0), "unit": "MB"},
                "avg_cpu": {"value": sum(metrics["agent"]["cpu_usage"]) / len(metrics["agent"]["cpu_usage"]) if metrics["agent"]["cpu_usage"] else 0, "unit": "%"},
            },
            "main_ollama": {
                "peak_memory": {"value": max(metrics["main_ollama"]["memory_usage"], default=0), "unit": "MB"},
                "avg_cpu": {"value": sum(metrics["main_ollama"]["cpu_usage"]) / len(metrics["main_ollama"]["cpu_usage"]) if metrics["main_ollama"]["cpu_usage"] else 0, "unit": "%"},
            },
            "dynamic_ollama": {
                "peak_memory": {"value": max(metrics["dynamic_ollama"]["memory_usage"], default=0), "unit": "MB"},
                "avg_cpu": {"value": sum(metrics["dynamic_ollama"]["cpu_usage"]) / len(metrics["dynamic_ollama"]["cpu_usage"]) if metrics["dynamic_ollama"]["cpu_usage"] else 0, "unit": "%"},
            },
            "latency": {
                "average": {"value": sum(filtered_latency) / len(filtered_latency) if filtered_latency else 0, "unit": "ms"},
                "min": {"value": min(filtered_latency, default=0), "unit": "ms"},
                "max": {"value": max(filtered_latency, default=0), "unit": "ms"},
            },
        },
    }

    # Print results
    print(json.dumps(results, indent=4))

    results["raw_metrics"] = metrics

    # Save results to a file
    save_results_to_file(results, log_file)

    return results


def save_results_to_file(data, file_name="resource_usage_log.json"):
    """Save results to a file."""
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {file_name}")


if __name__ == "__main__":
    main_ollama_pid = 93421
    log_file_name = "ollama_resource_tracking.json"  # Change as needed
    benchmark_agent(main_ollama_pid, log_file_name)