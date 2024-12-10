import os
import json

def calculate_average_statistics(folder_path):
    cpu_usage = []
    memory_usage = []
    latency = []

    
    # Read all JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                data = json.load(file)
                cpu = data['details']['agent']['avg_cpu']['value'] + data['details']['main_ollama']['avg_cpu']['value'] + data['details']['dynamic_ollama']['avg_cpu']['value']
                cpu_usage.append(cpu)
                memory = data['details']['agent']['peak_memory']['value'] + data['details']['main_ollama']['peak_memory']['value'] + data['details']['dynamic_ollama']['peak_memory']['value']
                memory_usage.append(memory)
                latency.append(data['details']['latency']['average']['value'])

    # Calculate average statistics
    avg_cpu = sum(cpu_usage) / len(cpu_usage)
    avg_memory = sum(memory_usage) / len(memory_usage)
    avg_latency = sum(latency) / len(latency)

    print(f'Average CPU Usage: {avg_cpu}%')
    print(f'Average Memory Usage: {avg_memory}MB')
    print(f'Average Latency: {avg_latency}ms')

if __name__ == "__main__":
    folder_path = './14b'
    calculate_average_statistics(folder_path)