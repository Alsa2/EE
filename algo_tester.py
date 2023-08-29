import time
import psutil
import os
import csv
from multiprocessing import Process, Manager

from map import Map
from algorithms import BFS_checker, Astar, PRM, DQN

def run_test(test_map, algorithm, results, index):
    start_time = time.time()
    status, path = algorithm(test_map.map, test_map.start, test_map.end)
    end_time = time.time()

    results[index] = {
        'algorithm': algorithm.__name__,
        'size': test_map.width,  # assuming square map so width = height = size
        'time': end_time - start_time,
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'path_length': len(path) if status else None
    }

def main():
    sizes = [10, 50, 100, 200]
    algorithms = [BFS_checker, Astar, PRM, DQN]

    # Warm up
    print("Warming Up Baby")
    #test_map = Map(10000, 10000)
    print("Warming Up D0n€ :)")

    # How many tests for average value
    average = 200

    results = []

    for size in sizes:
        for _ in range(average):  # Run 10 times
            test_map = Map(size, size)
            manager = Manager()
            algorithm_results = manager.dict()

            # Run each algorithm on a separate process
            processes = []
            for i, algorithm in enumerate(algorithms):
                process = Process(target=run_test, args=(test_map, algorithm, algorithm_results, i))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            results += list(algorithm_results.values())

    # Save results to CSV
    with open('results.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['algorithm', 'size', 'time', 'cpu_percent', 'memory_percent', 'path_length'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    main()