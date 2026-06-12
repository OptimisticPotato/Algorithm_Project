from utilities import *
from greedy import calculate_greedy_tsp
from two_opt import run_2opt_heuristic
from held_karp import run_held_karp
from proposed import optimize_drilling_path

import time

N = 300           # 생성할 홀의 개수
x_max = 200.0      # 가로 크기
y_max = 200.0      # 세로 크기
r_hole = 2.0      # 홀 반지름

use_predefined_holes = False  # True로 설정하면 14개의 사전 정의된 홀 좌표를 사용합니다. False로 설정하면 무작위 홀을 생성합니다.

import time
import random
import numpy as np
import pandas as pd
import tracemalloc

if __name__ == "__main__":

    if use_predefined_holes:
        N_list = [14]
    else:
        # N_list = [10, 20, 50, 100, 300, 1000]
        N_list = [100]

    results = []

    random.seed(0)
    np.random.seed(0)

    for N in N_list:

        print(f"\n===== Running N = {N} =====")

        # -----------------------------
        # Hole generation
        # -----------------------------
        t0 = time.perf_counter()
        tracemalloc.start()

        if use_predefined_holes:
            hole_positions = hole_14positions
        else:
            hole_positions = generate_hole_coordinates(N, x_max, y_max, r_hole)

        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t1 = time.perf_counter()

        gen_time = t1 - t0
        gen_mem = mem_peak / 1024  # KB

        # -----------------------------
        # Greedy TSP
        # -----------------------------
        t0 = time.perf_counter()
        tracemalloc.start()

        greedy_path, greedy_distance = calculate_greedy_tsp(hole_positions)

        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t1 = time.perf_counter()

        greedy_time = t1 - t0
        greedy_mem = mem_peak / 1024

        # -----------------------------
        # 2-opt
        # -----------------------------
        if N < 1000:
                
            
            t0 = time.perf_counter()
            tracemalloc.start()

            initial_path = greedy_path
            two_opt_path, two_opt_distance, iteration_count = run_2opt_heuristic(initial_path)

            mem_current, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            t1 = time.perf_counter()

            two_opt_time = t1 - t0
            two_opt_mem = mem_peak / 1024
        else:
            two_opt_distance = None
            two_opt_time = None
            two_opt_mem = None

        # -----------------------------
        # Held-Karp
        # -----------------------------
        if N <= 20:
            t0 = time.perf_counter()
            tracemalloc.start()

            held_karp_path, held_karp_distance = run_held_karp(hole_positions)

            mem_current, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            t1 = time.perf_counter()

            hk_time = t1 - t0
            hk_mem = mem_peak / 1024
        else:
            held_karp_path = None
            held_karp_distance = None
            hk_time = None
            hk_mem = None

        # -----------------------------
        # Proposed method
        # -----------------------------
        t0 = time.perf_counter()
        tracemalloc.start()

        proposed_path, proposed_distance = optimize_drilling_path(hole_positions, portal_k=3)

        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t1 = time.perf_counter()

        proposed_time = t1 - t0
        proposed_mem = mem_peak / 1024

        # -----------------------------
        # Save result row
        # -----------------------------
        results.append({
            "N": N,

            "greedy_distance": greedy_distance,
            "two_opt_distance": two_opt_distance,
            "held_karp_distance": held_karp_distance,
            "proposed_distance": proposed_distance,

            "greedy_time_s": greedy_time,
            "two_opt_time_s": two_opt_time,
            "held_karp_time_s": hk_time,
            "proposed_time_s": proposed_time,

            "greedy_mem_kb": greedy_mem,
            "two_opt_mem_kb": two_opt_mem,
            "held_karp_mem_kb": hk_mem,
            "proposed_mem_kb": proposed_mem,

            "hole_gen_time_s": gen_time,
            "hole_gen_mem_kb": gen_mem,

            "2opt_iterations": iteration_count
        })
        
        if use_predefined_holes | len(N_list) == 1:
            visualize_four_paths(
            hole_positions=hole_positions,
            greedy_path=greedy_path,
            two_opt_path=two_opt_path,
            held_karp_path=held_karp_path, # 만약 N이 너무 커서 안 돌렸다면 [] 를 넣으시면 됩니다.
            proposed_path=proposed_path
            )

        # -----------------------------
        # incremental CSV save (fail-safe)
        # -----------------------------
        df = pd.DataFrame(results)
        df.to_csv("tsp_experiment_results.csv", index=False)

    print("\n===== Experiment Complete =====")
    
    
    