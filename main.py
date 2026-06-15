import os
import csv
import math
import time
import random
import tracemalloc
import matplotlib.pyplot as plt
from collections import Counter

# 모듈 임포트 (기존에 설정하신 파일명에 맞춤)
from Utilities import *
from GreedyAlgorithm import greedy_tsp
from Held_Karp import held_karp
from TwoOptBaseline import *
from Proposed import *
from LowerBound import held_karp_lower_bound

NUM_HOLES = 300
WIDTH     = 500
HEIGHT    = 500
TOOL_TYPES = ['M4', 'M8']  # 예시 공구 종류

def save_results_to_csv(num_holes, width, height, tool_types, tool_counts, results_list):
    """
    수집된 벤치마크 결과를 Results 폴더 내에 result_1.csv, result_2.csv 형태로 스택 저장합니다.
    tool_types 및 tool_counts 정보도 함께 저장합니다.
    """
    res_dir = "Results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # 기존 result_N.csv 파일들을 확인하여 가장 큰 N 값을 찾음
    max_idx = 0
    for f in os.listdir(res_dir):
        if f.startswith("result_") and f.endswith(".csv"):
            try:
                idx = int(f.replace("result_", "").replace(".csv", ""))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
                
    new_idx = max_idx + 1
    filepath = os.path.join(res_dir, f"result_{new_idx}.csv")
    
    # CSV 파일 작성
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 헤더 작성 (TOOL_COUNTS 추가)
        writer.writerow(["NUM_HOLES", "WIDTH", "HEIGHT", "TOOL_TYPES", "TOOL_COUNTS", "Algorithm", "Cost", "Time(s)", "Peak Memory(KB)"])
        # 데이터 작성
        for row in results_list:
            writer.writerow([num_holes, width, height, tool_types, tool_counts, row[0], row[1], row[2], row[3]])
            
    print(f"\n[Info] Benchmark results successfully saved to {filepath}")


def main_function():
    random.seed(42)
    print("=== State-Dependent Cost TSP (TCP-TSP) Benchmark ===")
    
    # 홀 생성
    holes = generate_random_holes(num_holes=NUM_HOLES, width=WIDTH, height=HEIGHT, tool_types=TOOL_TYPES)
    print(f"[Info] Generated {len(holes)} holes with different tools.")
    
    # [추가된 부분] 각 홀 종류별 개수 카운트
    tool_counts_dict = dict(Counter(h.tool for h in holes))
    tool_counts_str = str(tool_counts_dict)  # CSV에 텍스트로 넣기 위해 문자열 변환
    print(f"[Info] Tool counts: {tool_counts_str}\n")

    # 결과를 담을 빈 리스트
    benchmark_results = []

    # 1. Greedy Algorithm
    start_t = time.perf_counter()
    tracemalloc.start()
    greedy_path = greedy_tsp(holes, debug=False)
    greedy_cost = calculate_path_cost(greedy_path, holes)
    greedy_time = time.perf_counter() - start_t
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    greedy_mem = mem_peak / 1024
    print(f"[Greedy] \t\tCost: {greedy_cost:7.2f} | Time: {greedy_time:.4f}s | Peak Memory: {greedy_mem:.2f} KB")
    benchmark_results.append(["Greedy Algorithm", greedy_cost, greedy_time, greedy_mem])

    # 2. Tool-Bundled 2-Opt (현업 묶음 가공 모사)
    if NUM_HOLES > 400:
        print(f"[Tool-Bundled 2-Opt] \tCost: N/A (Too many holes for 2-Opt) | Time: N/A | Peak Memory: N/A")
        # 값이 없을 경우 결측치(nan)로 기록
        tb_cost, tb_time, tb_mem = float('nan'), float('nan'), float('nan')
        benchmark_results.append(["Tool-Bundled 2-Opt", tb_cost, tb_time, tb_mem])
    else:
        start_t = time.perf_counter()
        tracemalloc.start()
        tb_path = tool_bundled_two_opt(holes)
        tb_cost = calculate_path_cost(tb_path, holes)
        tb_time = time.perf_counter() - start_t
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tb_mem = mem_peak / 1024
        print(f"[Tool-Bundled 2-Opt]\tCost: {tb_cost:7.2f} | Time: {tb_time:.4f}s | Peak Memory: {tb_mem:.2f} KB")
        benchmark_results.append(["Tool-Bundled 2-Opt", tb_cost, tb_time, tb_mem])
        

    # 3. State-Aware 2-Opt (초기 경로는 Greedy 사용)
    if NUM_HOLES > 400:
        print(f"[State-Aware 2-Opt] \tCost: N/A (Too many holes for 2-Opt) | Time: N/A | Peak Memory: N/A")
        # 값이 없을 경우 결측치(nan)로 기록
        sa_cost, sa_time, sa_mem = float('nan'), float('nan'), float('nan')
        benchmark_results.append(["State-Aware 2-Opt", sa_cost, sa_time, sa_mem])
    else:
        start_t = time.perf_counter()
        tracemalloc.start()
        sa_path = state_aware_two_opt(holes, greedy_path)
        sa_cost = calculate_path_cost(sa_path, holes)
        sa_time = time.perf_counter() - start_t
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sa_mem = mem_peak / 1024
        print(f"[State-Aware 2-Opt] \tCost: {sa_cost:7.2f} | Time: {sa_time:.4f}s | Peak Memory: {sa_mem:.2f} KB")
        benchmark_results.append(["State-Aware 2-Opt", sa_cost, sa_time, sa_mem])

    # 4. Held-Karp (Optimal - 진정한 최적해 보장)
    if NUM_HOLES > 20:
        print(f"[Held-Karp (Optimal)] \tCost: N/A (Too many holes for exact solution) | Time: N/A | Peak Memory: N/A")
        # 값이 없을 경우 결측치(nan)로 기록
        hk_cost, hk_time, hk_mem = float('nan'), float('nan'), float('nan')
        benchmark_results.append(["Held-Karp (Optimal)", hk_cost, hk_time, hk_mem])
    else:
        start_t = time.perf_counter()
        tracemalloc.start()
        hk_path, hk_cost = held_karp(holes)
        hk_time = time.perf_counter() - start_t
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        hk_mem = mem_peak / 1024
        print(f"[Held-Karp (Optimal)] \tCost: {hk_cost:7.2f} | Time: {hk_time:.4f}s | Peak Memory: {hk_mem:.2f} KB")
        benchmark_results.append(["Held-Karp (Optimal)", hk_cost, hk_time, hk_mem])

    # 5. Proposed method
    start_t = time.perf_counter()
    tracemalloc.start()
    proposed_path = proposed_h_dp(holes,num_portals=10)
    proposed_cost = calculate_path_cost(proposed_path, holes)
    proposed_time = time.perf_counter() - start_t
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    proposed_mem = mem_peak / 1024
    print(f"[Proposed Method] \tCost: {proposed_cost:7.2f} | Time: {proposed_time:.4f}s | Peak Memory: {proposed_mem:.2f} KB")
    benchmark_results.append(["Proposed Method", proposed_cost, proposed_time, proposed_mem])
    
     # 6. Mathematical Lower Bound (1-Tree)
    best_upper = min(greedy_cost, tb_cost, sa_cost, proposed_cost) # 수렴을 돕는 Upper Bound
    start_t = time.perf_counter()
    tracemalloc.start()
    # 1000번 반복 최적화를 통해 하한선을 찾습니다.
    lb_cost = held_karp_lower_bound(holes, upper_bound=best_upper, max_iter=1000)
    lb_time = time.perf_counter() - start_t
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lb_mem = mem_peak / 1024
    print(f"[1-Tree Lower Bound] \tCost: {lb_cost:7.2f} | Time: {lb_time:.4f}s | Peak Memory: {lb_mem:.2f} KB")
    benchmark_results.append(["1-Tree Lower Bound", lb_cost, lb_time, lb_mem])

    # CSV 파일로 결과 저장 호출 (TOOL_COUNTS 문자열 추가)
    # save_results_to_csv(NUM_HOLES, WIDTH, HEIGHT, str(TOOL_TYPES), tool_counts_str, benchmark_results)

    # ==========================================
    # 플롯 그리기
    # ==========================================
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    # fig.suptitle("TCP-TSP Algorithm Benchmark Paths", fontsize=16, fontweight='bold')

    # # 각 알고리즘별 서브플롯 그리기
    # plot_path(axes[0, 0], holes, greedy_path, "1. Greedy Algorithm", greedy_cost, greedy_time)
    # plot_path(axes[0, 1], holes, tb_path,     "2. Tool-Bundled 2-Opt", tb_cost, tb_time)
    # plot_path(axes[1, 0], holes, sa_path,     "3. State-Aware 2-Opt", sa_cost, sa_time)
    # # Held-karp는 NUM_HOLES가 커서 작동 안 할 경우 오류가 날 수 있으므로 Proposed로 대체
    # plot_path(axes[1, 1], holes, proposed_path, "5. Proposed Method", proposed_cost, proposed_time)

    # # 화면에 출력
    # plt.show()
    
    
if __name__ == "__main__":
    N_list = [100]
    for N in N_list:
        print(f"\n=== Running benchmark with {N} holes ===")
        NUM_HOLES = N
        main_function()