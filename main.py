import time
from Utilities import *
from GreedyAlgorithm import greedy_tsp
from Held_Karp import held_karp
from TwoOptBaseline import *

def main():
    print("=== State-Dependent Cost TSP (TCP-TSP) Benchmark ===")
    
    # 벤치마크를 위해 Held-Karp가 돌아갈 수 있는 수준(약 12~14개)의 홀 생성
    holes = generate_random_holes(num_holes=12, width=100, height=100, tool_types=['M4', 'M8'])
    print(f"[Info] Generated {len(holes)} holes with different tools.\n")

    # 1. Greedy Algorithm
    start_t = time.time()
    greedy_path = greedy_tsp(holes)
    greedy_cost = calculate_path_cost(greedy_path, holes)
    greedy_time = time.time() - start_t
    print(f"[Greedy] \t\tCost: {greedy_cost:7.2f} | Time: {greedy_time:.4f}s")

    # 2. Tool-Bundled 2-Opt (현업 묶음 가공 모사)
    start_t = time.time()
    tb_path = tool_bundled_two_opt(holes)
    tb_cost = calculate_path_cost(tb_path, holes)
    tb_time = time.time() - start_t
    print(f"[Tool-Bundled 2-Opt]\tCost: {tb_cost:7.2f} | Time: {tb_time:.4f}s")

    # 3. State-Aware 2-Opt (초기 경로는 Greedy 사용)
    start_t = time.time()
    sa_path = state_aware_two_opt(holes, greedy_path)
    sa_cost = calculate_path_cost(sa_path, holes)
    sa_time = time.time() - start_t
    print(f"[State-Aware 2-Opt] \tCost: {sa_cost:7.2f} | Time: {sa_time:.4f}s")

    # 4. Held-Karp (Optimal - 진정한 최적해 보장)
    start_t = time.time()
    hk_path, hk_cost = held_karp(holes)
    hk_time = time.time() - start_t
    print(f"[Held-Karp (Optimal)] \tCost: {hk_cost:7.2f} | Time: {hk_time:.4f}s")

    print("\n※ The proposed H-DP algorithm will be compared against these baselines.")

if __name__ == "__main__":
    main()