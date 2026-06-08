from utilities import *
from greedy import calculate_greedy_tsp
from two_opt import run_2opt_heuristic
from held_karp import run_held_karp
from proposed import optimize_drilling_path

import time

N = 100            # 생성할 홀의 개수
x_max = 100.0      # 가로 크기
y_max = 100.0      # 세로 크기
r_hole = 2.0      # 홀 반지름

use_predefined_holes = False  # True로 설정하면 14개의 사전 정의된 홀 좌표를 사용합니다. False로 설정하면 무작위 홀을 생성합니다.

if __name__ == "__main__":

    if use_predefined_holes:
        hole_positions = [
    (10.0, 10.0),     # Hole 1
    (10.0, 50.0),     # Hole 2
    (18.0, 53.5),     # Hole 3
    (18.0, 42.5),     # Hole 4
    (32.32, 12.66),   # Hole 5
    (37.71, 26.40),   # Hole 6
    (37.71, 43.60),   # Hole 7
    (62.29, 43.60),   # Hole 8
    (62.29, 26.40),   # Hole 9
    (80.0, 10.0),     # Hole 10
    (82.0, 16.5),     # Hole 11
    (82.0, 27.5),     # Hole 12
    (72.59, 53.5),    # Hole 13
    (90.0, 55.75)     # Hole 14
]
    else:
        hole_positions = generate_hole_coordinates(N, x_max, y_max, r_hole)
    
    # Greedy TSP 알고리즘 실행
    greedy_path, greedy_distance = calculate_greedy_tsp(hole_positions)

    # 2-Opt
    initial_path = [(0.0, 0.0)] + hole_positions + [(0.0, 0.0)]
    two_opt_path, two_opt_distance, iteration_count = run_2opt_heuristic(initial_path)

    # Held-Karp
    if N <= 30:  # Held-Karp 제한
        held_karp_path, held_karp_distance = run_held_karp(hole_positions)
    else:
        print(f"[Held-Karp] N={N}은(는) 너무 커서 Held-Karp 알고리즘을 실행하지 않습니다. (최대 허용 N=30)")
        held_karp_path, held_karp_distance = None, None
    
    proposed_path, proposed_distance = optimize_drilling_path(hole_positions)
    
    
    visualize_four_paths(
    hole_positions=hole_positions,
    greedy_path=greedy_path,
    two_opt_path=two_opt_path,
    held_karp_path=held_karp_path, # 만약 N이 너무 커서 안 돌렸다면 [] 를 넣으시면 됩니다.
    proposed_path=proposed_path
    )