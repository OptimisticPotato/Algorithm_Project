import math
import time
from utilities import *

def calculate_total_distance(path):
    """주어진 순회 경로(Path)의 총 유클리드 이동 거리를 계산하는 함수"""
    distance = 0.0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        distance += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distance
    
def run_2opt_heuristic(initial_path):
    """
    ========================================================================
    [2-Opt Heuristic 알고리즘 설명]
    
    1. 핵심 원리 (Local Search):
       - 현재 경로에서 임의의 두 간선 (i-1 -> i)와 (j -> j+1)을 선택합니다.
       - 만약 두 간선이 서로 교차하거나 우회하여 비효율적이라면, 해당 간선을 끊습니다.
       - i부터 j까지의 하위 경로 순서를 '역순(Reverse)'으로 뒤집어 재연결(Swap)합니다.
       - 이 과정을 통해 경로가 '꼬여 있는 부분'이 풀리며 기하학적 거리가 단축됩니다.
       
    2. 시간 복잡도 (Time Complexity):
       - 두 정점 i, j를 고르는 조합의 수로 인해 1회 반복(Iteration)당 O(N^2)이 소요됩니다.
       - 더 이상 경로 개선이 일어나지 않는 국소 최적해(Local Optimum)에 도출할 때까지 
         이 O(N^2) 탐색을 지속적으로 반복(Iterative loop)합니다.
         
    3. 본 공정 시뮬레이션에서의 규칙:
       - 출발지이자 도착지인 path[0]과 path[-1]은 공구의 원점 (0,0)으로 고정됩니다.
       - 따라서 인덱스 i와 j는 원점(0,0)을 훼손하지 않는 범위(1 ~ N) 내에서만 스왑을 수행합니다.
    ========================================================================
    """
    
    best_path = list(initial_path)
    best_dist = calculate_total_distance(best_path)
    
    improved = True
    iteration_count = 0
    
    start_time = time.time()

    # 더 이상 거리 단축(개선)이 일어나지 않을 때까지 반복 (Local Optimum 수렴 조건)
    while improved:
        improved = False
        iteration_count += 1
        
        # 원점(Index 0)은 고정되므로, 첫 번째 간선은 인덱스 1부터 탐색
        for i in range(1, len(best_path) - 2):
            # 두 번째 간선은 i 뒤쪽부터 탐색 (원점 복귀 인덱스인 len-1 전까지)
            for j in range(i + 1, len(best_path) - 1):
                
                # [2-Opt Swap 수행]
                # 0부터 i-1까지 유지 + i부터 j까지 역순 뒤집기 + j+1부터 끝까지 유지
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                new_dist = calculate_total_distance(new_path)
                
                # 만약 뒤집은 경로가 기존보다 거리가 단축되었다면 즉시 업데이트
                if new_dist < best_dist:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True # 개선 플래그 활성화하여 다음 전체 루프 다시 수행
                    break
            if improved:
                break
                
    end_time = time.time()
    print("-" * 50)
    print(f"[2-Opt] 최적화된 총 거리: {best_dist:.2f} mm")
    print(f"[2-Opt] 반복 횟수: {iteration_count}")
    print_time(start_time, end_time)
    
    return best_path, best_dist, iteration_count