import math
import itertools
import time
from utilities import *

def calculate_distance_matrix(nodes):
    """모든 점들 간의 유클리드 거리를 사전 계산하여 2차원 배열로 반환합니다."""
    n = len(nodes)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = math.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
    return dist_matrix

def run_held_karp(hole_positions):
    
    # 0번 인덱스에 공구의 출발점(0.0, 0.0) 추가
    nodes = [(0.0, 0.0)] + hole_positions
    n = len(nodes)
    
    start_time = time.time()
    
    # 거리 행렬 사전 계산
    dist = calculate_distance_matrix(nodes)
    
    # Memoization 테이블 초기화
    memo = {}
    
    # [Base Case] 원점(0)에서 출발해 각 홀(i)을 처음 방문하는 비용
    for i in range(1, n):
        initial_mask = 1 | (1 << i) 
        memo[(initial_mask, i)] = (dist[0][i], 0)
        
    # [DP Step] 방문할 노드 개수를 늘려가며 탐색
    for r in range(2, n):
        for subset in itertools.combinations(range(1, n), r):
            bits = 1 
            for bit in subset:
                bits |= (1 << bit)
            
            for next_node in subset:
                prev_mask = bits & ~(1 << next_node)
                
                min_cost = float('inf')
                min_prev_node = -1
                
                for prev_node in subset:
                    if prev_node == next_node:
                        continue
                        
                    cost = memo[(prev_mask, prev_node)][0] + dist[prev_node][next_node]
                    if cost < min_cost:
                        min_cost = cost
                        min_prev_node = prev_node
                        
                memo[(bits, next_node)] = (min_cost, min_prev_node)
                
    # [Final Step] 원점으로 돌아오는 최소 비용 계산
    all_visited_mask = (1 << n) - 1
    min_total_cost = float('inf')
    last_node_before_return = -1
    
    for i in range(1, n):
        cost = memo[(all_visited_mask, i)][0] + dist[i][0]
        if cost < min_total_cost:
            min_total_cost = cost
            last_node_before_return = i
            
    # [경로 역추적]
    path = []
    curr_mask = all_visited_mask
    curr_node = last_node_before_return
    
    while curr_node != 0:
        path.append(curr_node)
        next_node = memo[(curr_mask, curr_node)][1]
        curr_mask &= ~(1 << curr_node)
        curr_node = next_node
        
    path.append(0)
    path.reverse()
    path.append(0)
    
    end_time = time.time()
    
    print("-" * 50)
    print(f"[Held-Karp] 총 이동 거리: {min_total_cost:.2f} mm")
    print_time(start_time, end_time)
    
    return path, min_total_cost