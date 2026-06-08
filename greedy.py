import math
import time
from utilities import *

def calculate_greedy_tsp(holes):
    """
    Greedy (Nearest Neighbor) 알고리즘을 사용한 TSP 경로 및 총 거리를 계산합니다.
    1. 공구는 (0,0)에서 출발
    2. 모든 홀을 탐욕적으로 방문
    3. 다시 (0,0)으로 복귀
    """
    start_time = time.time()
    start_point = (0.0, 0.0)
    unvisited = list(holes)  # 아직 방문하지 않은 홀들의 목록
    current_pos = start_point
    
    path = [start_point]     # 이동 경로 저장
    total_distance = 0.0
    
    # 모든 홀을 방문할 때까지 반복
    while unvisited:
        nearest_idx = -1
        min_dist = float('inf')
        
        # 현재 위치에서 가장 가까운 다음 홀 검색 (Pure Greedy)
        for i, (hx, hy) in enumerate(unvisited):
            dist = math.sqrt((current_pos[0] - hx)**2 + (current_pos[1] - hy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                
        # 가장 가까운 홀 선택 및 이동 등록
        next_hole = unvisited.pop(nearest_idx)
        total_distance += min_dist
        path.append(next_hole)
        current_pos = next_hole  # 현재 위치 업데이트
        
    # 마지막 홀에서 다시 출발지(0,0)로 복귀 비용 추가
    return_dist = math.sqrt((current_pos[0] - start_point[0])**2 + (current_pos[1] - start_point[1])**2)
    total_distance += return_dist
    path.append(start_point)
    end_time = time.time()
    
    print("-" * 50)
    print(f"[Greedy] 총 이동 거리: {total_distance:.2f} mm")
    print_time(start_time, end_time)
    
    return path, total_distance