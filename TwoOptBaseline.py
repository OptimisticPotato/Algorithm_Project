from Utilities import get_distance, calculate_path_cost

def tool_bundled_two_opt(holes):
    """
    Baseline 1: 공구별 묶음 2-Opt (현업 관행 모사)
    공구 종류가 같은 홀끼리 먼저 묶은 뒤, 그 묶음 '내부'에서만 2-Opt를 수행합니다.
    동선을 낭비하더라도 공구 교체 횟수를 강제로 최소화하는 보수적 전략입니다.
    """
    # 1. 공구별로 노드 그룹화
    tools = {}
    for h in holes:
        tools.setdefault(h.tool, []).append(h.id)
        
    path = []
    for tool_type, group in tools.items():
        path.extend(group) # 예: [0, 2, 4 (M4)] + [1, 3, 5 (M8)]
        
    # 2. 공구 그룹 내부에서만 순수 거리 기반 2-Opt 수행
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue
                
                n1, n2 = path[i-1], path[i]
                n3, n4 = path[j-1], path[j % len(path)]
                
                # 공구 경계를 넘어서는 스왑(Swap)은 원천 차단
                if holes[n1].tool != holes[n2].tool or holes[n3].tool != holes[n4].tool:
                    continue
                if holes[n2].tool != holes[n3].tool: # 뒤집히는 구간 전체가 같은 공구여야 함
                    continue
                
                # '순수 물리적 거리'만을 기준으로 교환 이득 평가
                old_dist = get_distance(holes[n1], holes[n2]) + get_distance(holes[n3], holes[n4])
                new_dist = get_distance(holes[n1], holes[n3]) + get_distance(holes[n2], holes[n4])
                
                if new_dist < old_dist:
                    path[i:j] = list(reversed(path[i:j]))
                    improved = True
    return path


def state_aware_two_opt(holes, initial_path):
    """
    Baseline 2: 상태 추적형 2-Opt
    경로를 뒤집을 때마다 공구 순서가 꼬이는 것을 감지하기 위해, 
    매 교환 시마다 전체 경로의 Cost(거리 + 누적 T_ATC)를 다시 시뮬레이션합니다.
    연산량이 O(N^3)으로 폭발하며 쉽게 Local Minima에 빠집니다.
    """
    path = initial_path[:]
    improved = True
    
    while improved:
        improved = False
        # 현재 경로의 총 '비용(거리+페널티)' 계산
        best_cost = calculate_path_cost(path, holes)
        
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue
                
                # 경로 부분 반전 (Reverse segment)
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                
                # O(1) 간선 계산을 포기하고, 전체 사이클 타임을 다시 평가
                new_cost = calculate_path_cost(new_path, holes)
                
                if new_cost < best_cost:
                    path = new_path
                    best_cost = new_cost
                    improved = True
                    break # while 루프 재시작을 위해 탈출
            if improved:
                break
                
    return path