from Utilities import get_cost, get_distance

def greedy_tsp(holes, start_idx=0, debug=False):
    """
    비용(거리 + 공구 교체 페널티)을 기준으로 가장 가까운 홀을 탐욕적으로 선택합니다.
    명확한 평가 로직과 타이 브레이커(Tie-breaker)를 포함한 완전한 버전입니다.
    """
    # 0번(Depot)은 시작점이므로 방문할 목록에서 제외합니다.
    unvisited = set(range(1, len(holes))) 
    curr = start_idx
    path = [curr]
    
    if debug:
        print("\n--- [Greedy Algorithm Debug Log] ---")
        
    while unvisited:
        best_cost = float('inf')
        best_node = -1
        best_dist = float('inf')
        
        # 현재 위치에서 갈 수 있는 모든 미방문 홀들을 하나하나 꼼꼼하게 평가합니다.
        for nxt in unvisited:
            cost = get_cost(holes[curr], holes[nxt])
            dist = get_distance(holes[curr], holes[nxt])
            
            # 1순위: '총 비용(물리적 거리 + ATC 페널티)'이 가장 저렴한 노드 갱신
            if cost < best_cost:
                best_cost = cost
                best_node = nxt
                best_dist = dist
            # 2순위(타이 브레이커): 만약 비용이 완벽하게 같다면, 순수 거리가 더 짧은 노드를 선택
            elif cost == best_cost:
                if dist < best_dist:
                    best_node = nxt
                    best_dist = dist
                    
        # 내부적으로 어떻게 판단해서 이 노드를 골랐는지 로그 출력
        if debug:
            is_changed = (holes[curr].tool != holes[best_node].tool)
            tool_transition = f"{holes[curr].tool}->{holes[best_node].tool}"
            print(f"[{curr:2d} -> {best_node:2d}] Tool: {tool_transition} | Dist: {best_dist:6.2f} | Total Cost: {best_cost:6.2f} | ATC Changed: {is_changed}")
            
        # 최적의 노드로 이동 확정
        path.append(best_node)
        unvisited.remove(best_node)
        curr = best_node
        
    if debug:
        print("------------------------------------\n")
        
    return path