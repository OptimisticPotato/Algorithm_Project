from Utilities import get_distance, calculate_path_cost

def tool_bundled_two_opt(holes):
    """
    Baseline 1: 공구별 묶음 2-Opt (분할 정복 방식)
    """
    if len(holes) <= 1:
        return [0]

    # 1. 공구별로 노드의 '배열 인덱스(idx)' 분리
    # h.id 대신 절대적인 배열 인덱스를 사용하여 ID 중복/불일치 에러를 원천 차단합니다.
    tools = {}
    for idx in range(1, len(holes)):
        tools.setdefault(holes[idx].tool, []).append(idx)
        
    start_tool = holes[0].tool
    
    # 2. 단일 공구 그룹 내부에서만 거리를 최적화하는 헬퍼 함수
    def optimize_group(group_nodes):
        if len(group_nodes) <= 2:
            return group_nodes
        
        curr_path = group_nodes[:]
        improved = True
        while improved:
            improved = False
            for i in range(len(curr_path) - 1):
                for j in range(i + 2, len(curr_path) + 1):
                    new_p = curr_path[:i] + curr_path[i:j][::-1] + curr_path[j:]
                    
                    # 배열 인덱스를 통해 holes 객체에 접근
                    old_dist = sum(get_distance(holes[curr_path[k]], holes[curr_path[k+1]]) for k in range(len(curr_path)-1))
                    new_dist = sum(get_distance(holes[new_p[k]], holes[new_p[k+1]]) for k in range(len(new_p)-1))
                    
                    if new_dist < old_dist - 1e-6:
                        curr_path = new_p
                        improved = True
                        break
                if improved: 
                    break
        return curr_path

    # 3. 최적화된 그룹들을 최종 경로에 하나로 병합
    final_path = [0]
    
    if start_tool in tools:
        start_group = tools.pop(start_tool)
        final_path.extend(optimize_group(start_group))
        
    for tool_type, group in tools.items():
        final_path.extend(optimize_group(group))
        
    # 안전 검사 (디버깅 메시지 강화)
    if len(set(final_path)) != len(holes):
        raise AssertionError(f"에러: Tool Bundled 최적화 중 점이 누락되었습니다. (예상 노드 수: {len(holes)}, 실제 고유 노드 수: {len(set(final_path))})")
    
    return final_path


def state_aware_two_opt(holes, initial_path):
    """
    Baseline 2: 상태 추적형 2-Opt
    """
    path = initial_path[:]
    improved = True
    
    while improved:
        improved = False
        best_cost = calculate_path_cost(path, holes)
        
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path) + 1):
                if j - i <= 1: continue
                if i == 1 and j == len(path): continue
                
                # 슬라이싱을 이용한 부분 경로 반전
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                
                # '전체 비용 (거리 + 페널티)'을 다시 계산
                new_cost = calculate_path_cost(new_path, holes)
                
                if new_cost < best_cost - 1e-6:
                    path = new_path
                    best_cost = new_cost
                    improved = True
                    break 
            if improved:
                break
                
    # 안전 검사
    if len(set(path)) != len(holes):
        raise AssertionError(f"에러: State Aware 최적화 중 점이 누락되었습니다. (예상 노드 수: {len(holes)}, 실제 고유 노드 수: {len(set(path))})")
    
    return path