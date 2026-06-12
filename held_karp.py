from Utilities import get_cost

def held_karp(holes):
    """
    Held-Karp 동적 계획법 (Optimal Baseline).
    비용 행렬(Cost Matrix)에 이미 공구 교체 페널티가 포함되어 있으므로,
    DP 상태 공간을 탐색하며 공구 유지와 동선 단축 간의 최적 트레이드오프를 찾아냅니다.
    """
    n = len(holes)
    # 거리 대신 '비용(Cost)' 행렬을 미리 계산하여 상태 의존적 성질을 반영
    cost_matrix = [[get_cost(holes[i], holes[j]) for j in range(n)] for i in range(n)]
    
    # memo: (방문한 노드 비트마스크, 마지막 방문 노드) -> 최소 비용
    memo = {}

    def solve(mask, last):
        # 모든 노드를 방문한 경우 (시작점 0번 노드로 귀환하는 비용 반환)
        if mask == (1 << n) - 1:
            return cost_matrix[last][0]

        if (mask, last) in memo:
            return memo[(mask, last)]

        min_cost = float('inf')
        for i in range(n):
            # i번 노드를 아직 방문하지 않은 경우
            if not (mask & (1 << i)):
                new_cost = cost_matrix[last][i] + solve(mask | (1 << i), i)
                min_cost = min(min_cost, new_cost)

        memo[(mask, last)] = min_cost
        return min_cost

    # 최적 경로 역추적 로직
    optimal_cost = solve(1, 0)
    
    mask = 1
    last = 0
    path = [0]
    
    while mask != (1 << n) - 1:
        next_node = -1
        best_val = float('inf')
        for i in range(n):
            if not (mask & (1 << i)):
                val = cost_matrix[last][i] + solve(mask | (1 << i), i)
                if val < best_val:
                    best_val = val
                    next_node = i
        path.append(next_node)
        mask |= (1 << next_node)
        last = next_node

    return path, optimal_cost