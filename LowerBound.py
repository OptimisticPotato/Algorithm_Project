import math
from Utilities import get_cost

def compute_1_tree(dist, pi, n):
    """
    주어진 페널티(pi)가 반영된 가중치를 사용하여 1-Tree와 그 비용을 계산합니다.
    1-Tree: 0번 노드를 제외한 나머지 노드들의 최소 신장 트리(MST) + 0번 노드에 연결된 가장 싼 2개의 간선
    """
    key = [float('inf')] * n
    parent = [-1] * n
    in_mst = [False] * n

    # 1번 노드를 MST의 시작점으로 설정
    key[1] = 0.0
    
    # 1~n-1 번 노드들에 대해 Prim 알고리즘 수행
    for _ in range(n - 1):
        min_k = float('inf')
        u = -1
        for i in range(1, n):
            if not in_mst[i] and key[i] < min_k:
                min_k = key[i]
                u = i

        if u == -1: break
        in_mst[u] = True

        for v in range(1, n):
            if not in_mst[v]:
                # 간선 가중치에 라그랑주 승수(pi)를 더함
                w_mod = dist[u][v] + pi[u] + pi[v]
                if w_mod < key[v]:
                    key[v] = w_mod
                    parent[v] = u

    # 0번(Depot) 노드에서 출발하는 가장 싼 2개의 간선 탐색
    min1, min2 = float('inf'), float('inf')
    v1, v2 = -1, -1
    for i in range(1, n):
        w_mod = dist[0][i] + pi[0] + pi[i]
        if w_mod < min1:
            min2 = min1
            v2 = v1
            min1 = w_mod
            v1 = i
        elif w_mod < min2:
            min2 = w_mod
            v2 = i

    # 차수(Degree) 및 트리 전체 비용 계산
    degree = [0] * n
    degree[0] = 2
    degree[v1] += 1
    degree[v2] += 1
    
    tree_cost = dist[0][v1] + dist[0][v2]
    
    for i in range(1, n):
        p = parent[i]
        if p != -1:
            degree[i] += 1
            degree[p] += 1
            tree_cost += dist[i][p]

    # 라그랑주 하한선(L) 계산: C(T) + sum(pi[i] * (degree[i] - 2))
    lb_val = tree_cost
    for i in range(n):
        lb_val += pi[i] * (degree[i] - 2)

    return lb_val, degree

def held_karp_lower_bound(holes, upper_bound, max_iter=1000):
    """
    Subgradient Optimization 기법을 사용하여 TSP의 절대 하한선(Lower Bound)을 도출합니다.
    """
    n = len(holes)
    if n <= 2: return 0.0

    # 거리(비용) 행렬 사전 계산 (속도 최적화)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = get_cost(holes[i], holes[j])
            dist[i][j] = d
            dist[j][i] = d

    pi = [0.0] * n # 각 노드의 페널티
    max_lower_bound = 0.0
    
    # Subgradient 파라미터
    lam = 2.0
    period = max(1, max_iter // 20)
    no_improve_count = 0

    for it in range(max_iter):
        lb_val, degree = compute_1_tree(dist, pi, n)
        
        if lb_val > max_lower_bound:
            max_lower_bound = lb_val
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= period:
            lam /= 2.0
            no_improve_count = 0
            
        # 모든 노드의 차수가 2라면, 그것은 곧 100% 최적 TSP 경로(Optimal Tour)를 의미!
        sum_sq_diff = sum((d - 2)**2 for d in degree)
        if sum_sq_diff == 0:
            max_lower_bound = lb_val
            break
            
        if lam < 1e-5:
            break
            
        # 서브그레디언트 상승법을 이용한 페널티(pi) 업데이트
        step_size = lam * (upper_bound - lb_val) / sum_sq_diff
        for i in range(n):
            pi[i] += step_size * (degree[i] - 2)

    return max_lower_bound