import math
import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
from utilities import *
import time

# =====================================================================
# 1. 공통 유틸리티 함수
# =====================================================================
def calc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# =====================================================================
# 2. [Base Solver] 원점 복귀를 인지하는 만능 Held-Karp DP
# =====================================================================
def solve_base_dp(nodes, start_node=None, end_node=None, depot=None):
    """
    depot이 주어지면 (최상위 계층), 원점에서 출발하고 복귀하는 비용까지 
    DP 점화식에 포함하여 가장 완벽한 형태를 찾아냅니다.
    """
    n = len(nodes)
    if n == 1:
        dist_only = 0.0
        if depot: dist_only = calc_dist(depot, nodes[0]) * 2
        return [nodes[0]], dist_only

    dist = [[calc_dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    
    start_idx = nodes.index(start_node) if start_node else None
    end_idx = nodes.index(end_node) if end_node else None
    
    memo = {}
    
    if start_idx is not None:
        memo[(1 << start_idx, start_idx)] = (0.0, -1)
    else:
        for i in range(n):
            if i != end_idx or n == 1:
                # Depot이 주어졌다면 원점에서 i번 노드로 진입하는 비용을 초깃값으로 설정
                cost = calc_dist(depot, nodes[i]) if depot else 0.0
                memo[(1 << i, i)] = (cost, -1)
                
    for r in range(2, n + 1):
        for subset in combinations(range(n), r):
            if start_idx is not None and (start_idx not in subset):
                continue
                
            mask = sum(1 << bit for bit in subset)
            
            for next_node in subset:
                if start_idx is not None and next_node == start_idx:
                    continue
                if end_idx is not None and r < n and next_node == end_idx:
                    continue
                    
                prev_mask = mask & ~(1 << next_node)
                min_cost = float('inf')
                min_prev_node = -1
                
                for prev_node in subset:
                    if prev_node == next_node or (prev_mask & (1 << prev_node)) == 0:
                        continue
                        
                    if (prev_mask, prev_node) in memo:
                        cost = memo[(prev_mask, prev_node)][0] + dist[prev_node][next_node]
                        if cost < min_cost:
                            min_cost = cost
                            min_prev_node = prev_node
                            
                if min_cost != float('inf'):
                    memo[(mask, next_node)] = (min_cost, min_prev_node)
                    
    final_mask = (1 << n) - 1
    min_total_cost = float('inf')
    best_last_node = -1
    
    if end_idx is not None:
        if (final_mask, end_idx) in memo:
            min_total_cost = memo[(final_mask, end_idx)][0]
            best_last_node = end_idx
    else:
        for i in range(n):
            if (final_mask, i) in memo:
                cost = memo[(final_mask, i)][0]
                # Depot이 주어졌다면 마지막 노드에서 원점으로 복귀하는 비용 추가
                if depot: cost += calc_dist(nodes[i], depot)
                if cost < min_total_cost:
                    min_total_cost = cost
                    best_last_node = i
                    
    if best_last_node == -1:
        return [], float('inf')
        
    path = []
    curr_mask = final_mask
    curr_node = best_last_node
    
    while curr_node != -1:
        path.append(nodes[curr_node])
        next_node = memo[(curr_mask, curr_node)][1]
        curr_mask &= ~(1 << curr_node)
        curr_node = next_node
        
    path.reverse()
    return path, min_total_cost

# =====================================================================
# 3. [Cluster Helpers] Portal Selection (동일)
# =====================================================================
def min_distance_pair(cluster_A, cluster_B):
    min_dist = float('inf')
    best_u, best_v = None, None
    for u in cluster_A:
        for v in cluster_B:
            d = calc_dist(u, v)
            if d < min_dist:
                min_dist = d
                best_u, best_v = u, v
    return best_u, best_v, min_dist

def select_portal_candidates(cluster, other_cluster, k):
    if not other_cluster: return list(cluster)
    scored = []
    for p in cluster:
        min_dist = min(calc_dist(p, q) for q in other_cluster)
        scored.append((min_dist, p))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored[:k]] or list(cluster)

def solve_cluster_path(nodes, start_node, end_node, max_M, portal_k, cache):
    nodes_key = tuple(sorted(nodes))
    cache_key = (nodes_key, start_node, end_node, max_M, portal_k)
    if cache_key in cache: return cache[cache_key]

    # 하위 군집 계산 시에는 depot을 넘기지 않음 (전체 DP에서 처리하므로)
    if len(nodes) <= max_M:
        path, total_dist = solve_base_dp(nodes, start_node=start_node, end_node=end_node)
        cache[cache_key] = (path, total_dist)
        return path, total_dist

    path = solve_recursive_hierarchy(nodes, start_node=start_node, end_node=end_node, max_M=max_M, portal_k=portal_k, cache=cache)
    total_dist = sum(calc_dist(path[i], path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0.0
    cache[cache_key] = (path, total_dist)
    return path, total_dist

# =====================================================================
# 4. [Cluster Order DP] 원점 거리를 반영하는 군집 스케줄링
# =====================================================================
def solve_cluster_order(dist_matrix, start_idx=None, end_idx=None, depot_dists=None):
    n = len(dist_matrix)
    if n == 0: return []
    if n == 1: return [0]

    memo = {}
    if start_idx is not None:
        memo[(1 << start_idx, start_idx)] = (0.0, -1)
    else:
        for i in range(n):
            if i != end_idx or n == 1:
                cost = depot_dists[i] if depot_dists else 0.0
                memo[(1 << i, i)] = (cost, -1)

    for r in range(2, n + 1):
        for subset in combinations(range(n), r):
            if start_idx is not None and (start_idx not in subset): continue
            mask = sum(1 << bit for bit in subset)

            for next_node in subset:
                if start_idx is not None and next_node == start_idx: continue
                if end_idx is not None and r < n and next_node == end_idx: continue

                prev_mask = mask & ~(1 << next_node)
                min_cost = float('inf')
                min_prev = -1

                for prev_node in subset:
                    if prev_node == next_node or (prev_mask & (1 << prev_node)) == 0: continue
                    if (prev_mask, prev_node) not in memo: continue

                    cost = memo[(prev_mask, prev_node)][0] + dist_matrix[prev_node][next_node]
                    if cost < min_cost:
                        min_cost = cost
                        min_prev = prev_node

                if min_cost != float('inf'):
                    memo[(mask, next_node)] = (min_cost, min_prev)

    final_mask = (1 << n) - 1
    min_total_cost = float('inf')
    best_last = -1

    if end_idx is not None:
        if (final_mask, end_idx) in memo:
            min_total_cost = memo[(final_mask, end_idx)][0]
            best_last = end_idx
    else:
        for i in range(n):
            if (final_mask, i) in memo:
                cost = memo[(final_mask, i)][0]
                if depot_dists: cost += depot_dists[i]
                if cost < min_total_cost:
                    min_total_cost = cost
                    best_last = i

    if best_last == -1: return []

    order = []
    curr_mask = final_mask
    curr_node = best_last

    while curr_node != -1:
        order.append(curr_node)
        next_node = memo[(curr_mask, curr_node)][1]
        curr_mask &= ~(1 << curr_node)
        curr_node = next_node

    order.reverse()
    return order

# =====================================================================
# 5. [Main Recursion] Generalized TSP with Depot Awareness
# =====================================================================
def solve_recursive_hierarchy(nodes, start_node=None, end_node=None, max_M=15, portal_k=3, cache=None, depot=None):
    n = len(nodes)
    if n == 0: return []
    if cache is None: cache = {}

    if n <= max_M:
        path, _ = solve_base_dp(nodes, start_node=start_node, end_node=end_node, depot=depot)
        return path

    K = min(max_M, n)
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(nodes) # KMeans로 군집화하여 노드들을 K개의 군집으로 분할

    clusters_dict = {i: [] for i in range(K)}
    for idx, label in enumerate(kmeans.labels_):
        clusters_dict[label].append(nodes[idx])
        
    # 빈 군집 제거 (안전 장치)
    valid_clusters = [c for c in clusters_dict.values() if c]
    K = len(valid_clusters)
    
    centroids = []
    for c in valid_clusters:
        centroids.append((sum(p[0] for p in c)/len(c), sum(p[1] for p in c)/len(c)))
        
    clusters = {i: valid_clusters[i] for i in range(K)}

    start_c_idx = None
    if start_node:
        dists = [calc_dist(start_node, c) for c in centroids]
        start_c_idx = dists.index(min(dists))
        
    end_c_idx = None
    if end_node:
        dists = [calc_dist(end_node, c) for c in centroids]
        end_c_idx = dists.index(min(dists))

    if start_c_idx is not None and end_c_idx is not None and start_c_idx == end_c_idx and K > 1:
        end_c_idx = None

    # [핵심] 최상위 계층일 경우 원점(Depot)과의 거리를 계산하여 Macro-Routing에 전달
    depot_dists = None
    if depot and not start_node and not end_node:
        depot_dists = [min(calc_dist(depot, p) for p in clusters[i]) for i in range(K)]

    # 거리 행렬 생성
    dist_matrix = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            _, _, d = min_distance_pair(clusters[i], clusters[j])
            dist_matrix[i][j] = dist_matrix[j][i] = d

    ordered_c_indices = solve_cluster_order(dist_matrix, start_idx=start_c_idx, end_idx=end_c_idx, depot_dists=depot_dists)
    if not ordered_c_indices: return []

    entry_candidates, exit_candidates = {}, {}
    for i, c_idx in enumerate(ordered_c_indices):
        cluster_nodes = clusters[c_idx]

        # 첫 군집 진입 후보 설정 (Depot 인지)
        if i == 0:
            if start_node: entry_list = [start_node]
            elif depot: entry_list = select_portal_candidates(cluster_nodes, [depot], portal_k)
            else: entry_list = list(cluster_nodes)
        else:
            prev_idx = ordered_c_indices[i - 1]
            entry_list = select_portal_candidates(cluster_nodes, clusters[prev_idx], portal_k)

        # 마지막 군집 진출 후보 설정 (Depot 인지)
        if i == len(ordered_c_indices) - 1:
            if end_node: exit_list = [end_node]
            elif depot: exit_list = select_portal_candidates(cluster_nodes, [depot], portal_k)
            else: exit_list = list(cluster_nodes)
        else:
            next_idx = ordered_c_indices[i + 1]
            exit_list = select_portal_candidates(cluster_nodes, clusters[next_idx], portal_k)

        entry_candidates[c_idx] = list(dict.fromkeys(entry_list))
        exit_candidates[c_idx] = list(dict.fromkeys(exit_list))

    # 하위 군집 DP 미리 계산
    cluster_costs = {}
    for c_idx in ordered_c_indices:
        sub_nodes = clusters[c_idx]
        for entry in entry_candidates[c_idx]:
            for exit in exit_candidates[c_idx]:
                if entry == exit and len(sub_nodes) > 1: continue
                path, total_dist = solve_cluster_path(sub_nodes, start_node=entry, end_node=exit, max_M=max_M, portal_k=portal_k, cache=cache)
                if path: cluster_costs[(c_idx, entry, exit)] = (path, total_dist)

    # [핵심] Portal 조합을 맞추는 최상위 DP (원점 진입 비용 포함)
    first_idx = ordered_c_indices[0]
    dp, prev_map, first_prev = {}, [], {}

    for entry in entry_candidates[first_idx]:
        for exit in exit_candidates[first_idx]:
            key = (first_idx, entry, exit)
            if key not in cluster_costs: continue
            
            cost = cluster_costs[key][1]
            if depot and not start_node: cost += calc_dist(depot, entry)
            
            if exit not in dp or cost < dp[exit]:
                dp[exit] = cost
                first_prev[exit] = (None, entry)

    prev_map.append(first_prev)

    for i in range(1, len(ordered_c_indices)):
        c_idx = ordered_c_indices[i]
        new_dp, new_prev = {}, {}

        for prev_exit, prev_cost in dp.items():
            for entry in entry_candidates[c_idx]:
                cross_dist = calc_dist(prev_exit, entry)
                for exit in exit_candidates[c_idx]:
                    key = (c_idx, entry, exit)
                    if key not in cluster_costs: continue
                    cost = prev_cost + cross_dist + cluster_costs[key][1]
                    if exit not in new_dp or cost < new_dp[exit]:
                        new_dp[exit] = cost
                        new_prev[exit] = (prev_exit, entry)

        dp = new_dp
        prev_map.append(new_prev)

    if not dp: return []

    # [핵심] 원점 복귀 비용을 고려한 최종 Exit 선정
    best_exit = None
    min_final_cost = float('inf')
    for exit, cost in dp.items():
        final_cost = cost
        if depot and not end_node: final_cost += calc_dist(exit, depot)
        if final_cost < min_final_cost:
            min_final_cost = final_cost
            best_exit = exit

    chosen_entries, chosen_exits = {}, {}
    exit_node = best_exit

    for i in range(len(ordered_c_indices) - 1, -1, -1):
        c_idx = ordered_c_indices[i]
        prev_exit, entry = prev_map[i][exit_node]
        chosen_entries[c_idx] = entry
        chosen_exits[c_idx] = exit_node
        exit_node = prev_exit

    final_path = []
    for c_idx in ordered_c_indices:
        entry = chosen_entries[c_idx]
        exit = chosen_exits[c_idx]
        path = cluster_costs[(c_idx, entry, exit)][0]
        if path: final_path.extend(path)

    return final_path

# =====================================================================
# 래퍼 함수 (최초 실행)
# =====================================================================
def optimize_drilling_path(hole_positions, max_M=15, portal_k=3):
    depot = (0.0, 0.0)

    start_time = time.time()    
    
    # [수정] 최상위 계층에 depot 정보를 전달하여 완벽한 Closed-Loop 최적화 유도
    pure_optimal_path = solve_recursive_hierarchy(
        hole_positions,
        start_node=None,
        end_node=None,
        max_M=max_M,
        portal_k=portal_k,
        cache={},
        depot=depot
    )
    
    # DP가 스스로 원점(Depot)과의 연결 비용을 계산했으므로 reverse 할 필요 없음
    full_path = [depot] + pure_optimal_path + [depot]
    total_dist = sum(calc_dist(full_path[i], full_path[i+1]) for i in range(len(full_path)-1))
    end_time = time.time()
    
    print("-" * 50)
    print(f"[Proposed] 최적화 완료! 총 이동 거리: {total_dist:.2f} mm")
    print_time(start_time, end_time) # 사용하시는 환경에 맞춰 주석 해제
    return full_path, total_dist