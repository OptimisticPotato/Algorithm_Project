import math
import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import time
from utilities import *

# =====================================================================
# 1. 공통 유틸리티 함수
# =====================================================================
def calc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calc_path_distance(path):
    if len(path) < 2: return 0.0
    return sum(calc_dist(path[i], path[i + 1]) for i in range(len(path) - 1))

# =====================================================================
# 2. [Base Solver] 원점 복귀를 인지하는 만능 Held-Karp DP
# =====================================================================
def solve_base_dp(nodes, start_node=None, end_node=None, depot=None):
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
                cost = calc_dist(depot, nodes[i]) if depot else 0.0
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
                min_prev_node = -1
                
                for prev_node in subset:
                    if prev_node == next_node or (prev_mask & (1 << prev_node)) == 0: continue
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
                if depot: cost += calc_dist(nodes[i], depot)
                if cost < min_total_cost:
                    min_total_cost = cost
                    best_last_node = i
                    
    if best_last_node == -1: return [], float('inf')
        
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
# 3. [Phase 1: Tree Builder] 일단 클러스터링을 최하위까지 완전히 진행
# =====================================================================
class ClusterTree:
    def __init__(self, nodes, max_M=15):
        self.nodes = nodes
        self.n = len(nodes)
        self.is_leaf = self.n <= max_M
        self.centroid = (sum(p[0] for p in nodes)/self.n, sum(p[1] for p in nodes)/self.n) if self.n > 0 else (0,0)
        self.children = []
        
        # 최하위(max_M 이하)가 될 때까지 무조건 재귀적으로 트리를 먼저 쪼갭니다.
        if not self.is_leaf and self.n > 0:
            K = min(max_M, self.n)
            kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(nodes)
            clusters_dict = {i: [] for i in range(K)}
            for idx, label in enumerate(kmeans.labels_):
                clusters_dict[label].append(nodes[idx])
                
            for child_nodes in clusters_dict.values():
                if child_nodes:
                    self.children.append(ClusterTree(child_nodes, max_M))

# =====================================================================
# 4. [Cluster Helpers] Portal Selection & Ordering
# =====================================================================
def min_distance_pair(nodes_A, nodes_B):
    min_dist = float('inf')
    for u in nodes_A:
        for v in nodes_B:
            d = calc_dist(u, v)
            if d < min_dist:
                min_dist = d
    return min_dist

def select_portal_candidates(cluster_nodes, other_nodes, k):
    if not other_nodes: return list(cluster_nodes)
    scored = []
    for p in cluster_nodes:
        min_dist = min(calc_dist(p, q) for q in other_nodes)
        scored.append((min_dist, p))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored[:k]] or list(cluster_nodes)

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
# 5. [Phase 2: Tree Solver] 계층적 포탈 생성 및 최하위 DP 병합
# =====================================================================
def solve_tree_hierarchy(tree_node, start_node=None, end_node=None, max_M=15, portal_k=3, depot=None):
    # [3-2. 최하위 군집 내부 경로 생성]
    if tree_node.is_leaf:
        path, _ = solve_base_dp(tree_node.nodes, start_node=start_node, end_node=end_node, depot=depot)
        return path

    K = len(tree_node.children)
    centroids = [c.centroid for c in tree_node.children]

    start_c_idx, end_c_idx = None, None
    if start_node:
        dists = [calc_dist(start_node, c) for c in centroids]
        start_c_idx = dists.index(min(dists))
    if end_node:
        dists = [calc_dist(end_node, c) for c in centroids]
        end_c_idx = dists.index(min(dists))
    if start_c_idx is not None and end_c_idx is not None and start_c_idx == end_c_idx and K > 1:
        end_c_idx = None

    depot_dists = None
    if depot and not start_node and not end_node:
        depot_dists = [min(calc_dist(depot, p) for p in child.nodes) for child in tree_node.children]

    dist_matrix = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            d = min_distance_pair(tree_node.children[i].nodes, tree_node.children[j].nodes)
            dist_matrix[i][j] = dist_matrix[j][i] = d

    ordered_c_indices = solve_cluster_order(dist_matrix, start_idx=start_c_idx, end_idx=end_c_idx, depot_dists=depot_dists)
    if not ordered_c_indices: return []

    # [3-1. 하위 군집 간의 포탈 후보 생성 (portal_k 개수만큼!)]
    entry_candidates, exit_candidates = {}, {}
    for i, c_idx in enumerate(ordered_c_indices):
        current_nodes = tree_node.children[c_idx].nodes
        
        if i == 0:
            if start_node: entry_list = [start_node]
            elif depot: entry_list = select_portal_candidates(current_nodes, [depot], portal_k)
            else: entry_list = list(current_nodes)
        else:
            prev_nodes = tree_node.children[ordered_c_indices[i - 1]].nodes
            entry_list = select_portal_candidates(current_nodes, prev_nodes, portal_k)

        if i == len(ordered_c_indices) - 1:
            if end_node: exit_list = [end_node]
            elif depot: exit_list = select_portal_candidates(current_nodes, [depot], portal_k)
            else: exit_list = list(current_nodes)
        else:
            next_nodes = tree_node.children[ordered_c_indices[i + 1]].nodes
            exit_list = select_portal_candidates(current_nodes, next_nodes, portal_k)

        entry_candidates[c_idx] = list(dict.fromkeys(entry_list))
        exit_candidates[c_idx] = list(dict.fromkeys(exit_list))

    # [4. 최하위 경로 확정 및 선택지 생성 (하위 트리 재귀 해결)]
    cluster_costs = {}
    for c_idx in ordered_c_indices:
        child_tree = tree_node.children[c_idx]
        for entry in entry_candidates[c_idx]:
            for exit in exit_candidates[c_idx]:
                if entry == exit and len(child_tree.nodes) > 1: continue
                # 포탈 후보 조합(최대 9개)마다 하위 경로 탐색
                path = solve_tree_hierarchy(child_tree, start_node=entry, end_node=exit, max_M=max_M, portal_k=portal_k)
                if path:
                    cluster_costs[(c_idx, entry, exit)] = (path, calc_path_distance(path))

    # [5. 최상위까지 올라오며 궤적 조립 (Stitching DP)]
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
# 6. 래퍼 함수 (전체 실행 흐름 통제)
# =====================================================================
def optimize_drilling_path(hole_positions, max_M=15, portal_k=3):
    depot = (0.0, 0.0)
    start_time = time.time()    
    
    # 1. 일단 트리 형태로 완벽하게 클러스터링을 최하위까지 진행합니다.
    full_tree = ClusterTree(hole_positions, max_M=max_M)
    
    # 2. 구축된 트리를 기반으로 최하위부터 조립하며 경로를 구합니다.
    pure_optimal_path = solve_tree_hierarchy(
        full_tree,
        start_node=None,
        end_node=None,
        max_M=max_M,
        portal_k=portal_k,
        depot=depot
    )
    
    # 3. 원점 시작, 복귀를 지키며 최종 경로를 완성합니다.
    full_path = [depot] + pure_optimal_path + [depot]
    total_dist = sum(calc_dist(full_path[i], full_path[i+1]) for i in range(len(full_path)-1))
    
    end_time = time.time()
    print("-" * 50)
    print(f"[Proposed] 최적화 완료! 총 이동 거리: {total_dist:.2f} mm")
    print_time(start_time, end_time)
    return full_path, total_dist