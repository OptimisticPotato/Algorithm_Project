import math
from sklearn.cluster import KMeans
from itertools import combinations
import time
from Utilities import get_distance, get_cost, Hole

# =====================================================================
# 1. 공통 유틸리티 함수 및 Hole 객체 호환
# =====================================================================
def calc_path_cost(path):
    """경로의 전체 비용(거리 + ATC 페널티)을 계산합니다."""
    if len(path) < 2: return 0.0
    return sum(get_cost(path[i], path[i + 1]) for i in range(len(path) - 1))

def find_node_index(node, nodes_list):
    """
    부동소수점 미세 오차로 인해 .index()가 점을 찾지 못하는 현상 방지.
    Hole 객체의 고유 id를 비교하여 완벽하게 안전한 인덱스를 반환합니다.
    """
    if node is None: return -1
    for idx, n in enumerate(nodes_list):
        if node.id == n.id:
            return idx
    return -1

# =====================================================================
# 2. [Base Solver] 만능 Held-Karp DP (상태 의존적 비용 적용)
# =====================================================================
def solve_base_dp(nodes, start_node=None, end_node=None, depot=None):
    n = len(nodes)
    if n == 1:
        cost_only = 0.0
        if depot: cost_only = get_cost(depot, nodes[0]) + get_cost(nodes[0], depot)
        return [nodes[0]], cost_only

    # 순수 거리가 아닌 get_cost(거리 + ATC 교체 비용) 적용
    dist = [[get_cost(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    
    start_idx = find_node_index(start_node, nodes)
    if start_idx == -1: start_idx = None
    
    end_idx = find_node_index(end_node, nodes)
    if end_idx == -1: end_idx = None
    
    memo = {}
    
    if start_idx is not None:
        memo[(1 << start_idx, start_idx)] = (0.0, -1)
    else:
        for i in range(n):
            if i != end_idx or n == 1:
                cost = get_cost(depot, nodes[i]) if depot else 0.0
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
                if depot: cost += get_cost(nodes[i], depot)
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
# 3. [Phase 1: Tree Builder] 이중 기준 재귀적 클러스터링 (Tool-First)
# =====================================================================
class ClusterTree:
    def __init__(self, nodes, max_M=15):
        self.nodes = nodes
        self.n = len(nodes)
        
        # 단일 공구인지 다중 공구인지 파악
        tool_types = list(set(h.tool for h in nodes))
        
        # 리프 노드 조건: 공구가 단일 종류이고, 노드 수가 max_M 이하일 때
        self.is_leaf = (len(tool_types) == 1 and self.n <= max_M)
        
        # Hole 객체의 x, y 속성을 이용하여 무게중심 계산
        self.centroid = (sum(h.x for h in nodes)/self.n, sum(h.y for h in nodes)/self.n) if self.n > 0 else (0,0)
        self.children = []
        
        if not self.is_leaf and self.n > 0:
            if len(tool_types) > 1:
                # [Tool-First] 공구 종류가 섞여있다면, 거리를 무시하고 무조건 공구별로 가지치기
                for t in tool_types:
                    t_nodes = [h for h in nodes if h.tool == t]
                    self.children.append(ClusterTree(t_nodes, max_M))
            else:
                # [Spatial-Second] 단일 공구 그룹인데 한계치를 넘었다면 K-Means 공간 분할
                K = min(max_M, self.n)
                coords = [[h.x, h.y] for h in nodes] # K-Means용 2D 배열
                kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(coords)
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
            # get_cost 적용: 다른 공구 트리면 ATC 페널티가 포함되어 행렬에 기록됨
            d = get_cost(u, v) 
            if d < min_dist: min_dist = d
    return min_dist

def select_portal_candidates(cluster_nodes, other_nodes, k):
    if not other_nodes: return list(cluster_nodes)
    scored = []
    for p in cluster_nodes:
        min_dist = min(get_cost(p, q) for q in other_nodes)
        scored.append((min_dist, p))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored[:k]] or list(cluster_nodes)

def solve_cluster_order(dist_matrix, start_idx=None, end_idx=None, depot_dists=None):
    # 기존과 논리 동일 (거리 행렬을 받아 DP 수행)
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
    if tree_node.is_leaf:
        path, _ = solve_base_dp(tree_node.nodes, start_node=start_node, end_node=end_node, depot=depot)
        return path

    K = len(tree_node.children)

    actual_start_c_idx = None
    if start_node:
        for idx, child in enumerate(tree_node.children):
            if find_node_index(start_node, child.nodes) != -1:
                actual_start_c_idx = idx
                break

    actual_end_c_idx = None
    if end_node:
        for idx, child in enumerate(tree_node.children):
            if find_node_index(end_node, child.nodes) != -1:
                actual_end_c_idx = idx
                break

    start_c_idx = actual_start_c_idx
    end_c_idx = actual_end_c_idx

    if start_c_idx is not None and end_c_idx is not None and start_c_idx == end_c_idx and K > 1:
        end_c_idx = None

    depot_dists = None
    if depot and not start_node and not end_node:
        depot_dists = [min(get_cost(depot, p) for p in child.nodes) for child in tree_node.children]

    dist_matrix = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            d = min_distance_pair(tree_node.children[i].nodes, tree_node.children[j].nodes)
            dist_matrix[i][j] = dist_matrix[j][i] = d

    ordered_c_indices = solve_cluster_order(dist_matrix, start_idx=start_c_idx, end_idx=end_c_idx, depot_dists=depot_dists)
    if not ordered_c_indices: return []

    entry_candidates, exit_candidates = {}, {}
    for i, c_idx in enumerate(ordered_c_indices):
        current_nodes = tree_node.children[c_idx].nodes
        
        # 진입 노드 처리
        if start_node and c_idx == actual_start_c_idx:
            if i == 0:
                entry_list = [start_node]
            else:
                prev_nodes = tree_node.children[ordered_c_indices[i - 1]].nodes
                entry_list = select_portal_candidates(current_nodes, prev_nodes, portal_k)
                if find_node_index(start_node, entry_list) == -1:
                    entry_list.append(start_node)
        else:
            if i == 0:
                if depot: entry_list = select_portal_candidates(current_nodes, [depot], portal_k)
                else: entry_list = list(current_nodes)
            else:
                prev_nodes = tree_node.children[ordered_c_indices[i - 1]].nodes
                entry_list = select_portal_candidates(current_nodes, prev_nodes, portal_k)

        # 진출 노드 처리
        if end_node and c_idx == actual_end_c_idx:
            if i == len(ordered_c_indices) - 1:
                exit_list = [end_node]
            else:
                next_nodes = tree_node.children[ordered_c_indices[i + 1]].nodes
                exit_list = select_portal_candidates(current_nodes, next_nodes, portal_k)
                if find_node_index(end_node, exit_list) == -1:
                    exit_list.append(end_node)
        else:
            if i == len(ordered_c_indices) - 1:
                if depot: exit_list = select_portal_candidates(current_nodes, [depot], portal_k)
                else: exit_list = list(current_nodes)
            else:
                next_nodes = tree_node.children[ordered_c_indices[i + 1]].nodes
                exit_list = select_portal_candidates(current_nodes, next_nodes, portal_k)

        entry_candidates[c_idx] = list(dict.fromkeys(entry_list))
        exit_candidates[c_idx] = list(dict.fromkeys(exit_list))

    cluster_costs = {}
    for c_idx in ordered_c_indices:
        child_tree = tree_node.children[c_idx]
        for entry in entry_candidates[c_idx]:
            for exit in exit_candidates[c_idx]:
                if entry == exit and len(child_tree.nodes) > 1: continue
                path = solve_tree_hierarchy(child_tree, start_node=entry, end_node=exit, max_M=max_M, portal_k=portal_k)
                if path:
                    cluster_costs[(c_idx, entry, exit)] = (path, calc_path_cost(path))

    first_idx = ordered_c_indices[0]
    dp, prev_map, first_prev = {}, [], {}

    for entry in entry_candidates[first_idx]:
        for exit in exit_candidates[first_idx]:
            key = (first_idx, entry, exit)
            if key not in cluster_costs: continue
            
            cost = cluster_costs[key][1]
            if depot and not start_node: cost += get_cost(depot, entry)
            
            if exit not in dp or cost < dp[exit]:
                dp[exit] = cost
                first_prev[exit] = (None, entry)

    prev_map.append(first_prev)

    for i in range(1, len(ordered_c_indices)):
        c_idx = ordered_c_indices[i]
        new_dp, new_prev = {}, {}

        for prev_exit, prev_cost in dp.items():
            for entry in entry_candidates[c_idx]:
                cross_cost = get_cost(prev_exit, entry) # 거리 대신 비용(페널티) 반영
                for exit in exit_candidates[c_idx]:
                    key = (c_idx, entry, exit)
                    if key not in cluster_costs: continue
                    cost = prev_cost + cross_cost + cluster_costs[key][1]
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
        if depot and not end_node: final_cost += get_cost(exit, depot)
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
def proposed_h_dp(hole_positions, max_cluster_size=10, num_portals=3):
    """
    메인 함수: TCP-TSP 구조에 맞춰 래퍼 함수 파라미터 이름을 통일시켰습니다.
    """
    if len(hole_positions) <= 1:
        return [h.id for h in hole_positions]
        
    depot = hole_positions[0] # 0번 노드를 항상 Depot(시작/종료점)으로 사용
    working_holes = hole_positions[1:] # 1번 노드부터 클러스터링 트리에 삽입
    
    start_time = time.perf_counter()

    full_tree = ClusterTree(working_holes, max_M=max_cluster_size)
    
    pure_optimal_path = solve_tree_hierarchy(
        full_tree,
        start_node=None,
        end_node=None,
        max_M=max_cluster_size,
        portal_k=num_portals,
        depot=depot
    )
    
    # 최종 결과는 0번으로 시작해서 0번으로 돌아오는 경로
    full_path = [depot] + pure_optimal_path
    
    # utilities.py와 main.py 호환을 위해 Hole 객체 리스트 대신 id 리스트로 변환하여 리턴
    final_id_path = [h.id for h in full_path]
    return final_id_path