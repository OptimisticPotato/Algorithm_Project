from Utilities import get_cost

def greedy_tsp(holes, start_idx=0):
    unvisited = set(range(len(holes)))
    curr = start_idx
    path = [curr]
    unvisited.remove(curr)

    while unvisited:
        # 물리적 거리가 아니라 '최소 비용(Cost)'을 기준으로 다음 노드 선택
        next_node = min(unvisited, key=lambda x: get_cost(holes[curr], holes[x]))
        path.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
        
    return path