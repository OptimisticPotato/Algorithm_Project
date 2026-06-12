import math
import random

# 공구 교체 페널티 (Constant Tool Change Penalty: T_ATC)
T_ATC = 15.0 
FEED  = 100.0 # mm/s

class Hole:
    def __init__(self, id, x, y, tool):
        self.id = id
        self.x = x
        self.y = y
        self.tool = tool

    def __repr__(self):
        return f"Hole({self.id}: {self.tool} @ {self.x},{self.y})"

def get_distance(h1, h2):
    """두 홀 사이의 순수 물리적 유클리드 거리를 계산합니다."""
    return math.hypot(h1.x - h2.x, h1.y - h2.y)

def get_cost(h1, h2):
    """
    [핵심] 상태 의존적 비용 계산 함수
    물리적 거리에 더해, 두 홀의 필요 공구가 다를 경우 T_ATC 페널티를 부과합니다.
    """
    cost = get_distance(h1, h2) / FEED  # 이동 시간으로 변환
    if h1.tool != h2.tool:
        cost += T_ATC
    return cost

def calculate_path_cost(path, holes, return_to_start=True):
    """
    주어진 경로(순서)의 전체 사이클 타임(물리적 이동 + 누적 T_ATC 페널티)을 계산합니다.
    """
    if not path:
        return 0.0
        
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += get_cost(holes[path[i]], holes[path[i+1]])
        
    if return_to_start and len(path) > 1:
        # 마지막 노드에서 다시 시작 노드로 돌아오는 비용 (TSP 순환)
        total_cost += get_cost(holes[path[-1]], holes[path[0]])
        
    return total_cost

def generate_random_holes(num_holes=10, width=100, height=100, tool_types=['M4', 'M8']):
    """
    무작위 좌표와 공구 종류(Tool type)를 가진 홀(Hole) 데이터 세트를 생성합니다.
    """
    holes = []
    for i in range(num_holes):
        x = round(random.uniform(0, width), 2)
        y = round(random.uniform(0, height), 2)
        tool = random.choice(tool_types)
        holes.append(Hole(id=i, x=x, y=y, tool=tool))
    return holes